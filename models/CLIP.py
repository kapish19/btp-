import torch
import torch.nn as nn
import numpy as np
from .clip import clip
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)


    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class DomainClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.bn1 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        out = self.fc1(x)
        if x.shape[0] == 1:
            out = out.repeat(2, 1)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.fc2(out)
            return F.log_softmax(out, dim=1)[0]
        else:
            out = self.bn1(out)
            out = self.relu(out)
            out = self.fc2(out)
            return out # Return raw logits for CrossEntropyLoss

class PartVisibilityGAT(nn.Module):
    def __init__(self, d_model=768, num_parts=3):
        super().__init__()
        self.num_parts = num_parts
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.vis_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        self.recon_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, patch_tokens, disable_part_branch=False):
        if disable_part_branch:
            # Return dummy zeros if disabled
            B = patch_tokens.shape[0]
            dummy_feat = torch.zeros(B, self.num_parts, patch_tokens.shape[-1], device=patch_tokens.device)
            dummy_vis = torch.zeros(B, self.num_parts, 1, device=patch_tokens.device)
            return dummy_feat, dummy_vis
            
        B, N, D = patch_tokens.shape
        part_size = N // self.num_parts
        
        # 3-way split (Head, Torso, Legs) -> [B, 3, part_size, D]
        parts = []
        for i in range(self.num_parts):
            start = i * part_size
            end = start + part_size if i < self.num_parts - 1 else N
            part_feat = patch_tokens[:, start:end, :].mean(dim=1) # [B, D]
            parts.append(part_feat)
        
        part_nodes = torch.stack(parts, dim=1) # [B, 3, D]
        
        # GAT
        Q = self.q_proj(part_nodes) # [B, 3, D]
        K = self.k_proj(part_nodes) # [B, 3, D]
        V = self.v_proj(part_nodes) # [B, 3, D]
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1) # [B, 3, 3]
        attended_nodes = torch.matmul(attn_weights, V) # [B, 3, D]
        
        # Visibility Prediction
        vis_scores = self.vis_predictor(part_nodes) # [B, 3, 1]
        
        # Internal Reconstruction Gate
        combined = torch.cat([part_nodes, attended_nodes], dim=-1) # [B, 3, 2D]
        gate = self.recon_gate(combined) # [B, 3, D]
        reconstructed = part_nodes * gate + attended_nodes * (1 - gate)
        
        # Masking by visibility
        part_features = reconstructed * vis_scores
        
        return part_features, vis_scores


class GradReverse(torch.autograd.Function):
    def __init__(self):
        super(GradReverse, self).__init__()

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None


class GRL(torch.nn.Module):
    def __init__(self, lambd=.1):
        super(GRL, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        lam = torch.tensor(self.lambd)
        return GradReverse.apply(x, lam)


class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_num, dtype, token_embedding):
        super().__init__()
        ctx_init = "A photo of a X X X X person."
        ctx_init_domain = "A photo of a X X X X person from X dataset."

        ctx_dim = 512
        n_ctx = 4

        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        tokenized_prompts = clip.tokenize(ctx_init).to(device)
        tokenized_prompts_domain = clip.tokenize(ctx_init_domain).to(device)

        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
            embedding_domain = token_embedding(tokenized_prompts_domain).type(dtype)

        self.tokenized_prompts = tokenized_prompts
        self.tokenized_prompts_domain = tokenized_prompts_domain

        n_cls_ctx = 4
        n_dm_ctx = 1
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        dom_vectors = torch.empty(dataset_num, n_dm_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(dom_vectors, std=0.02)
        # We only keep dmctx for orthogonality loss, no clsctx
        self.dmctx = nn.Parameter(dom_vectors, requires_grad=True)

        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def get_orthogonality_loss(self):
        # ‖ĜᵀĜ − I_K‖_F
        G = self.dmctx.squeeze(1) # [K, D]
        G_norm = F.normalize(G, p=2, dim=1)
        identity = torch.eye(G.shape[0], device=G.device)
        return torch.norm(torch.matmul(G_norm, G_norm.t()) - identity, p='fro')
        
    def forward(self, label, domain=None):
        return None # No prompt composition path in v1


class Model(nn.Module):
    def __init__(self, num_classes,args, epsilon=.1, domain_num=4):
        super(Model, self).__init__()
        self.h_resolution = int((args.size_train[0] - 16) // 16 + 1)
        self.w_resolution = int((args.size_train[1] - 16) // 16 + 1)
        self.vision_stride_size = 16
        self.model_name = args.backbone
        self.neck_feat = 'before'
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        if self.model_name == 'ViT-B-32':
            self.in_planes = 768
            self.in_planes_proj = 512
            self.h_resolution = int((args.size_train[0] - 32) // 32 + 1)
            self.w_resolution = int((args.size_train[1] - 32) // 32 + 1)
            self.vision_stride_size = 32
        if self.model_name == 'ViT-L-14':
            self.in_planes = 768
            self.in_planes_proj = 512
            self.h_resolution = int((args.size_train[0] - 14) // 14 + 1)
            self.w_resolution = int((args.size_train[1] - 14) // 14 + 1)
            self.vision_stride_size = 14
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        elif self.model_name == 'RN101':
            self.in_planes = 2048
            self.in_planes_proj = 512
        self.num_classes = num_classes
        self.grl = GRL(epsilon)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)
        
        # Local classifier
        self.local_classifier = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.local_classifier.apply(weights_init_classifier)

        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        # Use dynamic device
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        clip_model.to(device)
        
        # Domain Classifier for GRL on CLS token
        self.domain_classifier = DomainClassifier(self.in_planes, 128, domain_num)
        
        # Part branch
        self.part_gat = PartVisibilityGAT(d_model=self.in_planes, num_parts=3)
        
        self.image_encoder = clip_model.visual

        self.promptlearner = PromptLearner(num_classes, domain_num, clip_model.dtype, clip_model.token_embedding)

    def forward(self, x=None, label=None, get_image=False, get_text=False, cam_label=None, view_label=None,
                domain=None, prior=False, getdomain=False, grl_alpha=0.0, disable_grl=False, disable_part_branch=False):
        if get_text:
            return None, None # text pathway is disabled

        # Get visual features
        if "RN" in self.model_name:
            # RN not modified for patch_tokens in this baseline, sticking to ViT logic
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]
            patch_tokens = image_features.view(x.shape[0], image_features.shape[1], -1).permute(0, 2, 1)
        elif "ViT" in self.model_name:
            image_features_last, image_features, image_features_proj = self.image_encoder(x, None)
            # image_features is [B, N, D]
            img_feature = image_features[:, 0] # CLS token
            img_feature_proj = image_features_proj[:, 0]
            patch_tokens = image_features[:, 1:] # Patch tokens
            
        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)
        
        # Domain logits via GRL
        if disable_grl:
            domain_logits = torch.zeros(x.shape[0], self.domain_classifier.fc2.out_features, device=x.device)
        else:
            domain_logits = self.domain_classifier(self.grl(img_feature))
            
        # Part / occlusion branch
        part_features, vis_scores = self.part_gat(patch_tokens, disable_part_branch=disable_part_branch)
        
        # Process part features -> local_feat [B, 512] reusing self.image_encoder.proj? 
        # Design doc: local_feat = proj(part_features.sum(dim=1)) 
        part_sum = part_features.sum(dim=1)
        if self.image_encoder.proj is not None:
            local_feat_proj = part_sum @ self.image_encoder.proj
        else:
            local_feat_proj = part_sum
            
        local_feat = self.bottleneck_proj(local_feat_proj)

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            local_logits = self.local_classifier(local_feat)
            
            return {
                'id_logits': cls_score_proj, 
                'global_feat': feat_proj, 
                'local_logits': local_logits, 
                'local_feat': local_feat,
                'vis_scores': vis_scores, 
                'domain_logits': domain_logits,
                'ortho_loss': self.promptlearner.get_orthogonality_loss()
            }
        else:
            # Inference: final = normalize(g + l)
            final_feat = feat_proj + local_feat
            return final_feat


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
