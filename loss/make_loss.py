import torch
import torch.nn as nn
import torch.nn.functional as F
from .triplet_loss import TripletLoss
from .softmax_loss import CrossEntropyLabelSmooth

class TotalLoss(nn.Module):
    def __init__(self, num_classes, args):
        super().__init__()
        self.xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        self.triplet = TripletLoss(margin=0.3)
        self.bce = nn.BCELoss()
        self.domain_xent = nn.CrossEntropyLoss()
        
        self.lambda_id = 1.0
        self.lambda_tri = 1.0
        self.lambda_id_local = 0.5
        self.lambda_tri_local = 0.5
        self.lambda_occ = 0.3
        self.lambda_adv = 0.1
        self.lambda_ortho = 0.05

    def forward(self, outputs, target, occ_mask=None, domain_labels=None, stage=2, grl_alpha=0.0):
        # outputs is a dict from Model.forward
        id_logits = outputs['id_logits']
        global_feat = outputs['global_feat']
        local_logits = outputs['local_logits']
        local_feat = outputs['local_feat']
        vis_scores = outputs['vis_scores']
        domain_logits = outputs['domain_logits']
        ortho_loss = outputs['ortho_loss']

        # Global branch
        L_id = self.xent(id_logits, target)
        L_tri = self.triplet(global_feat, target)[0]

        # Part branch
        L_id_local = self.xent(local_logits, target)
        L_tri_local = self.triplet(local_feat, target)[0]
        
        if occ_mask is not None:
            # occ_mask is [B, 3, 1] for 3 parts
            L_occ_sup = self.bce(vis_scores, occ_mask)
        else:
            # Fallback if no mask (e.g. dummy test, or no occlusion)
            L_occ_sup = torch.tensor(0.0, device=target.device)

        L_ortho = ortho_loss
        
        loss = (self.lambda_id * L_id + 
                self.lambda_tri * L_tri + 
                self.lambda_id_local * L_id_local + 
                self.lambda_tri_local * L_tri_local + 
                self.lambda_occ * L_occ_sup + 
                self.lambda_ortho * L_ortho)

        if stage == 3 and domain_labels is not None:
            L_adv = self.domain_xent(domain_logits, domain_labels)
            loss = loss + self.lambda_adv * L_adv * grl_alpha
            
        return loss

def make_loss(num_classes, args=None):
    return TotalLoss(num_classes, args)
