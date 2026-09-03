# DG-ReID: Final Architecture, Novelty, and Decisions Log

**Status:** This is the authoritative spec. It supersedes `OA-DG-ReID_Architecture.md`,
`complete_architecture.md`, and the current `DG_ReID_Colab.ipynb` wherever they disagree
with what's written here. Everything in this document is either (a) verified against the
actual notebook code as of this review, or (b) a decision made explicitly in this doc to
resolve a contradiction found during that review. Nothing here is aspirational — where a
component isn't built yet, it's labeled **NOT YET IMPLEMENTED**, not described as if it runs.

---

## 1. Problem Statement, Gap, and Positioning

**Base paper:** CILP-FGDI (Zhao, Qi, Geng — IEEE TIFS 2025) — CLIP-based fine-grained,
domain-generalizable person Re-ID via a single learnable domain-invariant/domain-relevant
text prompt and three-stage training.

**Gap CILP-FGDI leaves open:** it targets domain shift alone, assuming reasonably clean,
unoccluded pedestrian crops. Real surveillance introduces **domain shift and partial
occlusion simultaneously**, and the two interact — when part of a person is occluded, a
model that hasn't learned to discount unreliable regions tends to lean harder on
background/domain-specific cues, which is exactly the wrong failure mode for
generalization to an unseen domain.

**Related work checked (Aug 2026 pass):**
- Part-Aware Transformer for DG-ReID (ICCV 2023) — part-aware domain generalization,
  transformer-based, not CLIP. Closest architectural neighbor for the part branch; cite
  and differentiate explicitly (we use CLIP patch tokens + graph attention + visibility
  gating, not a parallel transformer stream).
- Normalization-based DG-ReID work (ECCV 2025) — addresses domain shift via feature
  normalization, orthogonal to our contribution; worth a related-work mention, not a
  competitor to the core claim.
- No paper found combining **CLIP-based domain generalization with zero-shot occlusion
  robustness** as a joint problem. This is the open niche — but this was checked with a
  handful of targeted searches, not a systematic literature review. **Action item, not yet
  done:** a proper systematic search (semantic scholar / arXiv full-text, not just web
  search) before this claim goes in a paper draft.

**Positioning statement for the paper:**
> A CLIP-based framework that learns domain-invariant global identity representations and
> visibility-aware local representations, using only synthetic occlusion for supervision,
> so that occlusion robustness generalizes zero-shot to an unseen domain (Occluded-Duke)
> without ever training on real occlusion labels.

---

## 2. Novelty — Final, Scoped List

Earlier drafts of this project claimed **six** simultaneous contributions (multi-granular
domain prompts, visual GRL, part visibility, GAT, weighted APN, occlusion reconstruction).
That is too many for one paper to isolate credibly, and — as of this review — three of
those six were never actually wired into training in the implementation. The scope is now
cut to what is real and defensible:

### Kept as core contributions:
1. **Domain-adversarial global branch.** CLS token → gradient-reversal layer → domain
   classifier, forcing the global identity feature to be domain-invariant. This is
   implemented and actively trains (verified: `losses['adv']` is computed and included in
   `TotalLoss.forward()` from stage 3 onward).
2. **Visibility-aware part branch with zero-shot occlusion supervision.** Patch tokens →
   part split → graph attention → per-part visibility prediction → visibility-weighted
   pooling, supervised **only** by synthetic occlusion masks generated on source-domain
   images. This was NOT wired into any loss in the reviewed notebook — see §7, Decision 3
   for the fix that makes this real.

### Kept as a minor regularizer, explicitly NOT oversold:
3. **K orthogonal domain tokens.** Used only for an orthogonality penalty during training
   and for unsupervised test-time adaptation (TT-DTA) against gallery patch statistics.
   Not connected to any text-prompt composition or hard-negative mining pathway. Framed in
   the paper as a lightweight domain-factor regularizer and a TT-DTA mechanism — not as
   "multi-granular domain-aware prompt learning," because that mechanism doesn't exist in
   this implementation (see §7, Decision 4).

### Cut entirely from this version:
4. **Distance-weighted APN (image-text hard-negative mining).** Fully coded
   (`DistanceWeightedAPNLoss`, EMA running stats, style distance) but never called from
   `TotalLoss`, and its config weights (`lambda_apn_global`, `lambda_apn_local`) are never
   read anywhere. Building it properly requires a prompt-composition step
   (ID token + domain token → CLIP text prompt, positive/negative construction per batch)
   that does not exist yet. **Decision: cut from v1** (§7, Decision 5).
5. **VisCoMA text conditioning.** Fully coded but structurally unreachable — the training
   loop never passes `text_tokens` to the model, so `viscoma()` is called zero times during
   training. **Decision: cut from v1** (§7, Decision 5).
6. **Feature reconstruction for heavily-occluded parts.** Kept as an internal gating
   mechanism inside `PartVisibilityGAT` (see §4.4) but not claimed as a standalone
   contribution — it's a implementation detail of the visibility-aware branch, not a
   separately ablatable module, until/unless a dedicated ablation shows it matters.

**Net result: two core contributions, not six.** This is a feature, not a compromise —
it's exactly the ablation-defensibility fix earlier feedback (both external and this
review) asked for, arrived at by cutting rather than by claiming more than the code does.

---

## 3. Full Architecture

```
                              INPUT
                       Image [B, 3, 256, 128]
                              │
              ┌───────────────────────────────┐
              │        CLIPBackbone            │
              │  CLIP ViT-B/16, last 2 of 12    │
              │  resblocks trainable, rest      │
              │  frozen. Text encoder frozen.   │
              │  Positional embedding BICUBIC-  │
              │  INTERPOLATED to 16x8 grid      │
              │  (fixed — see §7 Decision 1)    │
              └───────────────┬─────────────────┘
                               │
                 cls_token [B,768]   patch_tokens [B,128,768]
                               │                │
              ┌────────────────┘                └───────────────┐
              │                                                  │
   ┌──────────▼──────────┐                          ┌────────────▼────────────┐
   │  GLOBAL BRANCH        │                          │  PART / OCCLUSION BRANCH │
   │  proj(768→512)        │                          │  PartVisibilityGAT       │
   │  BNNeck                │                          │  - 3-way split (H/T/L)  │
   │  classifier            │                          │  - Q/K/V graph attn      │
   │  → id_logits            │                          │  - visibility predictor  │
   │                         │                          │  - synthetic-occlusion   │
   │  VisualDomainClassifier│                          │    BCE supervision       │
   │  (GRL, stage 3 only)    │                          │    (NEW — §7 Decision 3) │
   │  → domain_logits         │                          │  - recon gate (internal) │
   └──────────┬──────────────┘                          │  → part_features,        │
              │                                          │     vis_scores           │
              │                                          └────────────┬─────────────┘
              │                                                       │
              │                                          NOW WIRED INTO LOSS
              │                                          (id_local + tri_local +
              │                                           occ_sup — §7 Decision 3)
              │                                                       │
              └───────────────────────┬───────────────────────────────┘
                                       │
                          global_feat (g) + local_feat (l)
                                       │
                              final = normalize(g + l)
                                       │
                              USED FOR RETRIEVAL / MATCHING

   ┌─────────────────────────────────────────────────────────────┐
   │  MultiGranularDomainPrompts (K=4, 512-d)                      │
   │  - orthogonality loss (always active, minor regularizer)      │
   │  - TT-DTA: unsupervised adaptation vs. gallery patch stats     │
   │  - NOT connected to any text-prompt / APN pathway (cut, §2)    │
   └─────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────┐
   │  VisCoMA + DistanceWeightedAPNLoss — CUT FROM v1 (§2, §7-5)    │
   │  Code retained in repo, disabled, not part of the paper story │
   └─────────────────────────────────────────────────────────────┘
```

### 3.1 Data flow, restated as pseudocode (matches the fixed implementation)

```python
# forward pass, training mode
cls_token, patch_tokens = backbone.encode_image(images)          # [B,768], [B,128,768]

# --- global branch (unchanged, already correct) ---
cls_proj   = backbone.proj(cls_token)                             # [B,512]
bn_feat    = bn_neck(cls_proj)                                    # [B,512]
id_logits  = classifier(bn_feat)                                  # [B,N_cls]
domain_logits = domain_classifier(cls_token, grl_alpha)           # [B,N_domains]

# --- part / occlusion branch (NOW WIRED, was previously dead) ---
part_features, vis_scores = part_gat(patch_tokens)                # [B,3,768], [B,3,1]
local_feat   = proj(part_features.sum(dim=1))                     # [B,512] -- reuse global proj
local_logits = local_classifier(local_feat)                       # [B,N_cls] -- NEW head, see §7-3

# --- domain tokens (unchanged) ---
ortho_loss = domain_prompts.get_orthogonality_loss()              # scalar
```

---

## 4. Module-by-Module Specification

### 4.1 CLIPBackbone
- Base: OpenAI CLIP ViT-B/16, loaded via `clip.load('ViT-B/16')`.
- Frozen: `conv1`, `class_embedding`, `ln_pre`, resblocks `[0:10]`, `ln_post`, entire text
  encoder (`transformer`, `token_embedding`).
- Trainable: resblocks `[10:12]` (~14.2M params), new `proj: Linear(768→512)` (393K params).
- Input `[B,3,256,128]` → patch grid **16 rows × 8 cols = 128 patches** (not 196 — that
  would be the 14×14 grid from the original 224×224 pretraining resolution).
- **Positional embedding: bicubic-interpolated from the pretrained 14×14 grid to the
  actual 16×8 grid** (§7 Decision 1 — this replaces the previous naive `[:129]` truncation,
  which misaligned spatial position for every patch).

### 4.2 MultiGranularDomainPrompts
- `K=4` learnable tokens, `[4, 512]`, init `randn() * 0.02`.
- `get_orthogonality_loss()`: `‖ĜᵀĜ − I_K‖_F` where `Ĝ` is row-L2-normalized.
- Referred to as **K latent domain factors** everywhere in the paper and code comments —
  no claim that any token maps to a specific human-nameable axis (illumination, camera,
  resolution) unless a dedicated probing experiment is run and reported (unchanged from
  the previous fix; restated here as still binding).
- Consumed by: (a) orthogonality loss during training, (b) TT-DTA at test time. Not
  consumed by anything else — see §2 for why the "domain-aware prompt" framing is cut.

### 4.3 VisualDomainClassifier + GRL
- `Linear(768→384) → BatchNorm1d → ReLU → Linear(384→N_domains)`, ~297K params.
- Gradient reversal: forward is identity, backward negates and scales gradient by `α`.
- `α` warmup: `α(e) = min(1, (e − 30) / 10) × 1.0`, active stage 3 only (epochs 31–60).
- **Verified active in training** — this is one of the two real contributions.

### 4.4 PartVisibilityGAT
- Patch tokens `[B,128,768]` → 3-way split (head/torso/legs) via `N // 3` **dynamic**
  partition (correct — adapts to actual patch count, unlike the earlier fixed-index
  `0:70/70:140/140:` slicing written for a 196-patch assumption).
- Q/K/V graph attention across the 3 parts (`[B,3,3]` attention matrix).
- Visibility predictor: `Linear(768→384) → ReLU → Linear(384→1) → Sigmoid` → `vis_scores
  ∈ [0,1]`.
- Internal reconstruction gate: parts with `vis < 0.4` get blended with the graph-attended
  representation via a learned sigmoid gate; parts above threshold pass through unchanged.
  This stays an internal detail of the module, not a separate contribution (§2).
- Output: `reconstructed × vis_scores` → `part_features [B,3,768]`.
- **Newly added: `occ_sup_head`**, a `BCE` target consumer that supervises `vis_scores`
  against the synthetic occlusion mask (mean-pooled to 3 parts) — see §7 Decision 3 for
  the full loss wiring. This is what makes "occlusion-aware" a true claim instead of an
  aspirational one.

### 4.5 SyntheticOcclusionAugment *(carried over from the previous fix, restated as binding)*
- Applied only to source-domain training images (Market/MSMT/CUHK*).
- Pastes a random rectangular or occluder-bank patch onto a clean image; returns the
  corrupted image plus the **exact** patch-grid ground-truth mask.
- Occluded-Duke images and masks never enter this function or any other part of training.
- This mask is mean-pooled per body-part region and used as the `occ_sup` BCE target
  (§4.4, §7 Decision 3) — closing the loop between "we claim zero-shot occlusion
  generalization" and "here is the only supervision the occlusion head ever sees."

### 4.6 VisCoMA, DistanceWeightedAPNLoss — retained in codebase, disabled
- Both remain as dead/disabled code for a possible v2 extension (§7 Decision 5), clearly
  marked `# DISABLED IN v1 — see architecture doc §2, §7-5` at the top of each class.
- Not described anywhere else in the paper draft as active.

### 4.7 BNNeck
- `BatchNorm1d(512)`, bias frozen at 0, weight init `Normal(1.0, 0.02)`. Standard
  norm-decoupling trick from strong Re-ID baselines (Luo et al., BoT). Unchanged.

---

## 5. Design Decisions and Rationale

| Decision | Why |
|---|---|
| CLIP ViT-B/16 backbone, last 2 blocks trainable | CILP-FGDI's base is CLIP; keeping the backbone choice makes this a legible extension rather than a different paper. Freezing most of the backbone keeps the trainable parameter count (~18.2M of ~151M, ~12%) low enough to be feasible on limited compute — full fine-tuning of CLIP is not affordable on a BTP budget. |
| Domain-adversarial GRL on the CLS token, not the patch tokens | The global identity feature is what carries cross-domain style leakage most directly (background, camera characteristics baked into the CLS summary). Applying GRL to patch tokens as well was considered and dropped — it would fight the part branch's need to preserve *some* patch-level information for occlusion reasoning. |
| 3-way body-part split, dynamic `N // 3` | Coarse enough to be robust to imperfect person detection/cropping, fine-grained enough to let head/torso/leg occlusion be handled separately. A learned soft part-attention (vs. fixed split) was considered (external feedback's Option B) but deferred — it's a real improvement but adds another trainable module and another thing to ablate, which conflicts with the "cut to two contributions" decision. Candidate for v2. |
| Synthetic-only occlusion supervision | The zero-shot generalization claim is the paper's central experimental story (train without ever seeing Occluded-Duke, test directly on it). Any leakage of real occlusion labels into training breaks that claim outright — this was the single most severe issue in the first review pass and is treated as a hard constraint, not a preference. |
| K=4 orthogonal domain tokens, kept as minor regularizer rather than cut entirely | Even without a text-prompt pathway, cheap orthogonality regularization plus a free TT-DTA mechanism (2,048 params, ~5 seconds to adapt) costs almost nothing to keep and gives a legitimate, small test-time-adaptation result to report. Cutting it entirely would remove a working, low-risk piece of the paper for no benefit. |
| APN + VisCoMA cut from v1 | Both require infrastructure that doesn't exist yet (prompt composition, positive/negative text construction) and were half-built in a way that silently produced a paper/code mismatch. Building them properly is a real v2 extension, not a v1 patch — see §7 Decision 5 for the full reasoning. |
| Bicubic interpolation of positional embeddings instead of truncation | CLIP's positional embeddings are spatially meaningful (learned for a specific 14×14 raster layout). Truncating to the first 129 rows silently reassigns each patch's position embedding to the wrong spatial location for a 16×8 grid. Interpolation is the standard fix used by TransReID and most ViT-based Re-ID work repurposing pretrained position embeddings for a different aspect ratio. |
| Two contributions instead of six | Ablation defensibility. A reviewer's first question for a stacked-module paper is "which component did the work" — six modules means six confounds. Two contributions, each independently ablatable and each verified to actually receive gradient during training, is a paper that survives that question. |

---

## 6. Loss Functions — Final

```text
L_id      = CrossEntropyLabelSmooth(id_logits, pids; eps=0.1)          # global branch
L_tri     = TripletLoss(cls_features, pids; margin=0.3)                # global branch, hard mining

L_id_local  = CrossEntropyLabelSmooth(local_logits, pids; eps=0.1)     # NEW — part branch, §7-3
L_tri_local = TripletLoss(local_feat, pids; margin=0.3)                # NEW — part branch, §7-3
L_occ_sup   = BCE(vis_scores, synthetic_occ_mask_per_part)             # NEW — part branch, §7-3
                                                                         # synthetic_occ_mask comes
                                                                         # ONLY from SyntheticOcclusionAugment
                                                                         # on source-domain images

L_adv     = CrossEntropy(domain_logits, domain_labels)                 # stage 3 only, GRL-reversed
L_ortho   = ‖ĜᵀĜ − I_K‖_F                                              # domain token regularizer

L_total = λ_id · L_id + λ_tri · L_tri
        + λ_id_local · L_id_local + λ_tri_local · L_tri_local + λ_occ · L_occ_sup
        + λ_adv · L_adv   (stage 3 only)
        + λ_ortho · L_ortho
```

| Weight | Symbol | Default | Active |
|---|---|---|---|
| `lambda_id` | λ_id | 1.0 | all stages |
| `lambda_tri` | λ_tri | 1.0 | all stages |
| `lambda_id_local` | λ_id_local | 0.5 *(new, needs tuning)* | stage 2+ (once part branch is unfrozen) |
| `lambda_tri_local` | λ_tri_local | 0.5 *(new, needs tuning)* | stage 2+ |
| `lambda_occ` | λ_occ | 0.3 *(new, needs tuning)* | all stages |
| `lambda_adv` | λ_adv | 0.1 | stage 3 only |
| `lambda_ortho` | λ_ortho | 0.05 | all stages |

`DistanceWeightedAPNLoss` and its `lambda_apn_global`/`lambda_apn_local` weights are
**removed from `TotalLoss`** in v1 — not set to zero, actually removed, so there's no
dead config to confuse a future reader.

---

## 7. Decisions Made in This Review (Change Log)

These are binding until explicitly revisited. Each was made to close a specific gap found
by reading the actual code, not by inspecting the documentation alone.

1. **Positional embeddings are bicubic-interpolated to the actual patch grid, not
   truncated.** Truncation silently misaligns spatial position for every patch in a
   16×8 grid built from position embeddings meant for 14×14. Fixed in §4.1.

2. **Occlusion supervision remains synthetic-only, Occluded-Duke masks remain eval-only.**
   (Carried over from the previous architecture review — restated here as still binding,
   now cross-checked against the actual notebook: confirmed the notebook currently has
   *no* occlusion supervision loss at all, synthetic or otherwise — `RandomErasing` is
   applied as a plain augmentation with no mask ever recorded. §7 Decision 3 fixes this.)

3. **`PartVisibilityGAT` is wired into the loss.** As reviewed, `part_features`/
   `vis_scores` were computed every forward pass and never consumed by any loss — zero
   gradient ever reached the module during training, meaning it ran at inference with
   effectively random weights blended into the retrieval feature. Fix: add
   `local_classifier` head, `L_id_local`, `L_tri_local` on the visibility-pooled part
   feature, and `L_occ_sup` (BCE against `SyntheticOcclusionAugment`'s ground-truth mask,
   mean-pooled per part). This is the single most important fix in this document — without
   it, "occlusion-aware" was not a true claim about the trained model.

4. **Domain tokens are downgraded from "multi-granular domain-aware prompt learning" to
   "orthogonal domain-factor regularizer + TT-DTA mechanism."** No prompt-composition
   pathway exists in the code, and building one is out of scope for v1. The paper's
   contribution list is written to match this, not the other way around.

5. **VisCoMA and `DistanceWeightedAPNLoss` are cut from v1, not silently left dead.**
   Both were fully coded but structurally unreachable (VisCoMA: training loop never
   passes `text_tokens`; APN: never called from `TotalLoss`, its config weights never
   read). Rather than leaving this as an undocumented gap between the paper's claims and
   the code — which is what the reviewed PDF was doing — the decision is to explicitly
   cut both from the v1 contribution list, mark the code as disabled, and treat them as a
   named v2 extension requiring real prompt-composition infrastructure.

6. **Contribution count is locked at two: domain-adversarial global branch + visibility-
   aware occlusion-supervised part branch.** Both are now verified real (branch 1 was
   already real; branch 2 is real as of Decision 3). Nothing else is claimed as a
   contribution in the current draft.

---

## 8. Training Protocol (unchanged in structure, losses updated per §6)

| Stage | Epochs | Backbone | Active heads | Active losses |
|---|---|---|---|---|
| 1 — Prompt/head learning | 1–10 | frozen | proj, bn_neck, classifier, part_gat, local_classifier, domain_prompts | `L_id + L_tri + L_id_local + L_tri_local + L_occ_sup + L_ortho` |
| 2 — Partial fine-tuning | 11–30 | blocks 10–11 unfrozen | all heads | same as stage 1 |
| 3 — Full pipeline + GRL | 31–60 | blocks 10–11 | all heads + domain_classifier | all of the above + `L_adv` (α warmup 0→1 over epochs 31–40) |

PK sampling: P=16 identities × K=4 images, batch size 64, multi-domain identity mixing
across Market-1501 (domain 0) and MSMT17 (domain 1). Occluded-Duke is query/gallery only,
never in `train_loader`.

---

## 9. Evaluation Protocol

- Standard: mAP, CMC Rank-1/5/10, same-camera exclusion.
- Occluded-Duke-specific: additionally report Rank-1 on the heavily-occluded subset
  (visibility ratio < 0.5 by the dataset's real annotations), used **strictly as a
  post-hoc filter for this reported breakdown**, never as a training or model-selection
  signal.
- Ablation table (locked before further code changes):

| Model | GRL (domain-adv) | Visibility part branch + occ_sup | Result |
|---|---|---|---|
| CILP-FGDI (baseline, reproduced) | ✗ | ✗ | *(reproduce first, before anything else)* |
| + GRL only | ✓ | ✗ | |
| + Visibility part branch only | ✗ | ✓ | |
| **Full (ours)** | ✓ | ✓ | |

If the full model's gain over each single-component row isn't clearly additive, that's a
finding to report honestly, not a result to suppress.

---

## 10. Known Limitations / Open Risks

- **Compute and timeline budget still unstated.** Every recommendation above assumes a
  BTP-scale budget (not a funded lab's). The actual GPU-hours and deadline available have
  not been provided in this conversation and materially affect whether even the
  two-contribution scope above is achievable in time to reproduce the CILP-FGDI baseline,
  run the new part-branch training, and complete the ablation table.
- **Related-work check was shallow.** A handful of targeted web searches, not a systematic
  literature search. The novelty claim in §1 should be treated as provisional until that's
  done properly.
- **Loss weights for the new local branch (`λ_id_local`, `λ_tri_local`, `λ_occ`) are
  placeholders**, not tuned values — they need a real hyperparameter sweep, not the
  defaults written in §6.
- **Soft/learned part attention (vs. the fixed 3-way split) remains a deferred
  improvement**, not implemented, candidate for v2 if the fixed split's ablation shows a
  ceiling effect.
- **No results have been generated with this fixed architecture yet.** Any numbers from
  earlier runs of the pre-fix notebook describe a different, effectively simpler model
  (CLIP-ReID + GRL) and should not be reported as measuring the occlusion-aware branch.
