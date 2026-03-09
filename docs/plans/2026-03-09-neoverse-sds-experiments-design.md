# NeoVerse SDS for 4D Object Insertion — Experiment Design

**Date:** 2026-03-09
**Status:** Approved design, pending implementation plan

## Problem

Insert a new object (blue ball) into an existing 4D Gaussian Splatting scene (2 bouncing balls → 3 balls) using:
1. **Photometric loss** — L1 at view0 (base_azimuth=90°) against GT composited video
2. **NeoVerse SDS loss** — at novel views (180°, 270°, 0°) leveraging NeoVerse's video diffusion prior via its control branch

Previous attempts failed because:
- GT conditioning showed scene WITHOUT the ball → control branch fought insertion
- tau_max=0.80 (should be 0.98), sds_weight=0.1 (should be 1.0+), max_gaussian_scale=0.1 (should be 10.0)

## Core Insight: VDT-Style Conditioning

NeoVerse was trained on the VDT paradigm: take a **degraded** 3D reconstruction, feed it to the control branch, and the DiT generates a **clean** video conditioned on the structure. We leverage this for SDS by treating the 4DGS rendering as the "degraded" input and using GT or self-generated videos for conditioning.

## Three Experiments

### Experiment A: GT-Conditioned Score Matching

**Idea:** Use full composite GT videos (3 balls, with inserted object) as control branch conditioning. Score matching compares model's perception of rendered vs GT.

**Flow:**
```
Control branch: GT video (3 balls) — same for both queries
Query 1: v_rendered = DiT(noised_4DGS_render, τ, text, control=GT)
Query 2: v_gt      = DiT(noised_GT,           τ, text, control=GT)
Gradient: v_rendered - v_gt → vanishes when rendered ≈ GT
```

**Why it works:**
- On-distribution for NeoVerse (full scene Gaussian splat rendering as conditioning)
- GT has the ball → control branch works FOR the insertion
- Score matching gradient vanishes at optimum (no SDS drift)
- v_gt is "easy" (GT matches conditioning) → strong anchor

**Data:**
- GT conditioning: `data/3ball_original_4views/rgb/videos/view*.mp4` (full composite scene)
- Photometric GT: same folder, view at base_azimuth

### Experiment B: Generate-and-Distill (Self-Conditioned)

**Idea:** No GT needed. Render the current 4DGS scene, run full NeoVerse inference to generate a "clean" video, use that as a pseudo-GT target.

**Flow:**
```
1. Render full 4DGS scene (3 balls) from novel view → degraded video
2. Feed degraded video to NeoVerse control branch + text prompt
3. Run full NeoVerse inference (4 or 50 steps) → clean video
4. L1 loss: ||4DGS_rendering - clean_video.detach()|| → backprop to 4DGS
5. Re-generate pseudo-GT every N iterations (200-500)
```

**Why it works:**
- Uses NeoVerse's FULL generative capability (not just 1-step SDS gradient)
- Self-conditioning IS on-distribution (NeoVerse was trained on degraded reconstructions)
- Global optimization target (full video) vs local nudge (SDS)
- No GT videos required at all

**Risks:**
- Expensive: full inference per regeneration (~5-10s on A100)
- NeoVerse might remove the ball early in training if rendering is very poor
- Stochastic targets → use fixed seed per view to reduce noise

**Mitigation:** Fixed seed, regenerate infrequently (every 200-500 iters), cache result.

### Experiment C: Hybrid Curriculum

**Idea:** Start with GT score matching to establish correct ball position, then switch to generate-and-distill for quality refinement.

**Flow:**
```
Phase 1 (iter 0-2000): Experiment A — GT score matching
  → Establishes ball position, motion, and scene integration
Phase 2 (iter 2000-5000): Experiment B — Generate-and-distill
  → Refines quality using NeoVerse's full generative prior
```

**Why it works:**
- Phase 1 ensures the ball is correctly placed (strong GT signal)
- Phase 2 polishes using NeoVerse's generation (better than 1-step SDS)
- Ball rendering is good enough by iter 2000 that NeoVerse won't remove it

## Models

| Model | Size | Purpose |
|-------|------|---------|
| WAN 2.1 DiT (14B) | ~64GB (6 shards) | Main diffusion transformer |
| NeoVerseControlBranch | Part of DiT checkpoint | Structural conditioning via RGB + depth + cameras + mask |
| WAN VAE | ~6GB | Video encode/decode for latent-space operations |
| T5 UMT5-XXL | ~11GB | Text encoding |
| LoRA (rank 64) | ~631MB | 4-step distillation (optional speed vs quality) |

**Model path:** `{NEOVERSE_ROOT}/models/NeoVerse/`

## Data Paths

| Data | Path (relative to Deformable-3D-Gaussians/) | Content |
|------|------|---------|
| Full-scene GT (3 balls) | `data/3ball_original_4views/rgb/videos/view*.mp4` | Composite scene for Experiment A |
| Baseline checkpoint | `output/2BouncingBalls_baseline/` | Pretrained 2-ball scene |
| Inserted object PLY | `output/BlueBallMasked_debug/point_cloud/iteration_40000/point_cloud.ply` | Blue ball gaussians |
| Object-only GT (1 ball) | `data/1ball_original_4views/rgb/videos/inserted_obj_*_deg.mp4` | Available but NOT used (out-of-distribution for NeoVerse control branch) |

## Corrected Hyperparameter Defaults

| Parameter | Old (broken) | New (corrected) | Rationale |
|-----------|-------------|-----------------|-----------|
| `neoverse_tau_max` | 0.80 | 0.98 | Working Feb 16 run used 0.98 |
| `neoverse_sds_weight` | 0.1 | 1.0 | Working run used 5.0; start at 1.0 |
| `max_gaussian_scale` | 0.1 | 10.0 | 0.1 prevents scaling deformations |
| `lambda_deform_reg` | 0.01 | 0.01 | Already fixed (was 1.0) |

## Control Branch Inputs

NeoVerse's control branch takes 4 conditioning signals:

| Input | Shape | How we provide it |
|-------|-------|-------------------|
| RGB latents | [1, 16, T', H', W'] | VAE-encode the conditioning video (GT or self-rendered) |
| Depth latents | [1, 16, T', H', W'] | Zeros (depth not available from GT videos) |
| Camera embeddings | [1, T, 6, H, W] | Plucker rays from spherical camera poses |
| Mask | [1, T, 1, H, W] | All ones (fully valid) |

## Design Decision: Full-Scene Rendering Only

All three experiments use **full-scene rendering** (3 balls + background) for both the SDS input and conditioning. Object-only rendering is out-of-distribution for NeoVerse's control branch (trained on full scene reconstructions). Gradient hooks ensure only inserted-ball gaussian params get updated regardless.

## Files to Modify

| File | Change |
|------|--------|
| `train_neoverse_sds/config.py` | Add experiment mode, generate-and-distill fields, fix defaults |
| `train_neoverse_sds/trainer.py` | Add generate-and-distill path, curriculum switching |
| `train_neoverse_sds/neoverse_sds/sds_loss.py` | Keep score matching, add generate-and-distill mode |
| `train_neoverse_sds/neoverse_sds/pipeline.py` | Add `full_inference()` method for generation |
| `scripts/run_neoverse_sds.sh` | Three script variants (A/B/C) with correct params |
| `train_neoverse_sds/PLAN.md` | Replace with unified plan |
| `train_neoverse_sds/lessons.md` | Update with new approach |

## Files to Remove

| File | Reason |
|------|--------|
| `NeoVerse/docs/plans/2026-03-08-gt-video-neoverse-sds.md` | Superseded by this design |

## Files Unchanged

- `train_with_video_sds/` — all utilities reused as-is
- `NeoVerse/docs/plans/2026-03-05-multi-gpu-inference.md` — separate feature
- Photometric loss, regularization, rendering, data loading — same as before
