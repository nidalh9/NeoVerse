# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Skills

ALWAYS use the following skills when applicable — do not skip them:

- **`/commit`** — Always use when creating git commits.
- **`/simplify`** — Always use after writing or modifying code to review for reuse, quality, and efficiency.
- **`/brainstorming`** — Always use before any creative work: creating features, building components, adding functionality, or modifying behavior.
- **`/writing-plans`** — Always use when you have a spec or requirements for a multi-step task, before touching code.
- **`/executing-plans`** — Always use when you have a written implementation plan to execute, with review checkpoints.
- **`/test-driven-development`** — Always use when implementing any feature or bugfix — write tests before implementation code.
- **`/systematic-debugging`** — Always use when encountering any bug, test failure, or unexpected behavior, before proposing fixes.
- **`/verification-before-completion`** — Always use before claiming work is complete, fixed, or passing — run verification and confirm output before making success claims.
- **`/requesting-code-review`** — Always use when completing tasks, implementing major features, or before merging.
- **`/dispatching-parallel-agents`** — Always use when facing 2+ independent tasks that can be worked on without shared state.
- **`/subagent-driven-development`** — Always use when executing implementation plans with independent tasks in the current session.

## Project Overview

NeoVerse is a 4D world model (CVPR 2026) that reconstructs 3D scenes from monocular videos and generates free-viewpoint video along custom camera trajectories. It combines a 3D reconstructor (WorldMirror or Depth Anything 3) with a WAN 2.1 video diffusion backbone controlled by a custom NeoVerse control branch.

## Commands

### Environment Setup
```bash
conda create -n neoverse python=3.10 -y && conda activate neoverse
# CUDA 12.1
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git
```

### Download Model Checkpoints
```bash
hf download Yuppie1204/NeoVerse --local-dir models/NeoVerse
```

### Run CLI Inference
```bash
# Predefined trajectory
python inference.py --input_path examples/videos/robot.mp4 --trajectory tilt_up --output_path outputs/out.mp4

# Custom trajectory from JSON
python inference.py --input_path examples/videos/movie.mp4 --trajectory_file examples/trajectories/orbit_left_pull_out.json --output_path outputs/out.mp4

# Validate trajectory without inference
python inference.py --trajectory_file my_trajectory.json --validate_only

# Low VRAM mode (reduces peak from ~74GB to ~38GB)
python inference.py --input_path video.mp4 --trajectory pan_left --low_vram --output_path outputs/out.mp4

# Full quality (50-step, no LoRA distillation)
python inference.py --input_path video.mp4 --trajectory pan_left --disable_lora --output_path outputs/out.mp4
```

### Run Gradio Web UI
```bash
python app.py
python app.py --low_vram
python app.py --reconstructor_path models/da3_giant_1.1.safetensors  # alternative reconstructor
```

## Architecture

### Entry Points
- **`inference.py`** — CLI inference script. Builds trajectory, loads pipeline, generates video.
- **`app.py`** — Gradio web UI with 4-step workflow: Upload → Reconstruct → Design Trajectory → Generate.

### Core Library: `diffsynth/`
The `diffsynth/` directory contains the full model and pipeline implementation (~280 Python files).

**Pipeline (`diffsynth/pipelines/wan_video_neoverse.py`)** — Main orchestrator `WanVideoNeoVersePipeline`. Processing is broken into sequential "Unit" classes:
1. `WanVideoUnit_ShapeChecker` → `WanVideoUnit_NoiseInitializer` → `WanVideoUnit_4DPreprocesser`
2. `WanVideoUnit_CameraProcesser` → `WanVideoUnit_RandomDrop` → `WanVideoUnit_4DEmbedder`
3. `WanVideoUnit_InputVideoEmbedder` → `WanVideoUnit_PromptEmbedder`
4. Diffusion loop with `WanVideoNeoVerseDiT` + `NeoVerseControlBranch`

**Models (`diffsynth/models/`):**
- `wan_video_dit.py` — 14B parameter diffusion transformer (WAN 2.1)
- `wan_video_vae.py` — Video VAE encoder/decoder
- `wan_video_text_encoder.py` — T5 UMT5-XXL text encoder
- `wan_video_neoverse_controller.py` — Control branch injected at 15 DiT block positions via Conv3d + attention

**Reconstructors (`diffsynth/auxiliary_models/`):**
- `worldmirror/` — Default reconstructor: finetuned on 3D/4D datasets, outputs Gaussian splats + camera poses
- `depth_anything_3/` — Alternative: DINOv2-based depth estimation, converts to pseudo Gaussian splats

**Utilities (`diffsynth/utils/`):**
- `auxiliary.py` — `CameraTrajectory` class (trajectory parsing, interpolation, predefined motions), `load_video()`, homogeneous matrix ops
- `app.py` — Gradio helpers: point cloud extraction from Gaussian splats, GLB scene export

**VRAM Management (`diffsynth/vram_management/`)** — CPU↔GPU model offloading via `AutoWrappedModule`/`AutoWrappedLinear` wrappers. Activated by `--low_vram` flag.

### Inference Flow
```
Input Video → Reconstructor → Gaussian Splats + Camera Poses
Target Trajectory → Rasterize Gaussians → RGB + Depth + Alpha Mask
Text Prompt → T5 Encoder → Embeddings
All conditioning → DiT + Control Branch → Diffusion (4 or 50 steps) → VAE Decode → Output Video
```

### Trajectory System
Camera paths can be specified four ways (see `docs/trajectory_format.md`):
1. **Predefined** — 13 built-in motions (`pan_left`, `orbit_right`, `push_in`, etc.)
2. **Keyframe operations JSON** — Compose operations at timestamp keyframes
3. **Sparse matrices JSON** — 4x4 camera-to-world matrices at sparse frame indices (SLERP interpolation)
4. **NPZ file reference** — External `.npz` with pre-computed matrices

Two trajectory modes: `relative` (composed with reconstructed camera) and `global` (absolute world-space).

### Coordinate System
OpenCV convention: +X right, +Y down, +Z forward. Camera-to-world matrices are 4x4 homogeneous. See `docs/coordinate_system.md`.

### Key Defaults
- Resolution: 560×336
- Output frames: 81 @ 16 FPS
- With LoRA: 4-step inference, cfg_scale=1.0
- Without LoRA: 50-step inference, cfg_scale=5.0
- Standard VRAM: ~47GB allocated, ~74GB peak
- Low VRAM: ~1GB allocated, ~38GB peak

### Model Checkpoints
Expected at `models/NeoVerse/` with DiT shards, T5 encoder, VAE, reconstructor, tokenizer config, and distilled LoRA. No build system — pure Python with runtime model loading.

### 4D Object Insertion (SDS Training)

NeoVerse is used as a video diffusion prior for 4D object insertion via Score Distillation Sampling.
The training code lives in the Deformable-3D-Gaussians repo at `train_neoverse_sds/`.

Three experimental approaches:
- **Experiment A**: GT-conditioned score matching — control branch sees GT composite video
- **Experiment B**: Generate-and-distill — full NeoVerse inference produces pseudo-GT targets
- **Experiment C**: Hybrid curriculum — A then B

See `docs/plans/2026-03-09-neoverse-sds-experiments-design.md` for full design.

Key entry points:
- `train_neoverse_sds/__main__.py` — CLI: `python -m train_neoverse_sds --experiment_mode A ...`
- `train_neoverse_sds/neoverse_sds/pipeline.py` — `NeoVerseSDS.full_inference()` for generate-and-distill
- `scripts/run_neoverse_sds.sh` (A), `run_neoverse_sds_B.sh` (B), `run_neoverse_sds_C.sh` (C)
