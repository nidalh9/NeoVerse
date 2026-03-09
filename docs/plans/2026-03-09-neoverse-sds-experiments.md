# NeoVerse SDS Experiments Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement three NeoVerse SDS experiments for 4D object insertion: (A) GT-conditioned score matching, (B) generate-and-distill, (C) hybrid curriculum. Fix broken hyperparameter defaults. Update documentation.

**Architecture:** The training loop renders a full 4DGS scene (3 balls) from novel views. Experiment A uses GT videos as control branch conditioning with score matching (v_rendered - v_gt). Experiment B runs full NeoVerse inference on the degraded rendering to produce a pseudo-GT target. Experiment C starts with A then switches to B at a configurable iteration.

**Tech Stack:** PyTorch, Deformable 3D Gaussians, NeoVerse (WAN 2.1 14B DiT + control branch), wandb, SLURM

**Critical Rule:** NEVER modify files outside `train_neoverse_sds/` and `scripts/` in the Deformable-3D-Gaussians repo (except PLAN.md and lessons.md in train_neoverse_sds/). Reuse existing utilities from `train_with_video_sds/` as-is. For NeoVerse repo, only modify `CLAUDE.md` and `docs/plans/`.

---

## Dependency Graph

```
Task 1 (config) ──┐
                   ├──→ Task 2 (pipeline.full_inference) ──→ Task 3 (sds_loss modes) ──┐
                   │                                                                     │
                   ├──→ Task 4 (trainer.py rewrite) ←───────────────────────────────────┘
                   │
                   ├──→ Task 5 (__main__.py) ─ independent
                   ├──→ Task 6 (scripts) ─ independent
                   └──→ Task 7 (docs cleanup) ─ independent
```

- Task 1 is prerequisite for all others
- Task 2 can start after Task 1
- Task 3 depends on Task 2
- Task 4 depends on Tasks 1+3
- Tasks 5, 6, 7 are independent of each other, can be done after Task 1

---

### Task 1: Update config.py — add experiment mode, fix defaults

**Files:**
- Modify: `train_neoverse_sds/config.py`

**Step 1: Add new fields to NeoVerseSDSConfig dataclass**

After line 80 (`neoverse_sds_mode`), add:

```python
    # ── Experiment mode (A/B/C) ─────────────────────────────────────────
    experiment_mode: str = "A"  # "A" = GT score matching, "B" = generate-and-distill, "C" = hybrid

    # ── Generate-and-distill (Experiment B) ─────────────────────────────
    distill_interval: int = 200       # Re-generate pseudo-GT every N iterations
    distill_num_steps: int = 4        # NeoVerse inference steps for generation
    distill_seed: int = 42            # Fixed seed for stable generation
    distill_loss_type: str = "l1"     # Loss type for distill: "l1" or "lpips"

    # ── Hybrid curriculum (Experiment C) ────────────────────────────────
    curriculum_switch_iter: int = 2000  # Switch from A→B at this new_model_iteration

    # ── GT video folder for SDS conditioning (Experiment A) ─────────────
    sds_gt_video_folder: Optional[str] = None  # Separate GT for SDS (3-ball composite)
```

**Step 2: Fix broken hyperparameter defaults**

Change these existing defaults in the dataclass:
- Line 27: `max_gaussian_scale: float = 0.1` → `max_gaussian_scale: float = 10.0`
- Line 94: `neoverse_tau_max: float = 0.80` → `neoverse_tau_max: float = 0.98`
- Line 95: `neoverse_sds_weight: float = 0.1` → `neoverse_sds_weight: float = 1.0`

**Step 3: Add new arguments to `add_neoverse_sds_arguments()`**

After the `--neoverse_sds_mode` argument (line 184), add:

```python
    parser.add_argument("--experiment_mode", type=str, default="A",
                        choices=["A", "B", "C"],
                        help="Experiment: A=GT score matching, B=generate-and-distill, C=hybrid (default: A)")
    parser.add_argument("--distill_interval", type=int, default=200,
                        help="Re-generate pseudo-GT every N iters in experiment B/C (default: 200)")
    parser.add_argument("--distill_num_steps", type=int, default=4,
                        help="NeoVerse inference steps for generate-and-distill (default: 4)")
    parser.add_argument("--distill_seed", type=int, default=42,
                        help="Fixed seed for stable generation in distill mode (default: 42)")
    parser.add_argument("--distill_loss_type", type=str, default="l1",
                        choices=["l1", "lpips"],
                        help="Loss for distill mode (default: l1)")
    parser.add_argument("--curriculum_switch_iter", type=int, default=2000,
                        help="Switch from A→B at this iteration in experiment C (default: 2000)")
    parser.add_argument("--sds_gt_video_folder", type=str, default=None,
                        help="GT video folder for SDS conditioning (3-ball composite, experiment A/C)")
```

Update the `--neoverse_sds_mode` choices to include new mode:

```python
    parser.add_argument("--neoverse_sds_mode", type=str, default="neoverse_sds",
                        choices=["neoverse_sds", "neoverse_score_matching", "generate_and_distill"],
                        help="NeoVerse SDS mode (default: neoverse_sds)")
```

Also fix the defaults for the three broken params in the argparser:
- `--neoverse_tau_max` default: `0.80` → `0.98`
- `--neoverse_sds_weight` default: `0.1` → `1.0`
- `--max_gaussian_scale` default: `0.1` → `10.0`

**Step 4: Update `build_neoverse_sds_config()` to include new fields**

Add to the `NeoVerseSDSConfig(...)` constructor call:

```python
        # Experiment mode
        experiment_mode=getattr(args, 'experiment_mode', 'A'),
        distill_interval=getattr(args, 'distill_interval', 200),
        distill_num_steps=getattr(args, 'distill_num_steps', 4),
        distill_seed=getattr(args, 'distill_seed', 42),
        distill_loss_type=getattr(args, 'distill_loss_type', 'l1'),
        curriculum_switch_iter=getattr(args, 'curriculum_switch_iter', 2000),
        sds_gt_video_folder=getattr(args, 'sds_gt_video_folder', None),
```

---

### Task 2: Add `full_inference()` method to pipeline.py

**Files:**
- Modify: `train_neoverse_sds/neoverse_sds/pipeline.py`

**Step 1: Add the `full_inference()` method to `NeoVerseSDS` class**

Add after the `sample_timestep()` method (after line 256):

```python
    def full_inference(
        self,
        conditioning: dict,
        context_pos: torch.Tensor,
        context_neg: torch.Tensor,
        cfg_scale: float = 1.0,
        control_scale: float = 1.0,
        num_inference_steps: int = 4,
        seed: int = 42,
        height: int = 336,
        width: int = 560,
        num_frames: int = 17,
    ) -> torch.Tensor:
        """Run full NeoVerse denoising inference to generate a clean video.

        Used by the generate-and-distill approach (Experiment B).
        Takes conditioning (from control branch) and generates a complete video
        through the full denoising loop.

        Args:
            conditioning: Dict from VideoConditioningCache.get_conditioning() with keys
                rgb_latents, depth_latents, camera_embed, mask.
            context_pos: Positive prompt embeddings [B, seq_len, dim].
            context_neg: Negative prompt embeddings [B, seq_len, dim].
            cfg_scale: Classifier-free guidance scale.
            control_scale: Scaling factor for control branch output.
            num_inference_steps: Number of denoising steps.
            seed: Random seed for reproducible generation.
            height: Output video height (must match NeoVerse training res).
            width: Output video width (must match NeoVerse training res).
            num_frames: Number of output frames (must be 4k+1 for VAE).

        Returns:
            Generated video tensor [T, H, W, 3] in [0, 1], detached.
        """
        # Set up scheduler for the given number of steps
        self._scheduler.set_timesteps(num_inference_steps)

        # Compute latent dimensions
        T_latent = (num_frames + 3) // 4
        H_latent = height // 8
        W_latent = width // 8
        C_latent = 16  # VAE z_dim

        # Initialize random noise with fixed seed
        generator = torch.Generator(device=self._device).manual_seed(seed)
        latents = torch.randn(
            (1, C_latent, T_latent, H_latent, W_latent),
            generator=generator,
            dtype=self._torch_dtype,
            device=self._device,
        )

        # Extract conditioning tensors
        target_rgb = conditioning["rgb_latents"].to(device=self._device, dtype=self._torch_dtype)
        target_depth = conditioning["depth_latents"].to(device=self._device, dtype=self._torch_dtype)
        target_camera_embed = conditioning["camera_embed"].to(device=self._device, dtype=self._torch_dtype)
        target_mask = conditioning["mask"].to(device=self._device, dtype=self._torch_dtype)

        # Denoising loop
        self._pipe.load_models_to_device(("dit", "control_branch"))
        for step_idx, timestep in enumerate(self._scheduler.timesteps):
            timestep_tensor = timestep.unsqueeze(0).to(dtype=self._torch_dtype, device=self._device)

            # Positive prediction (with conditioning)
            noise_pred_pos = model_fn_wan_video(
                dit=self.dit,
                control_branch=self.control_branch,
                latents=latents,
                timestep=timestep_tensor,
                context=context_pos,
                control_scale=control_scale,
                target_rgb=target_rgb,
                target_depth=target_depth,
                target_camera_embed=target_camera_embed,
                target_mask=target_mask,
            )

            if cfg_scale != 1.0:
                noise_pred_neg = model_fn_wan_video(
                    dit=self.dit,
                    control_branch=self.control_branch,
                    latents=latents,
                    timestep=timestep_tensor,
                    context=context_neg,
                    control_scale=control_scale,
                    target_rgb=target_rgb,
                    target_depth=target_depth,
                    target_camera_embed=target_camera_embed,
                    target_mask=target_mask,
                )
                noise_pred = noise_pred_neg + cfg_scale * (noise_pred_pos - noise_pred_neg)
            else:
                noise_pred = noise_pred_pos

            # Scheduler step
            latents = self._scheduler.step(noise_pred, timestep, latents)

        self._pipe.load_models_to_device([])

        # Decode latents to pixel space
        self._pipe.load_models_to_device(["vae"])
        video = self.vae.decode(latents, device=self._device)
        self._pipe.load_models_to_device([])

        # video is [1, 3, T, H, W] in [-1, 1] → [T, H, W, 3] in [0, 1]
        video = video.squeeze(0).permute(1, 2, 3, 0)  # [T, H, W, 3]
        video = (video + 1.0) / 2.0
        video = video.clamp(0, 1).detach()

        return video
```

---

### Task 3: Update sds_loss.py — add generate_and_distill mode

**Files:**
- Modify: `train_neoverse_sds/neoverse_sds/sds_loss.py`

**Step 1: Add `compute_distill_loss()` function**

Add after the `compute_tau_and_cfg()` function (after line 251):

```python
def compute_distill_loss(
    rendered_video: torch.Tensor,
    pseudo_gt_video: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    """Compute generate-and-distill loss between rendered video and NeoVerse-generated pseudo-GT.

    The pseudo_gt_video is produced by running full NeoVerse inference on the
    degraded rendering (self-conditioned). This function simply computes the
    pixel-space reconstruction loss — gradients flow through rendered_video
    back to the 4DGS parameters.

    Args:
        rendered_video: Rendered 4DGS frames [T, H, W, 3] in [0, 1], with grad.
        pseudo_gt_video: NeoVerse-generated target [T, H, W, 3] in [0, 1], detached.
        loss_type: "l1" or "lpips".

    Returns:
        Scalar loss tensor.
    """
    # Ensure pseudo_gt is detached (no grad through NeoVerse)
    target = pseudo_gt_video.detach()

    # Handle potential frame count mismatch (VAE temporal asymmetry)
    T_render = rendered_video.shape[0]
    T_target = target.shape[0]
    if T_render != T_target:
        # Subsample rendered to match target (VAE may produce fewer frames)
        indices = torch.linspace(0, T_render - 1, T_target).long()
        rendered_video = rendered_video[indices]

    # Handle potential resolution mismatch
    if rendered_video.shape[1:3] != target.shape[1:3]:
        # Resize rendered to match target
        rendered_for_loss = rendered_video.permute(0, 3, 1, 2)  # [T, 3, H, W]
        rendered_for_loss = F.interpolate(
            rendered_for_loss,
            size=(target.shape[1], target.shape[2]),
            mode="bilinear", align_corners=False,
        )
        rendered_for_loss = rendered_for_loss.permute(0, 2, 3, 1)  # [T, H, W, 3]
    else:
        rendered_for_loss = rendered_video

    if loss_type == "l1":
        return F.l1_loss(rendered_for_loss, target)
    elif loss_type == "l2":
        return F.mse_loss(rendered_for_loss, target)
    else:
        raise ValueError(f"Unknown distill loss_type: {loss_type}")
```

---

### Task 4: Rewrite trainer.py — add experiment A/B/C paths

**Files:**
- Modify: `train_neoverse_sds/trainer.py`

This is the largest task. The changes are in the SDS section of the training loop (lines 268-381).

**Step 1: Add imports**

At the top of trainer.py, add the import for the new loss function:

```python
from train_neoverse_sds.neoverse_sds.sds_loss import (
    compute_neoverse_sds_loss,
    compute_tau_and_cfg,
    compute_distill_loss,
)
```

**Step 2: Add state variables for distill mode**

After `gt_conditioning_cache = {}` (line 215), add:

```python
    # Generate-and-distill state (Experiment B/C)
    distill_pseudo_gts = {}  # azimuth → [T, H, W, 3] tensor (cached generated video)
    distill_last_generated_iter = -999  # Track when pseudo-GTs were last generated
```

**Step 3: Add helper function for determining active experiment mode**

Add before the `training()` function:

```python
def _get_active_mode(config: NeoVerseSDSConfig, new_model_iteration: int) -> str:
    """Determine which experiment mode is active at the current iteration.

    For experiment C (hybrid), returns "A" before curriculum_switch_iter
    and "B" after.
    """
    if config.experiment_mode == "C":
        if new_model_iteration < config.curriculum_switch_iter:
            return "A"
        else:
            return "B"
    return config.experiment_mode
```

**Step 4: Add helper function for generate-and-distill pseudo-GT generation**

Add before `training()`:

```python
def _generate_pseudo_gts(
    neoverse, conditioning_cache, context_pos, context_neg,
    gaussians, deform, pipe, background,
    num_pretrained, config, view0_cameras, is_6dof,
):
    """Generate pseudo-GT videos for all SDS angles using full NeoVerse inference.

    Renders the current 4DGS scene from each SDS angle, uses the rendering as
    self-conditioning for the control branch, and runs full NeoVerse inference
    to produce a clean video target.

    Returns:
        Dict[int, torch.Tensor]: azimuth → [T, H, W, 3] generated video in [0, 1].
    """
    pseudo_gts = {}
    ref_cam = view0_cameras[0]

    for offset in config.neoverse_sds_angles:
        azimuth = int((config.base_azimuth + offset) % 360)

        # Render current 4DGS scene from this angle
        with torch.no_grad():
            rendered = render_video_from_angle(
                gaussians, deform, pipe, background,
                num_pretrained, None, config.radius,
                azimuth, config.elevation,
                config.neoverse_video_num_frames,
                ref_cam, is_6dof,
                only_new_gaussians=False,
                max_deform_magnitude=config.max_deform_magnitude,
                max_gaussian_scale=config.max_gaussian_scale,
            )

        # Use rendered video as self-conditioning for control branch
        with torch.no_grad():
            conditioning_cache.prepare_from_tensors(
                rgb_frames=rendered.detach(),
                depth_frames=None, cameras=None,
                height=config.neoverse_sds_resolution_h,
                width=config.neoverse_sds_resolution_w,
            )
            conditioning = conditioning_cache.get_conditioning(
                device="cuda", dtype=torch.bfloat16,
            )

        # Run full NeoVerse inference
        with torch.no_grad():
            generated = neoverse.full_inference(
                conditioning=conditioning,
                context_pos=context_pos,
                context_neg=context_neg,
                cfg_scale=config.neoverse_cfg_scale,
                control_scale=config.neoverse_control_scale,
                num_inference_steps=config.distill_num_steps,
                seed=config.distill_seed + azimuth,  # Different seed per angle, but stable
                height=config.neoverse_sds_resolution_h,
                width=config.neoverse_sds_resolution_w,
                num_frames=config.neoverse_video_num_frames,
            )

        pseudo_gts[azimuth] = generated.cpu()  # Store on CPU to save VRAM
        print(f"  [Distill] Generated pseudo-GT for view {azimuth}deg: {generated.shape}")

    torch.cuda.empty_cache()
    return pseudo_gts
```

**Step 5: Replace the SDS section of the training loop**

Replace lines 268-381 (the entire "NeoVerse Video SDS" section) with:

```python
        # ==============================================================
        # NeoVerse Video SDS / Generate-and-Distill
        #
        # Experiment A: GT-conditioned score matching
        #   - Control branch: GT video (3-ball composite)
        #   - Gradient: v_rendered - v_gt (vanishes at optimum)
        #
        # Experiment B: Generate-and-distill (self-conditioned)
        #   - Render scene → NeoVerse full inference → pseudo-GT
        #   - L1 loss between rendering and pseudo-GT
        #
        # Experiment C: Hybrid (A then B)
        #   - iter < curriculum_switch_iter → Experiment A
        #   - iter >= curriculum_switch_iter → Experiment B
        # ==============================================================
        sds_loss = None
        sds_iter_active = (
            neoverse_sds_active
            and new_model_iteration >= config.neoverse_sds_start_iter
            and new_model_iteration % config.neoverse_sds_interval == 0
        )

        if sds_iter_active:
            # Lazy init pipeline on first SDS iteration
            if neoverse is None:
                neoverse, conditioning_cache, context_pos, context_neg = (
                    _init_neoverse_pipeline(config)
                )

                # Pre-encode GT videos for Experiment A/C conditioning
                sds_gt_folder = config.sds_gt_video_folder or config.gt_video_folder
                if sds_gt_folder and config.experiment_mode in ("A", "C"):
                    sds_gt_angles = [int((config.base_azimuth + offset) % 360)
                                     for offset in config.neoverse_sds_angles]
                    sds_gt_videos = load_gt_videos(
                        sds_gt_folder, sds_gt_angles,
                        config.neoverse_video_num_frames,
                        target_height=config.neoverse_sds_resolution_h,
                        target_width=config.neoverse_sds_resolution_w,
                    )
                    if sds_gt_videos:
                        print("[NeoVerse SDS] Pre-encoding SDS GT video conditioning...")
                        for angle, gt_vid in sds_gt_videos.items():
                            angle_cache = VideoConditioningCache(neoverse)
                            with torch.no_grad():
                                angle_cache.prepare_from_tensors(
                                    rgb_frames=gt_vid, depth_frames=None, cameras=None,
                                    height=config.neoverse_sds_resolution_h,
                                    width=config.neoverse_sds_resolution_w,
                                )
                            gt_conditioning_cache[angle] = angle_cache
                            print(f"  Cached GT conditioning for view {angle}deg")
                        torch.cuda.empty_cache()

            try:
                active_mode = _get_active_mode(config, new_model_iteration)

                if new_model_iteration % 100 == 0:
                    print(f"[SDS] iter {iteration} (new={new_model_iteration}), mode={active_mode}")

                # ----------------------------------------------------------
                # MODE A: GT-conditioned score matching
                # ----------------------------------------------------------
                if active_mode == "A":
                    tau, cfg_scale = compute_tau_and_cfg(new_model_iteration, {
                        "tau_max": config.neoverse_tau_max,
                        "tau_min": config.neoverse_tau_min,
                        "tau_annealing_iters": config.tau_annealing_iters,
                        "cfg_scale_init": config.neoverse_cfg_scale,
                        "cfg_scale_final": config.neoverse_cfg_scale,
                        "cfg_annealing_iters": 0,
                    })

                    # Pick random SDS angle
                    azimuth = int((config.base_azimuth + np.random.choice(config.neoverse_sds_angles)) % 360)
                    ref_cam = view0_cameras[0]

                    # Render video from 4DGS
                    rendered_video = render_video_from_angle(
                        gaussians, deform, pipe, background,
                        num_pretrained, None, config.radius,
                        azimuth, config.elevation,
                        config.neoverse_video_num_frames,
                        ref_cam, is_6dof,
                        only_new_gaussians=False,
                        max_deform_magnitude=config.max_deform_magnitude,
                        max_gaussian_scale=config.max_gaussian_scale,
                    )

                    latest_sds_video = (rendered_video.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

                    # Get GT conditioning (or self-conditioning fallback)
                    if azimuth in gt_conditioning_cache:
                        conditioning = gt_conditioning_cache[azimuth].get_conditioning(
                            device="cuda", dtype=torch.bfloat16,
                        )
                    else:
                        with torch.no_grad():
                            conditioning_cache.prepare_from_tensors(
                                rgb_frames=rendered_video.detach(),
                                depth_frames=None, cameras=None,
                                height=config.neoverse_sds_resolution_h,
                                width=config.neoverse_sds_resolution_w,
                            )
                        conditioning = conditioning_cache.get_conditioning(
                            device="cuda", dtype=torch.bfloat16,
                        )

                    # For score matching, also need the GT video itself
                    gt_video_for_sm = None
                    if config.neoverse_sds_mode == "neoverse_score_matching":
                        sds_gt_folder = config.sds_gt_video_folder or config.gt_video_folder
                        if sds_gt_folder and azimuth in (sds_gt_videos if 'sds_gt_videos' in dir() else {}):
                            gt_video_for_sm = sds_gt_videos[azimuth].to("cuda")

                    sds_loss = compute_neoverse_sds_loss(
                        neoverse=neoverse,
                        conditioning=conditioning,
                        rendered_video=rendered_video,
                        context_pos=context_pos,
                        context_neg=context_neg,
                        cfg_scale=cfg_scale,
                        control_scale=config.neoverse_control_scale,
                        tau=tau,
                        sds_grad_scale=config.sds_grad_scale,
                        max_sds_pixel_delta=config.max_sds_pixel_delta,
                        sds_loss_type=config.sds_loss_type,
                        mode=config.neoverse_sds_mode,
                        gt_video=gt_video_for_sm,
                    )

                    loss = loss + config.neoverse_sds_weight * sds_loss

                    if new_model_iteration % 100 == 0:
                        gt_or_self = "GT" if azimuth in gt_conditioning_cache else "self"
                        print(f"  [A] SDS loss={sds_loss.item():.6f}, "
                              f"tau={tau:.3f}, cfg={cfg_scale:.1f}, az={azimuth} ({gt_or_self}-cond)")

                # ----------------------------------------------------------
                # MODE B: Generate-and-distill
                # ----------------------------------------------------------
                elif active_mode == "B":
                    # Regenerate pseudo-GTs if stale
                    iters_since_gen = new_model_iteration - distill_last_generated_iter
                    if not distill_pseudo_gts or iters_since_gen >= config.distill_interval:
                        print(f"[Distill] Regenerating pseudo-GTs at iter {iteration}...")
                        distill_pseudo_gts = _generate_pseudo_gts(
                            neoverse, conditioning_cache, context_pos, context_neg,
                            gaussians, deform, pipe, background,
                            num_pretrained, config, view0_cameras, is_6dof,
                        )
                        distill_last_generated_iter = new_model_iteration

                    # Pick random SDS angle
                    azimuth = int((config.base_azimuth + np.random.choice(config.neoverse_sds_angles)) % 360)
                    ref_cam = view0_cameras[0]

                    # Render current 4DGS scene (WITH gradients)
                    rendered_video = render_video_from_angle(
                        gaussians, deform, pipe, background,
                        num_pretrained, None, config.radius,
                        azimuth, config.elevation,
                        config.neoverse_video_num_frames,
                        ref_cam, is_6dof,
                        only_new_gaussians=False,
                        max_deform_magnitude=config.max_deform_magnitude,
                        max_gaussian_scale=config.max_gaussian_scale,
                    )

                    latest_sds_video = (rendered_video.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

                    # Get cached pseudo-GT for this angle
                    if azimuth in distill_pseudo_gts:
                        pseudo_gt = distill_pseudo_gts[azimuth].to(device=rendered_video.device)
                    else:
                        print(f"[WARN] No pseudo-GT for azimuth {azimuth}, skipping distill")
                        continue

                    sds_loss = compute_distill_loss(
                        rendered_video=rendered_video,
                        pseudo_gt_video=pseudo_gt,
                        loss_type=config.distill_loss_type,
                    )

                    loss = loss + config.neoverse_sds_weight * sds_loss

                    if new_model_iteration % 100 == 0:
                        print(f"  [B] Distill loss={sds_loss.item():.6f}, az={azimuth}")

            except Exception as e:
                print(f"[SDS ERROR] iter {iteration}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                sds_loss = None
```

**Step 6: Add GT video storage for score matching**

In the lazy init section (Step 5), after pre-encoding GT videos, store the raw GT videos too:

The `sds_gt_videos` variable needs to be accessible in the score matching section. Move its declaration to the state variables section (alongside `gt_conditioning_cache`):

```python
    sds_gt_videos = {}  # Store raw GT video tensors for score matching
```

And in the lazy init block where GT videos are loaded, add:
```python
                    if sds_gt_videos:
                        # Store raw tensors for score matching mode
                        sds_gt_videos_store = sds_gt_videos
```

Then reference `sds_gt_videos_store` in the score matching section.

Actually, simpler: just keep `sds_gt_videos` as a variable that persists in the training loop scope. It's set during lazy init and accessed during mode A.

---

### Task 5: Update __main__.py — print new config fields

**Files:**
- Modify: `train_neoverse_sds/__main__.py`

**Step 1: Add experiment mode and distill params to config printout**

After the existing config print section (around line 55), add:

```python
    print(f"  Experiment Mode: {config.experiment_mode}")
    if config.experiment_mode in ("B", "C"):
        print(f"  Distill Interval: {config.distill_interval}")
        print(f"  Distill Steps: {config.distill_num_steps}")
        print(f"  Distill Seed: {config.distill_seed}")
        print(f"  Distill Loss: {config.distill_loss_type}")
    if config.experiment_mode == "C":
        print(f"  Curriculum Switch: iter {config.curriculum_switch_iter}")
    if config.sds_gt_video_folder:
        print(f"  SDS GT Video Folder: {config.sds_gt_video_folder}")
```

---

### Task 6: Create three SLURM script variants

**Files:**
- Modify: `scripts/run_neoverse_sds.sh` (base, becomes Experiment A)
- Create: `scripts/run_neoverse_sds_B.sh` (Experiment B)
- Create: `scripts/run_neoverse_sds_C.sh` (Experiment C)

**Step 1: Update `run_neoverse_sds.sh` for Experiment A**

Key changes to the existing script:
- Add `SDS_GT_VIDEO_FOLDER` pointing to 3-ball composite GT
- Add `--experiment_mode A`
- Add `--sds_gt_video_folder`
- Add `--neoverse_sds_mode neoverse_score_matching`
- Fix defaults: tau_max=0.98, sds_weight=1.0, max_gaussian_scale=10.0

```bash
# === Configuration ===
CHECKPOINT="./output/2BouncingBalls_baseline"
GT_VIDEO_FOLDER="./data/baseline_4view_videos"
SDS_GT_VIDEO_FOLDER="./data/3ball_original_4views"
INSERT_PLY="./output/BlueBallMasked_debug/point_cloud/iteration_40000/point_cloud.ply"
NEOVERSE_MODELS="${NEOVERSE_ROOT}/models"
LORA_PATH="${NEOVERSE_MODELS}/NeoVerse/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"
RUN_NAME="exp_A_gt_score_matching"

python -m train_neoverse_sds \
    --load_checkpoint "${CHECKPOINT}" \
    --insert_ply "${INSERT_PLY}" \
    --gt_video_folder "${GT_VIDEO_FOLDER}" \
    --sds_gt_video_folder "${SDS_GT_VIDEO_FOLDER}" \
    --model_path "./output/${RUN_NAME}" \
    --run_name "${RUN_NAME}" \
    --neoverse_model_path "${NEOVERSE_MODELS}/NeoVerse" \
    --neoverse_lora_path "${LORA_PATH}" \
    --experiment_mode A \
    --neoverse_sds_mode neoverse_score_matching \
    --neoverse_sds_weight 1.0 \
    --neoverse_control_scale 1.0 \
    --neoverse_video_num_frames 17 \
    --neoverse_sds_interval 10 \
    --neoverse_sds_start_iter 0 \
    --neoverse_tau_min 0.02 \
    --neoverse_tau_max 0.98 \
    --sds_grad_scale 0.1 \
    --max_sds_pixel_delta 0.1 \
    --max_gaussian_scale 10.0 \
    --base_azimuth 90 --elevation -10 --radius 4.0 --fov 39.6 \
    --resolution_h 800 --resolution_w 800 \
    --is_blender \
    --white_background \
    --iterations 5000 \
    --new_iterations 5000 \
    --test_iterations 40001 40500 41000 42000 43000 44000 45000 \
    --save_iterations 42000 45000
```

**Step 2: Create `run_neoverse_sds_B.sh` for Experiment B**

Same SBATCH header. Key differences:
- `--experiment_mode B`
- `--neoverse_sds_mode generate_and_distill` (though experiment_mode B overrides)
- No `--sds_gt_video_folder` (not needed)
- `--distill_interval 200`
- `--distill_num_steps 4`
- `--distill_seed 42`

```bash
RUN_NAME="exp_B_generate_and_distill"

python -m train_neoverse_sds \
    --load_checkpoint "${CHECKPOINT}" \
    --insert_ply "${INSERT_PLY}" \
    --gt_video_folder "${GT_VIDEO_FOLDER}" \
    --model_path "./output/${RUN_NAME}" \
    --run_name "${RUN_NAME}" \
    --neoverse_model_path "${NEOVERSE_MODELS}/NeoVerse" \
    --neoverse_lora_path "${LORA_PATH}" \
    --experiment_mode B \
    --distill_interval 200 \
    --distill_num_steps 4 \
    --distill_seed 42 \
    --neoverse_sds_weight 1.0 \
    --neoverse_control_scale 1.0 \
    --neoverse_video_num_frames 17 \
    --neoverse_sds_interval 10 \
    --neoverse_sds_start_iter 0 \
    --neoverse_tau_min 0.02 \
    --neoverse_tau_max 0.98 \
    --sds_grad_scale 0.1 \
    --max_sds_pixel_delta 0.1 \
    --max_gaussian_scale 10.0 \
    --base_azimuth 90 --elevation -10 --radius 4.0 --fov 39.6 \
    --resolution_h 800 --resolution_w 800 \
    --is_blender \
    --white_background \
    --iterations 5000 \
    --new_iterations 5000 \
    --test_iterations 40001 40500 41000 42000 43000 44000 45000 \
    --save_iterations 42000 45000
```

**Step 3: Create `run_neoverse_sds_C.sh` for Experiment C**

```bash
RUN_NAME="exp_C_hybrid_curriculum"

python -m train_neoverse_sds \
    --load_checkpoint "${CHECKPOINT}" \
    --insert_ply "${INSERT_PLY}" \
    --gt_video_folder "${GT_VIDEO_FOLDER}" \
    --sds_gt_video_folder "${SDS_GT_VIDEO_FOLDER}" \
    --model_path "./output/${RUN_NAME}" \
    --run_name "${RUN_NAME}" \
    --neoverse_model_path "${NEOVERSE_MODELS}/NeoVerse" \
    --neoverse_lora_path "${LORA_PATH}" \
    --experiment_mode C \
    --curriculum_switch_iter 2000 \
    --neoverse_sds_mode neoverse_score_matching \
    --distill_interval 200 \
    --distill_num_steps 4 \
    --distill_seed 42 \
    --neoverse_sds_weight 1.0 \
    --neoverse_control_scale 1.0 \
    --neoverse_video_num_frames 17 \
    --neoverse_sds_interval 10 \
    --neoverse_sds_start_iter 0 \
    --neoverse_tau_min 0.02 \
    --neoverse_tau_max 0.98 \
    --sds_grad_scale 0.1 \
    --max_sds_pixel_delta 0.1 \
    --max_gaussian_scale 10.0 \
    --base_azimuth 90 --elevation -10 --radius 4.0 --fov 39.6 \
    --resolution_h 800 --resolution_w 800 \
    --is_blender \
    --white_background \
    --iterations 5000 \
    --new_iterations 5000 \
    --test_iterations 40001 40500 41000 42000 43000 44000 45000 \
    --save_iterations 42000 45000
```

---

### Task 7: Update documentation

**Files:**
- Replace: `train_neoverse_sds/PLAN.md`
- Update: `train_neoverse_sds/lessons.md`
- Update: `NeoVerse/CLAUDE.md`

**Step 1: Replace `train_neoverse_sds/PLAN.md`**

Replace entire contents with a concise version that reflects the new architecture:

```markdown
# NeoVerse SDS for 4D Object Insertion

## Goal

Optimize a 4D Gaussian Splatting scene (with inserted object) using NeoVerse's
video diffusion prior. Three experimental approaches:

- **Experiment A**: GT-conditioned score matching
- **Experiment B**: Generate-and-distill (self-conditioned)
- **Experiment C**: Hybrid curriculum (A then B)

See `NeoVerse/docs/plans/2026-03-09-neoverse-sds-experiments-design.md` for full design.

## Architecture

```
Training Loop (per iteration):
  1. Photometric loss: Render view0 → L1 against GT video
  2. Regularization: deform_reg, position_anchor, temporal, ARAP
  3. NeoVerse SDS (every 10 iters):
     Experiment A: GT score matching
       - Control branch: GT video (3-ball composite)
       - Noise both rendered + GT with same ε at same τ
       - Gradient: v_rendered - v_gt
     Experiment B: Generate-and-distill
       - Render scene → self-condition → full NeoVerse inference → pseudo-GT
       - L1 loss: ||rendered - pseudo_gt||
       - Pseudo-GTs regenerated every distill_interval iters
     Experiment C: A for iter < 2000, then B
```

## File Structure

```
train_neoverse_sds/
├── PLAN.md              # This file
├── lessons.md           # Debugging history and lessons
├── __main__.py          # Entry point
├── config.py            # NeoVerseSDSConfig + argparser
├── trainer.py           # Main training loop (experiments A/B/C)
└── neoverse_sds/
    ├── __init__.py
    ├── pipeline.py      # NeoVerseSDS wrapper (encode, decode, predict, full_inference)
    ├── conditioning.py  # VideoConditioningCache (control branch inputs)
    └── sds_loss.py      # SDS loss + distill loss + tau/cfg annealing
```

## Key Hyperparameters (Corrected)

| Parameter | Default | Notes |
|-----------|---------|-------|
| neoverse_tau_max | 0.98 | Was 0.80, working run used 0.98 |
| neoverse_sds_weight | 1.0 | Was 0.1, working run used 5.0 |
| max_gaussian_scale | 10.0 | Was 0.1, prevented scale deformations |
| lambda_deform_reg | 0.01 | Was 1.0, root cause of frozen ball |
| experiment_mode | A | A/B/C |
| distill_interval | 200 | Regenerate pseudo-GTs every N iters |
| curriculum_switch_iter | 2000 | Switch A→B at this iter (experiment C) |
```

**Step 2: Update `train_neoverse_sds/lessons.md`**

Add a new section at the top of the "What We Need To Do Next" section:

```markdown
## Current Approach (2026-03-09)

### Three Experiments

The GT conditioning problem (control branch fighting insertion) is resolved by
providing GT videos that INCLUDE the inserted ball (3-ball composite from
`data/3ball_original_4views/`).

**Experiment A: GT-Conditioned Score Matching**
- Control branch sees 3-ball GT → works FOR insertion
- Score matching: gradient vanishes when rendered ≈ GT
- Corrected params: tau_max=0.98, sds_weight=1.0, max_gaussian_scale=10.0

**Experiment B: Generate-and-Distill**
- No GT needed for SDS (only for photometric view0)
- Full NeoVerse inference produces pseudo-GT targets
- Self-conditioning: rendered scene → control branch → generate clean video

**Experiment C: Hybrid**
- Phase 1 (iter 0-2000): Experiment A establishes ball position
- Phase 2 (iter 2000+): Experiment B refines quality

### Previous Root Causes (Resolved)
- GT conditioning WITHOUT ball → now using 3-ball GT
- tau_max=0.80 → now 0.98
- sds_weight=0.1 → now 1.0
- max_gaussian_scale=0.1 → now 10.0
```

**Step 3: Update `NeoVerse/CLAUDE.md`**

Add a section after "### Model Checkpoints" (before the closing of the file):

```markdown
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
```
