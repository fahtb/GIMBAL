# SAM2.1 hybrid tracking — continuation checkpoint, 2026-11-13

## Latest speed/recovery correction

Local search previously computed correlation over the whole image before excluding far locations. It now matches an actual crop, reuses resized templates and removes duplicate scales. Same-image30-repeat CPU benchmark:mean45.66→6.57ms,p9557.63→8.66ms (`test_results/hybrid/search_latency_ab.json`). This measures the proposal stage, not total tracking FPS.

Search now cycles0/90/180/270-degree views; confirmation still needs the same SAM/appearance thresholds, margin and three observations. Before acceptance, competing locations are checked across orientations. `search_burst_interval=0.05` applies during the first `search_burst_seconds=2`; afterwards each orientation has `search_interval/4` scheduling budget. Actual work can take longer. Bounded latest-frame capture is unchanged.

The Kalman outlier exception uses recent trusted structure plus LK/spatial agreement, avoiding unnecessary dependence on the initial view after an appearance change. No identity threshold was lowered. GPU synthetic relocated return improved0.4→0.2source seconds; real can/hand recovery has not been reproduced without a raw recording. Neither quarter-turn matching nor grayscale templates solve arbitrary3D rotation or severe blur.

## Run

No dependency upgrade or checkpoint download is required in the existing `D:\Sam\.venv`.

```powershell
cd D:\Sam
& .\.venv\Scripts\python.exe .\scripts\camera.py --source 0 --engine enhanced --sam-only --image-size 512 --cam-size 640x480 --debug-tracking --csv .\runtime\hybrid_live.csv --profile .\runtime\hybrid_live.profile.json --telemetry .\runtime\hybrid_target.json
```

Click a point or drag a box on the left preview. S/R selects again; H toggles debug; W/C records wrong/correct feedback; Q/Esc exits. K toggles Kalman, I toggles the new ranked recovery, O toggles optical flow. K/I/O are ablations, not recommended ways to improve accuracy. Feature changes are visible in debug/CSV. In offline playback, O controls foreground LK; there is no intermediate display-flow thread to toggle.

Default `camera.py` still uses `--engine baseline` to preserve existing behavior. Explicitly select enhanced for the new policy. `--engine baseline --small-target off` selects the original full-frame source predictor. Enhanced includes coordinate-safe small-target ROI epochs. `--small-target off` is a baseline comparison option, not an enhanced-engine ablation.

Use `--no-kalman`, `--no-reid`, `--no-flow` for startup ablations. `--no-reid` disables enhanced ranked global recovery; the baseline structural/geometry checks remain. Optional settings are in `configs/hybrid.json`; inherited ROI settings are in `configs/small_target.json`.

Record a new camera session with `--record-raw .\runtime\raw_session_01.mp4`. The adjacent `.timestamps.csv` preserves capture times. Existing recordings are refused before workers start; use a fresh filename. `--save-video` records annotated output separately. Logs and annotated output may replace the same output filename on subsequent runs, so use session-specific paths when preserving experiments.

## What actually runs

AI model: **SAM 2.1 Hiera Tiny**, checkpoint `sam2/checkpoints/sam2.1_hiera_tiny.pt`. The adapter imports `vendor/source_camera/sam2` and rejects conflicting already-imported `sam2` packages. It uses the source `SAM2CameraPredictor`, not the old SAMURAI/YOLO routing.

Source API remains `load_first_frame` → `add_new_prompt`/`add_new_mask` → `track`. New selection/reset clears both condition state and frame index. Unsupported `add_new_prompt_during_track(if_new_target=True)` is not called. Weights load once; proposal verification temporarily creates a fresh predictor state, restores the previous state on rejection, and preserves the external target ID on recovery.

Capture and inference use a bounded latest-frame slot. Display flow remains independent. BGR→RGB occurs in the source adapter; prompts/masks use the current source/ROI coordinate system. A changed ROI starts a fresh coordinate epoch from a previously verified image/mask. A changed input resolution requires selection again. Model pixels are distinct from display pixels.

The enhanced layer combines:

- Timestamp-based Kalman `[cx,cy,w,h,vx,vy,vw,vh]`; Joseph covariance update and Mahalanobis gating. Relative edge measurement noise handles size-dependent box deformation.
- Foreground Lucas–Kanade with forward/backward filtering and RANSAC. Existing dense display flow remains separate from SAM memory.
- Immutable original crop/mask/ID and up to five verified recent structural views. Features are normalized grayscale structure and multiscale template correlation, **not a learned ReID embedding or calibrated identity probability**. Color does not decide identity.
- Local search around predicted position with uncertainty-dependent radius, expansion, then global multiscale proposals. Top-1 must beat top-2 by a configured margin; SAM mask stability, structure and proposal agreement must pass three consecutive checks.
- Rejected SAM observations are removed from the recent bank. Verified recent frames can seed ROI epochs. The source conditioning state is not claimed to implement the full SAMURAI or DAM4SAM memory algorithm.
- Hardware-independent PID simulation with dead zone, smoothing, anti-windup and speed/acceleration limits. Only confirmed observations produce TRACK commands; missing/stale evidence produces HOLD. A writer watchdog also invalidates a command after 250 ms without fresh evidence. No motor commands are sent.

SAM quality is the measured ratio of pixels above logit +0.5 to pixels above -0.5. It is mask stability, not predicted IoU, object probability or identity confidence. The source API does not return those confidence values here.

## Faults found and corrected

1. Fixed 5-pixel Kalman measurement noise rejected natural deformation of a roughly 470-pixel-tall person. Logged frame70 had innovation25.06 despite a modest box change. Relative edge noise was added; identity thresholds were not lowered. The final natural person test retained all100 TRACKING states.
2. Partial occlusion shrank the measured box, triggering `kalman_gate`, followed by `appearance_gate` against a full-object template. This caused16 missing-mask frames in a47-frame synthetic fixture.
3. A partial mask is now checked against the **visible texture in aligned reference coordinates**. It needs sufficient texture, >=90% containment, >=0.90 grayscale correlation and at least32 pixels. It can display a yellow UNCERTAIN mask for at most1.2 seconds. It does not update SAM memory, gallery or Kalman measurement and does not supply gimbal control coordinates. Long partial visibility requires recovery; it cannot bypass the timeout through the ordinary update branch.
4. Display flow could restore dx/dy from an unconfirmed mask. Both display paths now keep those coordinates unavailable when `is_confirmed=False`.
5. Rejected/probation SAM proposal calls could be logged as idle. Their inference time/call flag and `candidate_probation` reason are now recorded.
6. Runtime ablation hotkeys were unfinished at the previous interruption. K/I/O now dispatch through the inference worker rather than mutating predictor state from the UI thread.

## Verified results and their scope

All GPU runs used the actual RTX3050 Laptop6GB and torch2.5.1+cu124. No accuracy number below is inferred from TRACKING status alone.

| Real video, same100 person frames,512 BF16 | Baseline | Enhanced after continuation |
|---|---:|---:|
| Processing Hz, excluding initial prompt |12.71|11.33|
| Whole pipeline mean / p95 ms |78.65 /87.09|88.23 /97.53|
| SAM adapter mean ms |66.36|64.65|
| TRACKING states |100/100|100/100|
| Peak torch allocated MiB |353.87|353.87|

Artifact: `test_results/hybrid/person_continuation.json`. This tests regression and speed, not identity accuracy; GT is absent. Hardware load affects timings; do not treat these as guaranteed rates.

Earlier asynchronous A/B on the same person clip: Live23.67→23.72FPS; SAM updates11.24→10.30Hz; mean SAM age151.67→163.33ms, age p95208.33ms both. Artifact: `test_results/hybrid/realtime/report.json`. Display-loop latency is not inference latency. New worker logs separately expose queue, compute and capture-to-worker completion timing.

Earlier1024 BF16 run: enhanced4.24Hz, mean236.09/p95249.30ms, peak674.94MiB allocated.512 stays default. FP16 did not show a clear speed advantage, and its early run exposed the pre-fix covariance rejection; it is not an accuracy comparison against the final BF16 engine. Triton is absent, so CUDA Inductor compilation was not benchmarked or enabled. Missing `_C` means source hole-filling postprocessing is skipped; inference itself ran.

Camera-only DSHOW640×480 MJPG:100 frames in5.036s =19.86FPS. Driver advertised30FPS but rejected buffer=1. No exposure setting was changed. Actual camera/CUDA lifecycle was also exercised with simulated UI callbacks: box, point, S/R/H/W/C/K/I/O/Q, rendering, CSV and final HOLD. This is an integration test on an uncontrolled scene, not proof of correct identity tracking. Files: `test_results/hybrid/live_continuation/`.

Synthetic masks below test logic only; they are not natural-video accuracy:

| Fixture | Baseline correct visible masks | Previous enhanced | Current enhanced | Current false-mask frames |
|---|---:|---:|---:|---:|
| Partial occlusion |47/47|31/47|47/47|0|
| Full occlusion |32/33|31/33|31/33|0|
| Same-color distractor |32/33|31/33|31/33|0|
| Exit and return at new position |15/33|31/33|31/33|0|

Partial visible masks remain UNCERTAIN/HOLD, not falsely reported as fully confirmed TRACKING. The long-partial test intentionally stops that grace interval:31/47 masks, no false locks; it verifies that the trusted seed remains unchanged through occlusion. Artifacts: `continuation_scenarios.json`, `partial_timeout_final.json` in `test_results/hybrid`.

Prior synthetic global relocation test recovered atframe28 after return22 at15Hz source timestamps (0.4 source seconds); five repeated resets passed. This is not a measured natural camera reacquisition time. A300-frame static GPU soak retained7 recent memories and229.16MiB allocated fromframe50 through299. Hours-long leak testing remains pending.

CPU checks:12 hybrid tests,5 session-audit tests,7 smooth-video tests,6 playback tests pass in this continuation. They cover variable timestamps, outliers, PID limits, stale HOLD, duplicate-candidate margin, LK translation, no-CUDA error, camera-release-on-read-failure, sparse-GT metrics, partial texture rejection and prevention of display control revival. The audit now counts one control-lock loss through UNCERTAIN/REDETECTING and one confirmed resumption, rather than missing the loss because no direct TRACKING→LOST transition occurred.224 Python files under scripts/lib/vendor/source_camera/training_bundle parsed without syntax errors; this is not exhaustive behavioral validation of every legacy module.

## Reproduce and evaluate

```powershell
cd D:\Sam
& .\.venv\Scripts\python.exe .\scripts\check_hybrid.py
& .\.venv\Scripts\python.exe .\scripts\check_hybrid_scenarios.py --cases partial,occlusion,same_color,exit --skip-soak --output test_results/hybrid/my_scenarios.json
& .\.venv\Scripts\python.exe .\scripts\eval_video.py --case person --engine both --mode realtime --max-frames 100 --output .\runtime\hybrid_ab
& .\.venv\Scripts\python.exe .\scripts\benchmark_hybrid.py --case person --frames 100 --output .\runtime\hybrid_person.json
```

For annotated natural videos, copy `scripts/eval_cases.json` to a new cases file and set its video, initial frame, initial `x,y,w,h` box and resize width. Add `gt` pointing to JSON records such as `{"frame":0,"visible":true,"box":[10,20,50,70],"distractors":[]}`. GT boxes use xyxy at processing resolution. Explicitly annotate both sides of visibility changes; interpolation is only within equal-visibility intervals. Run `eval_video.py --cases <file> --engine both`. Reports include IoU success, center error, identity switches, false locks when absent and verified recovery timing only where GT exists. W/C feedback remains human event labels; it is not dense frame GT.

## File map and continuation priorities

- `hybrid_tracker.py`: identity, policy, states, ranked search and verification, partial observations.
- `tracking_control.py`: Kalman and simulated gimbal output safety.
- `motion_flow.py`: LK evidence and aligned visible-region verification.
- `source_camera_tracker.py`: genuine source model adapter and observed mask stability.
- `camera.py`, `realtime_video.py`, `video_playback.py`, `mask_display.py`: integration, controls, telemetry and rendering.
- `small_target.py`: existing ROI/memory logic retained; enhanced debug HUD.
- `video_profile.py`: bounded timing sample buffers, exact accumulated counts/means.
- `eval_video.py`, `recovery/metrics.py`, `benchmark_hybrid.py`: A/B and labelled evaluation.
- `check_hybrid*.py`: CPU, GPU, scenarios, lifecycle/soak tests.
- `configs/hybrid.json`, `PROGRESS.md`, this document: configuration and resume record.

Remaining priorities: labelled real same-color/occlusion/rotation clips; threshold calibration on separate validation sessions; optional camera-motion compensation under genuine pan/rotation; stronger appearance features only after profiling and accuracy comparison; sustained webcam/VRAM tests. Two visually identical objects after complete occlusion may be impossible to disambiguate; manual selection remains necessary.

No fine-tuning was run. First collect reviewed masks and hard-negative identities, split by recording session/object into train/validation/test, then separate segmentation failures from identity-search failures. Never train on unreviewed tracker outputs. A learned encoder or SAM fine-tune needs measured improvement on the held-out sessions before integration.20–30 inferenceHz and p95<150ms end-to-end are not achieved consistently.

References consulted: [source online predictor](https://github.com/Gy920/segment-anything-2-real-time), [SAM2 installation](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md), [SAMURAI](https://github.com/yangchris11/samurai), [DAM4SAM](https://github.com/jovanavidenovic/DAM4SAM). The latter two inform external motion/quality/distractor checks; their full internal algorithms were not merged or relabelled as this implementation.
