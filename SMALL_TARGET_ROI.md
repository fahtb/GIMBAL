# Small-target ROI — 2026-10-17

Branch `feature/small-target-roi`; pre-edit backup `e9a8bc8` includes compact HUD changes.
No source directory, checkpoint or vendored SAM2 implementation was modified. One CUDA model, no new dependency/model.

## Run

```powershell
cd D:\Sam
& D:\Sam\.venv\Scripts\python.exe D:\Sam\scripts\camera.py --source 0 --sam-only --small-target auto --camera-config D:\Sam\configs\camera.json --cam-size 640x480 --debug-tracking --csv D:\Sam\runtime\small_live.csv --profile D:\Sam\runtime\small_live.profile.json
```

`camera.json` tries DirectShow first. Use the previous `configs/camera_msmf_measured.json` only if MSMF works on your current driver. During this task MSMF read failed with an OpenCV Mat/stride assertion; DirectShow delivered 640x480 at 2.88 FPS over a short six-frame probe. This does NOT verify 30 FPS webcam operation or 1280x720 capture. Camera is released after probes. No exposure changes were made.

- Default `--small-target auto`: guarded source predictor with adaptive ROI.
- `--small-target off`: original full-frame source predictor, unchanged decision policy, for A/B/fallback.
- `--small-target-config D:\Sam\configs\small_target.json`: ROI/gates/search configuration.
- `--debug-tracking`: show status/reason, native target pixel dimensions, equivalent model dimensions, ROI outline, structural similarity. H toggles debug while playing. Normal HUD stays model + Live FPS.
- Existing S/R reselect, W/C feedback, Q/Esc exit, camera/video, raw recording, CSV/JSON, save-video remain.
- Source 512 is default. `--image-size 1024` chooses the actual compatible 1024 config, not display upscaling, but was not performance-tested here and is not enabled by default.
- `--cam-size 1280x720` requests more native detail; check the new actual-frame-size message. Driver may refuse; more pixels only help if optical detail exists. Increasing camera resolution alone with full-frame SAM512 can shrink model-space target pixels.

Example deterministic A/B on a source video (same ROI selection):
```powershell
& D:\Sam\.venv\Scripts\python.exe D:\Sam\scripts\camera.py --source D:\Sam\videos\test1.mp4 --resize-width 960 --mode offline --init-frame 0 --init-bbox 288.5185185,293.1111111,7.3744856,12.6419753 --max-frames 40 --no-display --small-target auto --csv D:\Sam\runtime\beach_auto.csv
```
Repeat with `--small-target off --csv D:\Sam\runtime\beach_off.csv`.
`eval_video.py` also accepts `--small-target auto/off` and separates output names.

## Code findings, not a confidence guess

1. Source `perpare_data` resizes each RGB image directly to a square512, no padding. BGR->RGB and x/y normalization were correct. A 10px-wide object in a1280px frame becomes4px wide at the model input. Source Tiny512 has a32x32 top feature grid, further reducing small-target evidence. Increasing only display size cannot help.
2. Existing source mask threshold is logits>0 and accepts any nonempty mask. No confidence/min-area rejection in that adapter. Source drops its internal object/IoU scores from returned track output; debug therefore says identity confidence N/A rather than inventing one.
3. Dense display transport had bbox-area>=16 unless `small=True`. New path marks small-capable results; tested that a1pixel accepted mask reaches display and is removed immediately on a LOST result.
4. Camera selection required >3px width/height. Changed to >=1 native pixel while preserving display-to-camera mapping. Video selector already supports zoom; source point prompt still available through `--init-point`.
5. Full source track immediately writes every predicted frame to its memory, including a bad mask. New path restores the previous recent-memory dictionary on rejected/probation observations. Frame counter still advances, preserving online chronology.
6. In actual beach clip, initial accepted mask is4x8px at960x540. Full512 representation is~2.13x7.59px. A128x128 native crop represents it at16x32px. Pixel expansion preserves available detail; it cannot restore missing optical information.

## ROI/state design

`SourceCameraTracker` remains the only SAM model. `SmallTargetTracker` adapts its image/state inputs, using actual source `load_first_frame`, `add_new_mask`, and `track` APIs.

- Prefer original full-frame manual selection. If it returns an empty mask for an explicit small box, try that user-selected box on a context crop. This fallback is NOT automatic arbitrary-object discovery.
- Keep crop origin and size fixed through each epoch. Never shift a crop and continue its old positional memory.
- When target approaches an edge or its scale changes enough, form a context ROI around bounded velocity prediction, covering the previous accepted target. Seed a NEW coordinate state on the last accepted image/mask in the new ROI, then track the current frame. Same model weights; only state resets. This normally adds one extra inference on rebase frames.
- Map crop masks back by exact integer offset into an original-size mask. Bbox/center/dx/dy are recomputed in original coordinates.
- The original appearance anchor is immutable; up to five recent structural templates are added only on verified observations with similarity >=0.8. Structural grayscale template matching searches small translation/scale alignments to avoid falsely rejecting a tiny target after a one-pixel boundary change. It does not use HSV/color histograms. Similarity is a classical matching score, not a probability or SAM confidence.
- Accept with structural agreement + bounded position/motion + area change checks; tiny masks get explicit boundary-pixel tolerance. Only accepted observations replace the bounded last-trusted frame/mask and recent SAM memory.
- TRACKING -> UNCERTAIN -> LOST after configured failures. UNCERTAIN/LOST produce no mask, box, dx/dy or control output, rather than displaying the old target as a valid detection. Short display-flow interpolation between completed SAM observations remains age-limited.
- During loss, periodically expand crop up to full frame, re-seeded from trusted evidence. Candidate must pass stronger structure threshold and spatial consistency for3 observations. Low texture cannot confirm recovery. Default automatic recovery continues after timeout at a bounded search frequency; `auto_reacquire:false` restores manual-only timeout. This is conservative local recovery, not a global detector or guaranteed long-distance re-identification.
- Memory bounded: source recent bank7 plus conditioning, last trusted image/mask and at most six small appearance vectors. No growing video buffer or second model.

Default config highlights: trigger48 model pixels on short side; context5x max target dimension; minimum ROI128px; edge fraction0.18; minimum epoch8 inference frames; search every8 observations; expansion1.7x; lost after3 failures; recovery3 consistent observations; timeout8s. All values are in `configs/small_target.json`. No acceptance threshold was lowered to obtain the reported improvement; alignment and area-boundary handling were corrected based on observed false rejections.

## Measured results

Ground truth tests are procedural scenes,47 frames per case, exact same frames/box for both paths. Correct=mask IoU>=0.5 on visible target frames. Wrong frame=mask present while target absent or more overlap with distractor than target. These are NOT natural-video accuracy claims.

1280x720 tests (`test_results/small_target/ab_wide.json`):

| Case | Correct visible % full / ROI | Wrong frames full / ROI |
|---|---:|---:|
| Near |100 /100|0 /0|
| Near->far->near |93.6 /100|0 /0|
| Small lateral movement |100 /100|0 /0|
| Short occlusion |100 /93.9|0 /0|
| Disappearance then distant return |100 /45.5|0 /0|
| Same-color distractor |30.3 /93.9|37 /0|

The short-occlusion reduction is the2-frame recovery probation. The distant-return regression is real: the candidate is beyond the permitted motion evidence and remains LOST; manual reselection is needed. Do not call this a successful automatic recovery.
640x480 tests: near/scale/lateral100% both; occlusion/same-color100% full versus93.9% ROI; distant return100% full versus45.5% ROI. No wrong frames in either path at640x480. ROI is not uniformly better in all cases.

GPU: actual RTX3050 Laptop6GB, torch2.5.1+cu124, BF16. Sequential synthetic ROI mean48–54ms/frame, p95 up to83.3ms on rebase-heavy lateral movement. Peak allocated ~354.87MiB for these tests; this is allocated tensor memory, not total system GPU use.

Async1280x720 generated video,141 frames30FPS:

| Metric | Full | ROI |
|---|---:|---:|
| Display FPS |29.80|29.53|
| Completed tracking results/sec |20.01|17.50|
| Source SAM invocation p95 ms |53.90|52.97|
| Mean age ms |89.83|100.95|

SAM profile calls include re-seeding, so ROI SAM calls18.98Hz is NOT the17.50Hz rate of completed tracking results. Rebase overhead is captured by pipeline_ms/worker timings. These headless results include display compositing; no claim of guaranteed camera30FPS.

Natural clips: beach and person,40 frames each per mode, same existing eval box; both100% TRACKING and exit0. No GT, so no identity-accuracy conclusion. Source aquarium40 frames ROI also100% TRACKING/exit0. Mask rendering/save-video, CSV/telemetry and no-display transport exercised.

4 geometry/appearance/tiny-mask-loss unit tests;7 smooth-video and6 playback regression tests pass. Syntax checks pass. Generated-vs-natural reports/logs are in `test_results/small_target/` (not committed). No new packages installed.

## Limitations / files

No reliable recovery if both targets have indistinguishable structure, low texture, complete occlusion with large relocation, or only a few informative camera pixels remain. Real near/far/occlusion camera behavior and long soak are not verified in this task because camera capture was unstable. `_C` hole-filling extension remains unavailable as in baseline; no new compile/graph change.

Added `scripts/small_target.py`, `configs/small_target.json`, `scripts/check_small_target.py`, `scripts/check_small_geometry.py`, this report.
Updated `camera.py` factory/CLI/capture dimensions/selection/debug; `source_camera_tracker.py` trusted mask prompt + pixel telemetry; `realtime_video.py` and `video_playback.py` debug/CSV; `eval_video.py` A/B switch; `check_source_camera.py` explicit baseline; `PROGRESS.md` and baseline source-port documentation.


## Continuous recovery follow-up

Latest live log before this fix had168 timeout_reselect rows and205 appearance_gate rows. The old8-second terminal timeout prevented any subsequent recovery, and permanently expanding to a full image reduced small-object detail.

- `auto_reacquire:true` (default): timeout8s now starts low-duty search, not permanent stop. `slow_search_interval:0.35` bounds sampling after long loss. During candidate confirmation the normal observation rate resumes.
- Search alternates detail/context/full-frame ROI; it does not stay at full frame forever. ROI stays fixed during candidate confirmation.
- Immutable original template plus at most5 recent verified templates, `gallery_update_similarity:0.8`. Nothing is added during rejection/probation. Recovery similarity0.65, texture requirement and3-observation confirmation remain unchanged.
- Size uncertainty during loss grows with elapsed source time, `recovery_scale_rate:0.7`, bounded by `recovery_scale_limit:8` in linear size. This permits a genuinely smaller returning object without removing structural and motion checks.
- `scripts/check_late_recovery.py`: same generated frames before/after against2503b9c. Object absent14 timestamps/seconds, returns atframe20: old never recovered, new recovered at22 for both12px and6px object. Distractor-only same-color/different-pattern sequence: zero false locks. This is simulated source time, not a measured camera recovery delay.
- No raw recording of the user's latest failure was available; actual camera recovery is not yet verified. Large relocation outside the motion gate, unrecognizable new views, or textureless/tiny targets can still remain LOST. No guarantee for indistinguishable objects.
