# Memory admission, realtime checks and preparation for fine-tuning

Branch `feature/quarantined-memory`; backup `969f621` precedes changes. Only SAM/SAMURAI is used in the default pipeline. No checkpoint was overwritten and no foreign sam2 package was installed.

## Observed cause and memory audit

`identity_live.audit.json`: display TRACKING0.592s, TRACKING_LOW6.880s, AMBIGUOUS4.031s; one lock loss, no return. The loss reason is `sam_quality_low`; subsequent transitions include `probe_support_changed` and `probe_pending`. These two conditions describe unavailable verification, not demonstrated competing identities. There is no reviewed identity GT proving color itself caused the loss. No claim that a histogram threshold caused it.

The adapter stores mask-memory features/position encoding, object pointers, predicted IoU, objectness and SAMURAI Kalman quality. Original OBJECT code already froze commits for LOW/AMBIGUOUS/COAST, but lacked delayed admission and a complete rebase at recovery. Display-flow masks remain outside memory. CUDA graph outputs are now cloned before retention to avoid reusable-buffer aliasing.

## Changed behavior

- SAM-only AUTO selects ObjectContinuityRecovery; explicit PERSON continues its existing policy.
- Original anchor is permanent. A limited number of explicitly verified refresh anchors can be promoted (four by default); when the permanent pool is full, new frames remain ordinary quality/recent memories. Permanent anchors are never silently evicted. Recent bank <=24, quality conditioning <=4 additional entries; overall conditioning bounded <=10 including active prompt. No unlimited anchor promise.
- Quarantine retains <=3 compact snapshots. At least three clean observations **and** 0.2 seconds of uninterrupted eligibility precede admission. Any LOW/AMBIGUOUS/COAST/SEARCHING, repair, component filtering, insufficient confidence or intersecting known distractor clears pending admission. Only SAM-generated, identity-verified observations enter it. Predicted masks never do.
- Fused confidence continues to report evidence, not a calibrated probability. Poor distractor margin lowers control confidence without overriding an independently passing continuity gate. SAM quality, prototype and temporal checks still apply; identity thresholds were not simply lowered.
- Gates use measured frame dt: prediction/center allowance scale with velocity; IoU/area residual tolerances tighten at shorter dt and never loosen beyond the previous safety envelopes. Confirmation, probation and high-confidence hysteresis use elapsed seconds. The minimum quarantine observation count remains an additional evidence requirement.
- Only actual competing proposals produce AMBIGUOUS. Pending/stale-support probes keep recovery SEARCHING, or ongoing valid tracking LOW. After probation succeeds, rebuild from permanent anchors, reset predictor/Kalman/history/index together, reuse the current encoder and re-encode the verified SAM mask. Reject a disagreeing rebase; restart flow on the accepted current mask. Source frame number and SAM frame index remain separate telemetry fields.
- Telemetry uses bounded replace retries/backoff in the existing disk worker, retains pending latest state and journals failed attempts to `.retry.jsonl`. Frame CSV remains the event history; the JSON channel is intentionally latest-only, not delivery of every frame. Persistent filesystem failure is reported, not claimed solved.
- Camera HUD distinguishes cumulative `avg` FPS from recent measured capture FPS. Driver-requested FPS is not actual delivered rate. Distinct live frames cannot exceed camera delivery.

## Realtime references and experiments

See [reference/SOURCES.md](reference/SOURCES.md) for exact sources, licenses and the porting decision table. The supplied Gy920 ZIP was read selectively; no entire predictor was transplanted. Original CUDA graphs and SAMURAI motion-based selection remain enabled by default. Preprocessing keeps explicit BGR->RGB. `--sam-resize bicubic`, `--sam-jpeg-quality 90`, `--sam-precision bf16` are A/B switches; default remains linear/no extra JPEG/FP16 until a matched test warrants change. BF16 disables the FP16-specific graph wrappers. Current device reports BF16 support, no compiled Flash Attention, no installed Triton; foreign `vos_inference` compilation is not blindly applied.

## Run / label

```powershell
cd D:\Sam
& D:\Sam\.venv\Scripts\python.exe .\scripts\camera.py --source 0 --sam-only --cam-size 640x480 --csv .\runtime\memory_live.csv --profile .\runtime\memory_live.profile.json --record-raw .\runtime\memory_raw.mp4 --save-video .\runtime\memory_overlay.mp4
```

S/R selects, W/C labels wrong/correct, Q/Esc exits, 1–9 confirms a candidate. Explicit `--target-mode person` retains PERSON behavior. Camera exposure controls remain available; their units/acceptance depend on the driver. For detailed stage timing add `--timing-details` (extra synchronization overhead).

## Fine-tuning is NOT performed yet

The user confirmed no fine-tune data. First create reviewed target-only masks, including explicit empty masks during absence. Do not train on unreviewed tracker predictions. A simple annotation tool is provided:

```powershell
& D:\Sam\.venv\Scripts\python.exe .\scripts\annotate_target.py --source .\runtime\memory_raw.mp4 --output .\annotations\object_train_01 --split train --kind object
```

B requests a SAM proposal from a selected box; left mouse paints foreground, right mouse erases; E explicitly marks target absent; **S approves/saves** the frame; N skips; Q exits. Exclude same-color distractors. Files are saved only after approval; existing annotation directories are not overwritten. Repeat for separate validation videos and person train/validation videos, varying lighting/background. Keep contiguous groups of 4–8 reviewed frames for temporal training. Source paths identify videos for leakage checking: renamed/copied versions of the same source must still be kept in the same split by the annotator.

```powershell
& D:\Sam\.venv\Scripts\python.exe .\scripts\prepare_finetune_data.py .\annotations\object_train_01 .\annotations\object_val_01 .\annotations\person_train_01 .\annotations\person_val_01 --output .\datasets\reviewed_v1
```

Exporter checks review flags, empty-mask labels, dimensions, single-target IDs, person coverage and source-video train/val separation. It creates DAVIS-style JPEGImages/Annotations, train.txt/val.txt and a manifest. Tests use generated images, not the user's video.

`training_bundle/memory_tiny512_candidate/` contains a **draft**, based on the official local MOSE Hydra training config and reference Tiny512 architecture, with its Apache license. It uses official SAM2Train/runner, frozen image encoder, memory attention/encoder and mask decoder trainable (including its objectness head), 4 frames, batch1, LR1e-6, bf16 AMP and TensorBoard configuration. Launcher has explicit `--run`, mandatory dataset/checkpoint/output and refuses overwrite. No dataset or weights are bundled. Use a separate Linux training environment, not a pip replacement in the live .venv.

Model instantiation validation is blocked by missing `tensordict` in the live environment; no dependency upgrade was attempted. The candidate is therefore **not validated as runnable training**. Official template is train-only; gradient accumulation, validation-IoU logging and early stopping are not added or claimed. Those need implementation/verification against the reviewed dataset before training. No claim that 6GB is sufficient or insufficient for the proposed training run: inference VRAM does not establish training VRAM. No new checkpoint exists to compare or promote.

## Files

Runtime: `scripts/camera.py`, `live_tracker.py`, `sam_preprocess.py`, `recovery/continuity.py`, `recovery/quarantine.py`, `sam_memory.json`, `video_playback.py`, `eval_video.py`.

Data/training preparation: `scripts/annotate_target.py`, `prepare_finetune_data.py`, `export_finetune_bundle.py`, `scripts/training_tools/*`, `training_bundle/memory_tiny512_candidate/*`.

Verification: `scripts/check_quarantine.py`, `check_object_continuity.py`, `check_memory_soak.py`, `benchmark_memory_preprocess.py`, `check_training_data.py`; `PROGRESS.md`, this report and reference provenance. Existing async display, CSV/raw/save-video paths and shortcuts were regression-tested; actual live identity behavior still needs annotated scenes.

## Acceptance status

Generated A/B/D: zero identity switches, 100% valid-visible coordinates. Generated crossing C: zero switches, one unnecessary loss, only36.36% valid-visible; AMBIGUOUS18.18% previously vs4.55% after separating unavailable probes. This is not a success on the continuity-coverage goal. The generated case names are not the user's real A–D recordings, and there is no person-video GT benchmark yet.

Capture-only DSHOW/MJPG640x480:110 frames/5.025s =21.89FPS, driver30FPS, buffer1 rejected. No camera images were recorded by this benchmark. Ten-minute soak and preprocessing A/B results are recorded separately below when complete; invalid/suspended runs must not be treated as uninterrupted validation.

MSMF measured150 frames/5.014s =29.92FPS, but rejected MJPG. To reproduce with explicit settings add `--camera-config .\configs\camera_msmf_measured.json` to the run command. The default backend remains unchanged. These camera measurements do not measure identity correctness.

### Completed measurements

Repeat soak: **600.06 seconds,10415 frames,17.36 tracker Hz**, p9570.22ms, longest step170.20ms, no step>5s. After60s warmup, allocated VRAM450.20–470.13MiB, reserved828MiB throughout; recent24/conditioning5/quarantine2. End-minus-start allocated after60s was -4.36MiB, with bounded work-buffer fluctuations rather than increasing retained state. This is a generated static scene on CUDA, not a ten-minute camera accuracy test. The first run (2619.9s/2661frames) suffered an uncharacterized long stall and is excluded.

| Generated full realtime/save-video fixture | Prior reported baseline | Latest run |
|---|---:|---:|
| Live FPS |28.47|29.34|
| SAM updates Hz |11.00|12.33|
| Core SAM p95 ms |74.09|65.12|
| Mean SAM age ms |145.56|135.56|
| Display p95 ms |27.31|19.46|
| Valid xy |120/120|120/120|

Same procedural fixture, separate runs; scheduling variability prevents attributing all improvement to one patch. Latest states: TRACKING2.333s, LOW1.633s, AMBIGUOUS0. Tracking+LOW100% in this short fixture. Strict30FPS still not achieved. Probe expiration remains a reason for reduced-gain tracking; new `probe_wait_reason` distinguishes not-ready/expired/epoch-change/support-change instead of hiding them under a generic continuity reason.

Preprocessing A/B,32 generated noisy frames each, all32/32 valid:

| Variant | Mean/p95 pipeline ms | Consecutive mask IoU |
|---|---:|---:|
| FP16 linear, no extra JPEG |47.68 /66.45|0.99284|
| FP16 bicubic |46.69 /56.44|0.98373|
| FP16 linear, JPEG90 |46.19 /56.52|0.99063|
| BF16 eager, linear |49.06 /63.27|0.99197|

No variant improved mask stability on this small test; timing alone is not sufficient to change defaults. There is no fine-tuned-checkpoint ablation row because no reviewed dataset/checkpoint exists. Natural A–D/person metrics, recovery jitter under real occlusion and checkpoint-regression acceptance remain outstanding. Test artifacts: `test_results/quarantine/` and the latest `test_results/identity_confidence/realtime*`; source camera logs under `runtime/camera_*_20261013.log`.
