# OBJECT identity and asynchronous SAM verification

Branch `feature/identity-confidence`; pre-edit backup commit `0370c93`.

## Diagnosis, measured rather than inferred

`runtime/20261012_122056_frames.csv`: AMBIGUOUS occupied 33.37% of observed time, WAIT_SELECTION 47.98%. Of 677 SAM observations, 347 reported ambiguous_sam_objects; 69 ambiguous continuity observations still had coordinates. Thus missing coordinates were not universal. Old logs lack proposal masks and C/W identity labels: they cannot prove the nested-mask hypothesis or identify each false lock. Original core mean/p95 was 49.84/56.66 ms; 507 anchor checks and 541 point decodes accompanied 678 core calls. Probe overhead is measured, not inferred from GPU utilization.

## Changes

| Files | Role |
|---|---|
| scripts/recovery/continuity.py, identity_scores.py, native_memory.py | Group nested mask hypotheses; preserve multiple components supported by previous mask; temporal/anchor gates; confidence and TRACKING_LOW hysteresis; freeze untrusted memory; bounded coast and existing probation |
| scripts/recovery/probe_worker.py, scripts/live_tracker.py | Isolated second instance of the SAME SAM checkpoint, bounded verification job, separate CUDA stream, cloned encoder features and original conditioning anchor; main stream synchronization only |
| scripts/camera.py, mask_display.py, realtime_video.py | High-priority tracking stream, camera options/rate warning, confidence/gain HUD, candidate snapshot handling and LOW display support |
| scripts/recovery/selection.py | Colored contours, interior labels outside HUD, exact-frame manual selection, C/W score logging with legacy CSV compatibility |
| scripts/sam_memory.json | Grouping, confidence/hysteresis, probe cadence/expiry settings; existing identity acceptance thresholds unchanged |
| scripts/video_playback.py, session_audit.py, recovery/metrics.py, eval_video.py | Component evidence, reasons/timing CSV, LOW accounting, GT-visible ambiguity/coordinate metrics and score distributions |
| scripts/calibrate_confidence.py | Reports C/W distributions and proposes a cutoff only with both labeled classes; never edits config |
| scripts/check_identity_confidence.py, check_object_continuity.py, check_async_probe_gpu.py, check_identity_realtime.py | Geometry/state/label tests; CUDA feature-reuse check; generated A-D scenes and full asynchronous transport regression |

SAM/SAMURAI remains the only learned model. Optical flow, mask geometry and Kalman are classical. Color is not part of fused confidence or the OBJECT friend identity decision. The original anchor remains pinned; low-confidence/ambiguous/repaired observations do not update SAM memory/gallery. Confidence is an evidence score, **not a calibrated probability**. C/W data are absent, so new confidence thresholds are provisional. No claim that simply raising this score fixes identity.

Normal anchor/point probes run at most once per configured period on the separate verifier, with top-four hypotheses and batched point decoding. Current-frame features are cloned because CUDA graphs overwrite reusable buffers. Epoch, timestamp and support checks reject unrelated/stale evidence. Background results can veto identity but cannot silently initialize today's mask from an old image. Manual numbered choices display their exact probe frame (explicitly labeled), then reset using that frame. SMALL crop recenter/seed verification and signed-point repair remain synchronous because they need matching coordinates/current-frame correction.

## Run and collect evidence

```powershell
cd D:\Sam
& D:\Sam\.venv\Scripts\python.exe .\scripts\camera.py --source 0 --sam-only --target-mode object --recovery friend --cam-size 640x480 --csv .\runtime\identity_live.csv --profile .\runtime\identity_live.profile.json --record-raw .\runtime\identity_raw.mp4 --save-video .\runtime\identity_overlay.mp4
```

S/R select/reselect, W wrong, C correct, Q/Esc quit; 1–9 explicitly selects a displayed candidate. Raw video has a timestamp sidecar and uses the existing bounded writer. Add `--timing-details` for GPU-stage diagnostics (extra synchronization); `--sync-probes` is an A/B diagnostic switch, not the recommended live mode.

Optional camera controls: `--no-cam-auto-exposure --cam-exposure -6`. Exposure units and successful application depend on the device/backend; inspect printed requested/readback values. Do not assume -6 fits every room. Driver FPS is not measured FPS. Distinct live images cannot exceed camera delivery; redrawing the same image does not count as new frames. A warning reports camera delivery below 25 FPS.

```powershell
& D:\Sam\.venv\Scripts\python.exe .\scripts\session_audit.py .\runtime\identity_live.csv
& D:\Sam\.venv\Scripts\python.exe .\scripts\calibrate_confidence.py .\runtime\identity_live.labels.csv
& D:\Sam\.venv\Scripts\python.exe .\scripts\eval_video.py --cases .\scripts\eval_cases.json --variants 3 --recoveries friend --mode realtime --output .\runtime\identity_eval
```

Use eval cases pointing to the actual labeled A-D clips before treating results as acceptance. GT is sparse XYXY with explicit visible/occluded boundaries; visibility is not interpolated across an unknown boundary. `--recovery current` compares the legacy alternative, not this task's exact pre-edit revision. `--sync-probes` isolates scheduling, not the geometry change.

## Measurements and limits

52 regression tests passed, followed by 19 focused tests including the newly added legacy/new feedback CSV test. GPU reuse test prevents the verifier from invoking its image encoder after warmup. Generated same-color scene, 24 observations: valid coordinates 24/24 for sync and async; pipeline p95 initially 144 -> 69 ms, core p95 72 -> 60 ms. This compares scheduling with the same new identity policy, not real-camera accuracy.

Full asynchronous generated 640x480/30 FPS input, 120 frames, save-video/CSV enabled: Live **28.47 FPS**, SAM updates **11.00 Hz**, core compute mean/p95 **58.85/74.09 ms**, mean SAM age **145.56 ms**, display p95 **27.31 ms** (<33.33 ms input period). TRACKING 3.00 s, TRACKING_LOW 0.97 s; no lock loss in this short static scene. The strict >=30 Live and >=12–15 SAM Hz requirements are **not met**. Dense display flow p95 was 24.14 ms. Separate probe wall times include GPU contention and are not additive exclusive GPU measurements.

Second scheduling run (saved JSON): p95 pipeline sync/async **124.61/92.57 ms**, core **61.87/73.79 ms**, still 24/24 valid each. Timing varies; do not treat the initial 69 ms as a guaranteed bound. Allocated PyTorch peak was 510 MiB, excluding driver/reserved allocations.

Generated GPU identity fixtures (22 observations each; these are not the user's natural A-D clips):

| Scene | Identity switches | Unnecessary loss | Correct when visible | Ambiguous when visible | Valid xy when visible |
|---|---:|---:|---:|---:|---:|
| A: distractor enters/leaves | 0 | 0 | 100% | 0% | 100% |
| B: target removed/returns | 0 | 0 | 100% | 0% | 100% |
| C: crossing/occlusion | 0 | 1 | 36.36% | 18.18% | 36.36% |
| D: adjacent objects | 0 | 0 | 100% | 0% | 100% |

Crossing is still too conservative and fails the valid-coordinate goal. One B return was recognized on the first sampled return frame (0 frame delay); one synthetic event is not a general recovery-time statistic. No matched pre-edit natural benchmark exists.

The old live session is not a matched before/after benchmark: different scene, camera FPS, long waiting intervals. Do not compare its mean age directly to this generated clip. Test artifacts are under `test_results/identity_confidence/` (JSON, CSV, audit, profile, generated video).

Physical camera, natural same-color A-C, real-person regression and W/C calibration remain unverified. No fabricated false-lock counts or natural accuracy percentages. Nested masks caused by true overlapping distinct objects remain difficult; complete occlusion of visually identical objects is not identifiable with certainty. Conservative SEARCHING/manual choice is preferable. Automatic recovery in the async path currently requires a current temporal SAM candidate; old probe-only masks are offered for explicit manual choice, not auto-seeded. This can delay recovery. Tiny masks entirely behind HUD have contours but may lack an interior number; crop transitions can still cause latency peaks. These limitations remain acceptance work, not features silently removed.
