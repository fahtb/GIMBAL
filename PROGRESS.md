# SAM-only tracking — 2026-10-07

## Recovery latency follow-up (2026-11-13, latest)
- Branch fix/recovery-latency from clean b0a4ff9. User screenshots show REDETECTING/search_interval_wait, not evidence of slow GPU alone. Prior runtime audit:57.52% SAM-state time REDETECTING,6lock losses/5resumptions; several losses start at kalman_gate. Prior profile SAM mean51.15/p9559.21ms. Current frame CSV files were empty when inspected, so no full replay of this reported session.
- Fixed local search doing full-image matchTemplate then masking scores: search now crops before matching, restores source coordinates, caches bounded resized templates and deduplicates repeated template sizes.
- Added quarter-turn template proposals/identity comparison; orientations cycle under a bounded search cadence, and final confirmation checks competitors across all orientations. Thresholds/margin/three confirmations unchanged. Newly lost tracks probe every50ms when compute permits; after2s, four orientation passes share the prior250ms search interval. This is a scheduling budget, not a guaranteed execution time.
- Motion-gate exception still requires>=.8 structural agreement plus LK and spatial overlap, but compares the most recent trusted appearance instead of requiring agreement with an old view. No covariance or identity cutoff lowered.
- CPU identical local-search benchmark30 measured repetitions:mean45.66->6.57ms,p9557.63->8.66ms. GPU relocation smoke recovered frame25 afterreturn22 vs previousframe28 (15Hz source:0.2vs0.4s). Synthetic only. Partial/occlusion/same-color/exit regressions unchanged,0false-mask frames.13CPU tests pass including crop coordinates,rotation and cross-orientation distractor margin.
- Files: scripts/hybrid_tracker.py,check_hybrid.py,video_playback.py; configs/hybrid.json; PROGRESS.md,HYBRID_TRACKING.md. Added proposal_search_ms/search_rotation CSV fields.
- Artifacts: test_results/hybrid/search_latency_ab.json,recovery_latency_scenarios.json,gpu_smoke.json,latency_person.json. Need labelled raw replay of user's can/fast hand motion; grayscale appearance remains limited under out-of-plane rotation, blur and visually indistinguishable objects. Do not claim natural recovery success or 30SAMHz.
- Final natural person60frame smoke: both60TRACKING; enhanced17.36processingHz,mean57.59/p9562.48ms,peak353.87MiB. Baseline14.22Hz in preceding run; SAM core itself varied60.12->44.77ms, so do not attribute the whole FPS difference to these CPU search changes. Final cross-orientation competitor check retained relocation recoveryframe25;13CPU tests passed.

## Scope and checkpoint
- User requires Step 1 diagnostics and a report before Step 2 implementation.
- Initially no Git repository or PROGRESS.md. Created a local scripts-only baseline.
- Read only related scripts; do not inspect sam2, virtualenv, checkpoints or video contents for architecture.
- Legacy factory constructs YOLO; TargetTracker selects PERSON or OBJECT; OBJECT search uses YOLO/DINO.
- LiveTracker returns immediately after its lost latch. ObjectTracker stops calling SAM after LOST and delegates search.
- Realtime HUD divides 1000 by max(0.001, inference_ms), producing 1,000,000 FPS for idle inference.

## Step 1 implementation
- Baseline commit: 838680c (scripts only; no videos, weights or virtualenv tracked).
- `--sam-only` defaults ON and returns before all legacy detector imports, including `--target-mode person`.
- `--no-sam-only` restores legacy routing. `--keep-running` is SAM-only; `--lost-stride K` defaults to 1.
- New CSV columns: predictor_called, object_score_logits, predicted_iou, sam_observation_valid, keep_running.
- Skipped/idle inference has no invented scores. Raw observations remain available while status is SEARCHING.
- SAM idle now shows 0 FPS; profiling does not count a latched return as SAM inference.
- SAM-only HUD shows raw SAM observations instead of a misleading unavailable ReID score.
- Existing display/flow/ROI mask/contour, keys and output paths are retained.
- IMPORTANT: Step 1 is diagnostic, not recovery: a restored positive observation remains SEARCHING until a user selection. No identity acceptance is implemented here.
- Existing LiveTracker memory behavior remains unchanged; no new anchor/gallery policy has been added.

## Measurements (2026-10-07)
- test3.mp4, offline, width 960, frame 0 bbox 674.12,305.29,201.18,107.65, first 30 frames: 30/30 TRACKING, 0 losses, 22.14 processing FPS (not display FPS).
- Controlled test: repeat frame 0 ten times, five black frames, restore the same frame fifteen times. Not a natural reappearance benchmark.
- Both modes lose on artificial frame 10: object logit -7.7461, predicted IoU 0, mask area 0.
- Without keep-running: 0 predictor calls in the 15 restored frames; scores unavailable.
- With keep-running stride 1: 15/15 predictor calls and 15/15 valid observations after restoration. First restored frame 15: logit 8.58594, IoU 0.970703, area 12135 pixels. No new prompt was passed.
- Evidence: runtime/sam_only_step1_30.csv and runtime/sam_only_occlusion.csv (not committed video/log artifacts).
- Tests: check_sam_only 4 passed; check_video_playback 6 passed; check_smooth_video 7 passed (17 total). Smooth regression covers pause/step/selection, wheel zoom, labels, CSV and video queues.

## Reproduce
```powershell
.\.venv\Scripts\python.exe scripts\camera.py --source videos\test3.mp4 --mode offline --resize-width 960 --init-frame 0 --init-bbox 674.12,305.29,201.18,107.65 --no-display --max-frames 30 --keep-running --csv runtime\sam_only_step1_30.csv
.\.venv\Scripts\python.exe scripts\check_sam_keep_running_gpu.py --source videos\test3.mp4 --init-bbox 674.12,305.29,201.18,107.65 --output runtime\sam_only_occlusion.csv
```

## Pending before Step 2
- Await exact mountain-road video filename and initial frame/bbox with selection resize width. test1/test2 identities are unconfirmed; do not reuse test3 coordinates for a different clip.
- Therefore natural reappearances near frames 600/900 have NOT been measured; no recovery-rate/false-positive or SMALL performance claims.
- Report Step 1 results to user before memory pinning, prototype recovery, SMALL/tiled search or ablation work.

## Step 2 implementation (next checkpoint)
- User reports persistent loss/small-target failure; proceed from the previously reported diagnostic checkpoint.
- Default SAM-only now uses SAMMemoryTracker; `--sam-diagnostic` reproduces Step 1 and its optional `--keep-running` behavior.
- Prompt anchor + four best-quality conditioning frames are retained alongside bounded recent memory. Identity/quality gates prevent weak frames being committed.
- Multi-level masked prototypes reuse the tracking encoder output. Anchor is permanent; gallery capped at 24, redundancy removal preserves entry zero.
- Search uses cosine heatmap peaks and batched SAM point prompts on cached features, then SAM IoU/stability, Lab histogram and size gates. The existing inference thread owns all model operations; display never calls search.
- Strong/normal/single-candidate streaks precede mask-prompt seeding and five-frame orange probation. Failed probation rolls back candidate memory; accepted recovery preserves gallery and anchor.
- Thresholds: scripts/sam_memory.json. `--debug-search` displays faint candidate masks. CSV includes probation/gallery/pinned-memory/Lab/attempt counters.
- Controlled occlusion now completes one verified reacquisition; 30 original test3 frames remain 100% TRACKING, 17.63 offline processing FPS.
- Tests: 4 memory/overlay + 4 model-isolation/diagnostic + 7 smooth UI + 6 video playback passed.
- Still pending: natural-video recovery accuracy, source-resolution SMALL crops, tiling, explicit Kalman-region grid, full ablation. No claimed >=80% natural recovery or >=30 display FPS yet.
- Located saved user selections: test1 has tiny boxes (~7x13 px at frame 0); test2 has vehicle box 431,331,127,149 at frame 0 and a reselection at frame 536. Use these recorded coordinates instead of guessing from screenshots.

## Continuation checkpoint (2026-10-07)
- Confirmed fixtures: test1 = 4K beach, test2 = mountain-road car, test3 = original 300651. Saved selections are in scripts/eval_cases.json.
- Added native SourceFrame/crop transport, point prompts, dense ROI mask flow for tiny targets, Kalman/contrastive SAM + classical Lab search proposals, and bounded native tiled search. No non-SAM model added.
- Beach first 60 frames: diagnostic baseline 3.33% TRACKING; native crop 100% SMALL. Point initialization (292,299 at width960) also passed 20 frames. These are prefix tests, not whole-clip accuracy.
- Actual GUI + video output checkpoints: test3 180 frames, 100% TRACKING, Live 32.33 FPS, SAM compute 15.64 FPS, age135.97ms; beach120 frames, 100% SMALL, Live33.56, SAM compute12.89, age145.70ms. Later search changes still need final regression.
- PERSON prefix60: baseline and SAM-memory both100% TRACKING. Full120 regression pending.
- Mountain940 baseline diagnostic:50.43% TRACKING, lost1/reacquired0. Memory experiments range63.30-73.72%, with substantial loss/recovery churn; latest conditioning-prompt checkpoint63.30%, lost11/reacquired10, processing14.16FPS. NOT accepted as an accuracy improvement: visibility intervals and identity labels are absent.
- Corrected initial area gate rejecting valid shrinking car masks (anchor ratio0.0488 atframe594); recent-area collapse gate now distinguishes gradual scale changes. Isolated proposal decoding from tracker Kalman state.
- Tiled synthetic relocation is still FAILING:0 accepted frames of148 restored frames. Known-location probe finds the pasted person and tight-crop identity0.879/Lab0.912, but stability0.788 narrowly misses0.8. Testing SMALL-only stability0.75 did not alone fix end-to-end retrieval; investigating proposal ranking/refinement coverage. No natural recovery claim.
- Full4K tile sweep is approximately60 tiles, budget4 at2Hz (~7.5s minimum), so <=2s far re-entry is not met.
- Current CPU SAM tests12 pass. Pending: crop recenter identity/memory safety, A/B crop comparison, final UI regression, annotated natural recovery evaluation. Keep full mask/contour and legacy --no-sam-only routing.

## Friend recovery — Step 1 only (2026-10-10)
- Read current SAM-only factory, memory/crop/tile acceptance and friend README/entry scripts. No executable code changed, no tests/video/model loading, no writes to D:\Gimbal.
- Friend demo.py defaults to SAM2.1 Small and keeps native predictor propagation after empty masks; no custom recovery gate found at entry level. Core memory/identity policy remains unverified because sam2/ reads were explicitly excluded.
- Current default is NativeCropTracker + SAMMemoryTracker, not the old YOLO/HSV path. Found strong acceptance missing margin in both sam_memory.py and sam_tiles.py; motion penalty is not a common hard gate. These are hypotheses to test, not measured causes.
- Wrote PORTING_PLAN.md with evidence, comparison decisions, online constraints, proposed A/B interface and acceptance criteria. Need narrowly scoped core-read exception to establish any actual friend-specific algorithm before porting.
- Git master contains existing tracked modifications and untracked files. Preserved all. Branch feature/friend-recovery and scoped baseline commit are deferred until plan approval, before any code edits; never stage videos/checkpoints/venv or blindly stage the repository.
- STOPPED at the requested approval gate. Steps 2–4, baseline metrics and performance claims have not been executed for this task.

## Same-color correction authorized by user screenshots (2026-10-10)
- User requested actual SAM object memory after showing pen/shoe swaps. Created feature/friend-recovery and baseline commit d30516f (scoped Codex commit identity; no global Git settings changed).
- Implementing a separate native-memory policy: no color acceptance, no candidate-mask re-prompt, immutable selected anchor verification, shape guard and uncommitted probation. Retain the existing current policy for A/B and all live transports.
- Only friend entry behavior is reused/attributed; no sam2 source/weights from D:\Gimbal read or copied. Validation pending; no accuracy or FPS claim yet.
- GPU A/B: two procedural fixtures (white distractor separate, then moving into the old target location), 60 frames each policy/fixture. Both old and new made 0 false acceptances; native recovery returned the original after 0.233 source seconds versus old 0.200. This DOES NOT reproduce the user's pen/shoe false lock and is not evidence of reduced real false-ID rate.
- CPU regressions passed: native identity/probation 5, memory 5, SAM-only isolation 4, smooth transport 7, playback 6, SMALL crop 2, tiles 2. Display FPS >=30 not established; no performance acceptance claim.
- Final native CPU suite has 6 tests including replay of the old acceptance gate; every tentative mask remains outside committed memory. Added crop-anchor revalidation, bounded-motion guard and additional CSV diagnostics.
- Full headless realtime/save-video fixture completed with one recovery: Live23.51, SAM updates3.43, recovery first visible at source6.53s vs return3.00s. Does NOT meet >=30FPS/<=2s acceptance; separate OpenCV1-thread experiment was worse and was not applied. Actual camera/pen-shoe/annotated natural clips remain unverified.
- Details and run command: MEMORY_RECOVERY.md. No claim of a measured reduction in natural false identity switches; procedural GPU baseline also had zero false locks.

## Object continuity 2026-10-11 - diagnostics
- Branch feature/object-continuity, baseline 1a18235. Added bounded raw writer and replay timestamp sidecar before changing identity behavior.
- Static finding: immediate rejection on anchor disagreement; no flow-mask continuity gate. Real A-D clips not supplied, so observed cause and baseline accuracy remain unmeasured.

- Implementation: OBJECT-only continuity policy under existing --recovery friend; PERSON unchanged. Component gate, foreground LK prediction, signed-point repair without memory commit, bounded coast, recovery margin/streak/probation, numbered manual candidates. Camera reuses existing MaskDisplay and output worker. Validation pending.

- GPU procedural A-D reproduced old B swap and old C/D loss. Found and fixed coast LK following occluder; final A-D: new identity switches 0 each, unnecessary loss A/B/D=0 C=1, correct-with-distractor A/B/D=100% C=22.2%. Crossing remains conservative; not natural video accuracy. CPU continuity tests 11 passed.
- Initial full 640x480 realtime fixture Live25.02 FPS; scheduler drift corrected and retest underway. No >=30 FPS claim.

- Final regression suite: 45 tests PASS; py_compile and git diff --check PASS. Camera UI/output exercised with mock capture; raw recording timestamp roundtrip and signed SAM decoder exercised. No physical-camera accuracy measurement.
- Final full realtime fixture: Live29.67 FPS, SAM updates10.40/s, SAM compute19.38 FPS, age163.05ms, strict TRACKING82.3%, one skipped source frame. Strict >=30 requirement NOT met. Natural A-D/person/object clips and annotations remain pending.
- Added OBJECT_CONTINUITY.md with exact run/record/replay/evaluation commands, GT schema, file list, measured A-D before/after and conservative crossing limitation. All earlier features retained; no dependencies/checkpoint changes.

## Camera metrics correction from user logs
- Four attachments show the same bare camera.py launch: target-mode auto, preceding native policy, not OBJECT continuity. Capture ~19.8 FPS, Live19.75, SAM updates8.78/s, strict TRACKING22%, reacquired13; these are not accuracy labels. DSHOW buffer=1 rejected; MJPG accepted, nominal driver30 FPS.
- Fixed misleading HUD age/latency that used the new camera frame timestamp: preserve SAM evidence time, expose image latency separately, show actual requested mode/policy and camera/live FPS. No identity thresholds or camera exposure settings changed.
- 2 timing contracts and 13 continuity/UI checks passed. No physical-camera remeasurement; old pasted HUD latency cannot be interpreted as SAM end-to-end latency.

## Identity confidence 2026-10-12
- New branch feature/identity-confidence; baseline 0370c93 preserves prior diagnostic work. Latest session audit: AMBIGUOUS33.37% including WAIT_SELECTION47.98%; 347/677 SAM observations report ambiguous_sam_objects, 69 continuous ambiguous observations retain coordinates. No W/C labels; old logs cannot prove mask-part duplication or per-event false locks.
- Core SAM mean49.84/p9556.66ms; 507 anchor probes and 541 point decodes for678 core calls. Implementing mask containment grouping, transparent confidence/LOW hysteresis, isolated same-SAM verifier consuming cloned features, and camera/label telemetry. Accuracy/performance validation pending; no threshold calibration claim.
- Identity stage: nested-support grouping and supported disconnected components; fused evidence/LOW hysteresis; original anchor and hard identity gates retained. Color remains diagnostic. Existing same-frame signed repair and bounded coast preserved.
- Scheduling stage: bounded verifier on isolated same-SAM state/stream reuses cloned frame features; stale/epoch/support checks; current tracking stream prioritized. Crop transitions retain synchronous coordinate-safe verification. Old probe masks only support explicit manual snapshot selection.
- UI/diagnostic stage: confidence/gain and proposal/reason CSV, colored interior candidate numbers, C/W confidence logging, camera size/exposure flags, aligned camera/live rate accounting. Calibration script refuses to invent a cutoff without both labeled classes.
- Validation: 52 regression tests passed plus 19 focused checks after label fix. GPU generated A/B/D valid-visible100%, switches0; crossing C switches0, loss1, valid-visible36.36% (FAIL coverage goal). No natural clips/labels or physical-camera retest.
- Full realtime 120-frame generated640x480 save-video run: Live28.47, SAM11.00Hz, core p9574.09ms, age145.56ms, display p9527.31ms; does NOT meet30FPS/12Hz. Async short scheduling rerun p9592.57ms vs sync124.61ms (initial69 vs144); no deterministic80ms pipeline guarantee. Details/files/commands and limitations: IDENTITY_CONFIDENCE.md.

## Quarantined memory 2026-10-13
- Branch feature/quarantined-memory, baseline969f621. identity_live audit: TRACKING0.592s LOW6.880s AMBIGUOUS4.031s; actual loss sam_quality_low, then pending/support-changed probes incorrectly classified as identity ambiguity. Original outputs already bounded and flow not committed; no admission quarantine and no clean recovery rebase.
- User has no fine-tune dataset yet. Prepare annotation/data checks; do not train on self-generated predictions or claim a trained checkpoint. Reference Gy920 zip inspected selectively, never installed. SAMURAI motion selection remains native.
- Implementation underway: bounded compact-memory quarantine, cloned graph outputs, immutable original/limited refresh anchors, predictor rebase on verified recovery, time-based confirmation and gates, pending probe status, explicit camera rate windows and bounded telemetry backoff. Validation pending.
- Runtime stage implemented; 5 quarantine/alias/rebase/backoff/color-channel contracts pass, plus13 continuity,19 identity (includes continuity),6 native,2 crop,2 tile,4 SAM-only,7 smooth-video,6 playback,2 camera timing and5 audit tests. Generated C ambiguity drops18.18->4.55%, valid-visible remains36.36%; no claim of solved crossing.
- Data stage: manual SAM mask proposal/paint/explicit-empty annotation and reviewed-video split exporter;2 dataset rejection tests pass. Draft official training bundle created with provenance/license. Training instantiation blocked by missing tensordict; no training deps changed, no dataset/checkpoint fabricated, no fine-tune run. Accumulation/validation early-stop are not implemented in draft.
- Measurement stage: initial soak interrupted/stalled (2619s but2661frames) not valid10-minute evidence; repeat underway. Camera-only DSHOW21.89FPS, MSMF29.92FPS over5s; MSMF rejects MJPG, so supplied explicit MSMF test config without changing default. BF16 supported; Flash Attention unavailable; Triton absent.
- Completed repeat soak600.06s/10415frames,17.36Hz,p9570.22ms,no>5s stall; after60s allocated450.20–470.13MiB, reserved828MiB fixed, bank24+5/quarantine2. No unbounded retained-state growth observed. Final continuity suite16 tests passes, including elapsed probation rebase, pending-not-ambiguity and aggregate-low vs hard-continuity separation.
- Latest full generated realtime120frames: Live29.34,SAM12.33Hz,core p9565.12ms,age135.56ms,display p9519.46ms,validxy120/120,TRACKING2.333s+LOW1.633s. Strict30FPS and natural-scene acceptance remain unproven. Preprocessing32-frame A/B: allvalid; bicubic/JPEG/BF16 did not improve consecutive-mask IoU vs defaultlinearFP16. Defaults retained. Full file list/run/training prerequisites/reference decisions: QUARANTINED_MEMORY.md.

## 2026-10-14 Source camera port
- Baseline commit 48f3860 on feature/source-camera-port preserves previous changes.
- Read source README/demo/build/predictor/config/dependencies. Vendored Python/config/CUDA sources with SHA256 manifest and licenses, source directory untouched.
- Main create_tracker replaced; old recovery/probe/class routing no longer constructed. Keep asynchronous capture/display and outputs.
- Model load/warmup succeeded: vendored SAM2CameraPredictor, Tiny512, BF16, torch2.5.1+cu124 RTX3050. Both Tiny checkpoints have identical SHA256. All 41 source files unchanged.
- Integration passed 55 generated frames, box/point/reset/feedback and bounded memory; old tracker modules absent. Aquarium source video 60 frames + masks/CSV/save-video passed.
- Async generated video: Live29.77, SAM17.26Hz, compute p9571.44ms, age103.06ms. Camera opened/tracked/reselected/released/reopened; actual FPS varied15-30, no guaranteed30 claim.
- Native GUI smoke displayed and selected twice/C, but ended before complete injected sequence (initial assertion failed). Full real-camera main-loop callbacks with simulated window S/R/W/C/Q subsequently passed. Smooth-video regression7 passed.
- Fixed live SKIPPED timestamp clock mismatch found in GUI CSV; preserved outputs. Disabled legacy evaluator A/B names because all main paths now genuinely source-only.
- Dependencies all already satisfied; no pip upgrades. See SOURCE_CAMERA_PORT.md for commands/provenance/results/limitations (_C postprocessing unavailable, no compiled graph or identity guarantee).

- HUD simplified per request: SAM 2.1 + display FPS only (offline: per-frame processing FPS), including saved overlay video. Tracking, telemetry/CSV and shortcuts unchanged. Syntax checked for three edited scripts.

## 2026-10-17 Small target ROI
- Backup e9a8bc8, branch feature/small-target-roi; existing compact HUD preserved.
- Found fixed square512 resize, no min-area rejection in predictor, display bounding-area16 gate and UI >3px selection gate. Added small-target path to bypass display gate and >=1px selection.
- Single source model, fixed-coordinate ROI epochs seeded only from previously accepted frame/mask; rejected observations removed from recent memory. Full-frame --small-target off remains unchanged.
- A/B GPU tests: fixed crop-prompt granularity regression by preferring full-frame selected mask; corrected structural alignment for one-pixel small-mask boundaries instead of lowering similarity thresholds.
- 1280x720 synthetic scale correct93.6->100%, same-color30.3->93.9%, wrong37->0; near/lateral100% both. Short occlusion ROI93.9% due probation; distant return45.5% versus100% baseline (conservative rejection; documented limitation).
- Real beach/person40frame smoke both100% TRACKING, not accuracy GT; aquarium40ROI passed. Actual beach4x8px becomes16x32 model pixels on128ROI.
- Async141frame comparison: Live29.80->29.53, tracking20.01->17.50Hz, age89.83->100.95ms. ROI extra reseeding cost disclosed; memory peak354.87MiB in test.
- Camera MSMF read error; DSHOW returned640x480 at2.88FPS/6frames, so live30FPS not verified. No exposure changes. CPU geometry4 + smooth7 + playback6 tests pass.
- See SMALL_TARGET_ROI.md for exact commands, thresholds, evidence and limitations. No checkpoint/vendor/source changes or dependency installs.

## Live ROI edge follow-up 2026-10-17
- Latest camera_frames log: frame144 target y118..139, ROI top101; frame145 empty, epoch cooldown8 delayed necessary recenter. Edge/predicted-edge recenter now bypasses cooldown; only scale changes remain rate-limited. No threshold lowered.
- Same session source SAM mean49.93ms, p9554.76ms, Live7.29FPS; frame timestamps about133ms apart. Camera delivery limits FPS independently of ROI failure.
- Validation: five CPU geometry/display tests pass; same generated 20-frame fast upward motion at7.5FPS timestamps: correct-mask frames14/20 before,20/20 after. Not a replay of the real camera session (raw video unavailable).

## Continued recovery 2026-10-17
- Latest runtime/camera_frames.sam.csv has168 timeout_reselect rows,205 appearance_gate rows. Removed terminal timeout for default auto recovery; after8s searches throttle to0.35s intervals rather than stop.
- Alternate detail/broad ROI search, hold ROI during confirmation, immutable original anchor plus <=5 verified recent structural templates; no recovery threshold reduction. Full-frame --small-target off remains unchanged.
- GPU late-return regression against2503b9c: original never reacquired after timeout; new reacquired both12px and6px atframe22 after return at20 (3 observations), zero false locks on same-color distractor-only sequence. Source time simulated1s/frame, not measured camera timing. Bounded appearance gallery verified.

## Hybrid gimbal P0–P3 (2026-11-13)
- Branch feature/hybrid-gimbal; prior working changes preserved in snapshot 4daed39. Source/vendor/checkpoint untouched.
- CUDA verified: RTX3050 Laptop 6GB, torch2.5.1+cu124, torchvision0.20.1+cu124. Actual source predictor, not SAMURAI's offline internals.
- Added baseline/enhanced engine, external immutable identity + bounded structural views, timestamped Kalman8D, LK forward/backward+RANSAC, quality-gated memory, ranked local/expanded/global recovery with three SAM confirmations. Rejected proposals restore source state; logical ID survives reinitialization.
- GPU lifecycle smoke: global relocation after occlusion recovered at frame28 after return22 (15Hz simulated source); five resets pass. CPU initial4 tests pass. These are synthetic logic checks, not real tracking accuracy.

## Hybrid P4–P5
- Added simulated PID/rate/acceleration/deadzone control, HOLD on missing/stale target, telemetry watchdog and STOPPED output. No real motor driver.
- Added actual SAM logit stability (not identity probability), debug motion/search/identity fields, source-resolution-aware overlays; compact default HUD remains.
- First natural person100-frame benchmark found fixed5px Kalman measurement covariance rejected large deforming boxes. Added explicit relative box-edge noise; threshold25 unchanged. Re-testing before declaring results.
- Benchmarks run sequentially on same natural frames, 512BF16/FP16 and compatible1024YAML. No upgrades/downloaded AI models. Logs in test_results/hybrid; real accuracy requires GT.

## Continuation audit (2026-11-13)
- Read full progress and current integration diffs; no applicable AGENTS.md found. Historical legacy/SAMURAI milestones do not describe today's source-camera factory.
- Snapshot1150dc0 preserves ALL preceding hybrid changes; baseline4daed39 remains available. Unrelated untracked data/source/checkpoints untouched.
- VERIFIED from artifacts and code: 9 hybrid CPU checks; 5 ROI geometry,7 smooth UI,6 playback,Windows locked-file telemetry checks; GPU reset/global-recovery smoke; real camera/CUDA box+point,S/R/H/W/C/Q lifecycle with simulated windows; 300-frame memory stable229.16MiB afterframe50, recentbank7.
- VERIFIED PERFORMANCE (not accuracy): person100 natural frames baseline13.78 vs enhanced12.38 processingHz; both100 TRACKING states; 512 enhanced mean80.76/p9593.13ms. Async person Live23.67/23.72FPS; mean SAM age151.67/163.33ms. Camera capture19.86FPS; buffer1 rejected by DSHOW. Live lifecycle19.19FPS; selection/idle periods included, no annotated identity accuracy.
- IMPLEMENTED/NOT FULLY VALIDATED: optional background motion compensation, grayscale multi-view recovery under real rotations, FP16 stability after final covariance fix, hours-long memory stability.
- INCOMPLETE: final engineering report and threshold calibration with annotated natural clips; K/I/O runtime hotkeys attempted previously but patch NEVER applied (CLI ablations exist).
- NOT IMPLEMENTED: physical gimbal driver, learned ReID encoder, fine-tuning. No dataset; no new AI model installed. Compile disabled: Triton absent.
- KNOWN REGRESSION: synthetic partial occlusion enhanced31/47 visible correct masks vs baseline47/47; frame12 kalman_gate then appearance_gate. New-location return improves15/33->31/33; no false locks in these synthetic fixtures, not natural accuracy.
- CURRENT CHANGE: visible-only texture verification for partial masks in predicted coordinates; UNCERTAIN/HOLD, frozen SAM/gallery/Kalman measurement,1.2s maximum. Whole-object identity thresholds unchanged. Also fix candidate probation inference logging. Files: hybrid_tracker.py,motion_flow.py,mask_display.py,configs/hybrid.json,check_hybrid.py,check_hybrid_scenarios.py. Tests pending below.
- NEXT: run partial/occlusion/distractor GPU A/B, natural person regression, write final report, commit reviewed changes. Do not switch default baseline until real annotated acceptance is available.

## Continuation completed checkpoint (2026-11-13)
- Partial GPU A/B: baseline47/47, previous enhanced31/47, new enhanced47/47 visible masks;13 partial observations are UNCERTAIN/HOLD with unchanged seed/gallery and excluded SAM memory. No identity-confidence claim.
- Long-partial test exposed a timeout bypass into normal updates; fixed it and asserted trusted seed identity stays unchanged through frames12–39. Final long-partial31/47 masks,0false locks; intentionally stops displaying partial evidence after1.2s until recovery.
- Full-occlusion/same-color/exit fixtures unchanged: enhanced31/33 visible masks,0false masks each. Exit baseline15/33. Synthetic logic only.
- Added K/I/O runtime ablations through worker commands and CSV feature flags. Real camera/CUDA test reached PASS for box,point,S/R/H/W/C/K/I/O/Q and final HOLD; callbacks/windows simulated, rendering offscreen. Live19.21FPS in this uncontrolled session. See test_results/hybrid/live_continuation/lifecycle.json.
- Natural person100 regression: both100TRACKING; baseline12.71Hz/mean78.65/p9587.09ms; enhanced11.33Hz/mean88.23/p9597.53ms; peak353.87MiB each. Correct-ID accuracy not measured without GT. Artifact person_continuation.json.
- 11 hybrid CPU +7smooth+6playback passed before final audit-count test.224Python sources parsed,0syntax errors. Corrected event audit to count UNCERTAIN/unconfirmed control-lock loss once and subsequent RECOVERED resumption; focused final tests recorded next.
- Files changed in continuation: configs/hybrid.json; scripts/hybrid_tracker.py,motion_flow.py,mask_display.py,camera.py,realtime_video.py,video_playback.py,small_target.py,session_audit.py,eval_video.py,check_hybrid.py,check_hybrid_live.py,check_hybrid_scenarios.py; HYBRID_TRACKING.md and PROGRESS.md.
- Remaining: real annotated same-color/partial/rotation acceptance, optional camera compensation validation, final FP16 comparison, sustained hours-long soak, physical motor driver (not requested to send commands). Compile unavailable without Triton; no dependency upgrades. Enhanced remains explicit opt-in; baseline preserved.
- Resume priority: collect/replay reviewed natural GT with eval_video.py --engine both; improve appearance robustness only from measured failures. Do not lower identity thresholds or label synthetic accuracy as real performance. Run commands/configuration, evidence, limitations and training-data plan are in HYBRID_TRACKING.md.
- FINAL checks:12hybrid +5session-audit pass; git diff --check passes. Prior7smooth+6playback pass. No remaining known failing test in the active continuation scope. Performance goals20–30inferenceHz/consistentp95<150ms and natural identity accuracy remain unachieved/unverified; do not mark them complete.
