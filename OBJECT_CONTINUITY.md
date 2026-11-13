# OBJECT continuity — 2026-10-11

Branch `feature/object-continuity`; baseline commit `1a18235`. SAM2.1 Tiny,
SAMURAI memory/motion and CUDA graph retained. No additional learned model,
dependency or checkpoint change. Explicit `--target-mode object --recovery friend`
enables the new policy. `person` and `auto` retain the preceding native policy;
there is no semantic PERSON/OBJECT classifier in SAM-only mode.

## Diagnosis (code and procedural reproduction, not the missing real clips)
1. The preceding policy immediately searched after one anchor disagreement.
2. It did not compare SAM's observation with a transported previous mask.
3. A procedural occlusion with a same-color distractor reproduced one old switch.
4. A connected merged mask could gradually enlarge the temporal reference.
5. LK during coast followed the occluder and reversed the target's motion.
6. Lab is only diagnostic in the new policy; the above failures are SAM/temporal
   gate failures, not selection by a Lab threshold. Real A–D causes remain unverified.

## Behavior and limitations
- Original selected SAM conditioning memory and prototype remain pinned.
- Foreground LK with forward/backward validation transports the last mask.
  Mask IoU, center displacement and size jointly gate tracking. The displacement
  log uses constant-velocity prediction; SAMURAI's internal Kalman still runs.
- Disconnected masks: keep the component overlapping the predicted target.
  Connected sudden growth: attempt signed-point SAM repair; otherwise coast.
- Repair uses cached current-frame encoder features, positive interior point and
  a negative point outside target support. It does NOT seed candidate identity or
  commit its mask. Future clean SAM observations must verify before memory writes.
- During coast only last trusted velocity is used: occluder optical flow must not
  move the target. Coast is limited to 0.35 s, invalid for control, with zero gain.
- A continuous target can remain `AMBIGUOUS` and valid while competitors are
  present. Search ambiguity has no valid target. Both freeze memory/gallery.
- Recovery requires original prototype/shape/area, SAM quality, motion bounds,
  original-memory mask agreement, competitor margin, three consistent observations
  and five probation observations. No color score can bypass these gates.
- Recovery candidates come from SAM encoder cosine peaks and point decoding.
  The original anchor is never evicted; only clean, verified, nonduplicate
  observations enter the bounded gallery. ORB has not been added.
- `1`–`9` explicitly select a displayed candidate and initialize a new user-chosen
  anchor. This is logged separately from automatic recovery. S/R, W/C, Q/Esc,
  video P and A/D retain their functions.
- SMALL/crop/tile paths remain. Recenter resets flow coordinates; global proposals
  still require the old anchor. Large relocations or heavy occlusion can require
  manual confirmation. Identical objects after complete occlusion cannot be
  guaranteed distinguishable from these observations.

## Run and record A–D
```powershell
cd D:\Sam
& .\.venv\Scripts\python.exe .\scripts\camera.py --source 0 --sam-only --target-mode object --recovery friend --record-raw .\runtime\A.mp4 --csv .\runtime\A_live.csv --save-video .\runtime\A_overlay.mp4
```
Use new output names B/C/D for subsequent sessions. Raw recording is unannotated
MP4, not lossless pixels. A bounded disk worker writes every captured frame and
`A.timestamps.csv`; disk overload fails explicitly rather than silently dropping
recorded frames. Existing raw files are not overwritten. Sidecar timestamps are
automatically used for replay. Keep the MP4 and sidecar together.

Camera capture does not wait for inference; display reuses the existing mask flow
transport. The source driver may still buffer internally. Physical camera and
driver latency were not measured in this task; camera UI/output wiring was tested
with fake capture, and real CUDA with generated video.

```powershell
& .\.venv\Scripts\python.exe .\scripts\camera.py --source .\runtime\A.mp4 --sam-only --target-mode object --recovery friend --csv .\runtime\A_replay.csv
```
Select with S/R, or copy the printed `--init-frame ... --init-bbox ...` selection
arguments for a headless repeat using `--no-display --mode offline`.

Logs: `*.sam.csv` is each completed SAM inference (raw area/components, temporal
IoU, motion distances, SAM quality, prototype/Lab, merge flag, state/reason);
normal CSV is each displayed frame and source skips. `*.selections.csv`,
`*.choices.csv`, `*.labels.csv` record prompts, numbered confirmation and W/C.
No online algorithm can label “second object exists” perfectly; diagnostics are
therefore recorded continuously. W/C alone labels events, not per-frame accuracy.

## Evaluation
Copy `scripts/object_eval_cases.example.json`, replace placeholder box/frame/video
values, add B/C/D, ordinary object and disappearance/return cases. Boxes in the
manifest are XYWH; GT boxes/distractors are XYXY in processing-image coordinates.
GT file example (replace these made-up coordinates):
```json
[
  {"frame":0,"visible":true,"box":[100,100,130,200],"distractors":[]},
  {"frame":15,"visible":true,"box":[105,100,135,200],"distractors":[[200,100,230,200]]},
  {"frame":16,"visible":false,"box":null,"distractors":[[200,100,230,200]]},
  {"frame":30,"visible":false,"box":null,"distractors":[[200,100,230,200]]}
]
```
Add samples around every ~15 frames and exact visibility boundaries. No visibility
is invented between contradictory annotations; boxes interpolate only within
known-visible intervals. Bbox IoU≥0.3 and greater overlap with target than any
distractor define correct lock. Unnecessary loss counts correct→unlocked events
while visible; switches count entry into a distractor lock. Tight crossings are
hard to judge with bbox GT; mask GT would give stronger evidence.
```powershell
& .\.venv\Scripts\python.exe .\scripts\eval_video.py --cases .\runtime\object_cases.json --variants 3 --recoveries current,friend --mode realtime --output .\runtime\object_eval
```
`current` is the retained legacy policy. The immediate-before comparison below
instead instantiates `NativeMemoryRecovery` directly, to isolate this patch.
The evaluator reports unknown metrics as null without GT. W/C event labels can
also populate `reacquisition_labels` in a case; they do not replace sparse GT.

## Measured checks
PyTorch 2.5.1+cu124; CUDA available; RTX 3050 6GB Laptop; SAMURAI mode true.
Short procedural A–D: 22 frames/case/policy, source time 30 FPS, unpaced inference.
These are small tests of the mechanism, not real-camera acceptance or 20 trials.

| Case | Unnecessary loss before→after | Switches before→after | Correct with distractor before→after | Locked frames before→after |
|---|---:|---:|---:|---:|
| A | 0→0 | 0→0 | 100→100% | 100→100% |
| B | 0→0 | 1→0 | 0→100% | 63.6→54.5% |
| C | 1→1 | 0→0 | 0→22.2% | 4.5→36.4% |
| D | 1→0 | 0→0 | 0→100% | 18.2→100% |

B returned during bounded coast: correct at the first reappearance sample
(0 source seconds); old policy did not recover in the fixture. This is not a
measured general recovery median/p90. C remains ambiguous after overlap and
requires confirmation; the new policy does not force a guess.

Full asynchronous 640×480, 30-FPS synthetic D, 175 displayed frames, overlay video
and CSV enabled: **29.67 Live FPS**, **10.40 SAM updates/s**, **19.38 SAM compute
FPS**, mean age **163.05 ms**. The profile's strict TRACKING share is **82.3%**
(AMBIGUOUS is excluded by that existing counter). One source frame was skipped.
The initial same-configuration run measured 25.02 Live FPS; display-clock drift
was corrected. Hardware execution timing varied considerably between short runs.
The strict ≥30 FPS requirement is **not met**. Verification decoding and dense
flow spikes remain costs; no masks, contours or output features were removed.

45 CPU/regression tests passed. Contracts cover continuity, signed repair, merged masks, frozen memory,
coast, ambiguity, sparse GT, timestamp roundtrip, and mocked camera UI/output.
Existing native-memory, SAM-only, SMALL, tile, playback and smooth-display tests
also passed. Signed positive/negative decoder was exercised on real CUDA SAM.
Natural A–D, person accuracy and ordinary-object video regressions remain pending
the supplied clips/GT; no user media or model implementation sources were read.

Evidence: `test_results/object_continuity/report.json`, `A_before.csv`…`D_after.csv`,
`realtime_final_profile.json`, `realtime_final.csv`, `realtime_final.sam.csv`.

## Files
- `scripts/recovery/continuity.py`: OBJECT continuity, repair, coast and recovery.
- `scripts/recovery/native_memory.py`: expose SAM hypotheses for independent ranking.
- `scripts/recovery/selection.py`: numbered candidates and human feedback logs.
- `scripts/recovery/metrics.py`: sparse GT metrics without fabricated visibility.
- `scripts/live_tracker.py`: signed point prompts on current cached features.
- `scripts/sam_memory.json`: explicit continuity/merge/coast parameters.
- `scripts/raw_recording.py`: bounded raw recording worker and timestamps.
- `scripts/camera.py`: factory, recording, inference diagnostics, live flow/output.
- `scripts/video_playback.py`: timestamp replay, CSV signals, manual selection.
- `scripts/realtime_video.py`: source-frame mapping, numbered selection, display clock.
- `scripts/mask_display.py`: numbered contours; no control from invalid coast.
- `scripts/eval_video.py`, `scripts/object_eval_cases.example.json`: headless A/B/GT setup.
- `scripts/check_object_continuity.py`, `scripts/check_object_continuity_gpu.py`: tests.
- `PROGRESS.md`, `OBJECT_CONTINUITY.md`: evidence, limitations and instructions.
