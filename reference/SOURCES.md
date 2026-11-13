# Read-only realtime references

- https://github.com/facebookresearch/sam2/issues/90 (read 2026-10-13): discussion of persisting output_dict, RGB ordering, bicubic/JPEG experiments and resetting frame_idx with predictor state. These are hypotheses/implementation lessons, not accuracy or speed guarantees for this laptop.
- https://github.com/Gy920/segment-anything-2-real-time and the supplied `D:\segment-anything-2-real-time-main.zip`: selectively inspected README, camera predictor and tiny512 architecture configuration. No package installation, no replacement of the project's sam2.
- Source repository LICENSE: Apache-2.0. LICENSE_cctorch: BSD-3-Clause; no cctorch implementation copied into this patch.
- The exported **candidate training config** derives from the local official SAM2 MOSE training config plus the reference tiny512 architecture. Its accompanying Apache license and STATUS.json record this provenance. No predictor implementation was copied. Memory quarantine and runtime patches are new code.

## Decision table

| Lesson | Current project / decision |
|---|---|
| Retain output_dict | Already present; add cloned compact snapshots and identity quarantine rather than importing an entire predictor |
| Bound memory | Retain recent24, bounded quality frames, original/limited permanent refresh anchors; measure soak |
| RGB | Already BGR->RGB; contract test verifies channels |
| Bicubic | Add --sam-resize bicubic for matched A/B; no default change without evidence |
| JPEG | Add --sam-jpeg-quality 90 experiment; default0 preserves original decoded camera input |
| Reset prompts/index | Existing manual reset retained; automatic verified recovery now rebases all state and indices together |
| 512 config | Existing runtime SAMURAI512 and original checkpoint retained; architecture from reference only used in training candidate |
| vos_inference compilation | Do not port foreign build/predictor API into installed SAMURAI; retain tested CUDA graphs |
| BF16 | Device supports it; --sam-precision bf16 uses eager mode because current graph wrappers are FP16 |
| Flash Attention | Current PyTorch wheel reports unavailable; no installation/forced unsupported backend |
| Probe budget | Existing independent model/thread/stream kept; wall/source-time checks and low-confidence behavior corrected |

Training-specific files under sam2/training and its MOSE Hydra config were read as expressly requested in Step5. Other sam2 source, .venv source, checkpoint contents and user videos were not inspected.
