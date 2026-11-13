> Update 2026-10-17: default now adds guarded small-target ROI around this same source model. See SMALL_TARGET_ROI.md. Use --small-target off to reproduce the full-frame baseline described below.

# Chạy tracker lấy từ project nguồn

Bản chuyển ngày 2026-10-14, nhánh `feature/source-camera-port`.
Commit dự phòng trước khi sửa: `48f3860`. Project nguồn `D:/segment-anything-2-real-time-main` chỉ đọc.

## Chạy

Không cần activate môi trường. Dependencies đã kiểm tra bằng pip: tất cả có sẵn, không nâng cấp package nào.

```powershell
cd D:\Sam
& D:\Sam\.venv\Scripts\python.exe -m pip install -r D:\Sam\requirements-source-camera.txt
& D:\Sam\.venv\Scripts\python.exe D:\Sam\scripts\camera.py --source 0 --sam-only --camera-config D:\Sam\configs\camera_msmf_measured.json --csv D:\Sam\runtime\source_live.csv --profile D:\Sam\runtime\source_live.profile.json
```

- Kéo chuột khoanh box trên ô camera bên trái; S/R mở chọn lại; W/C ghi nhãn sai/đúng; Q/Esc thoát.
- Video: thêm `--source D:\duong_dan\video.mp4`; P tạm dừng, A/D bước frame. Lùi video yêu cầu chọn lại vì trạng thái online chỉ đi tới.
- `--save-video D:\Sam\runtime\source_overlay.mp4` lưu video có mask; `--record-raw D:\Sam\runtime\raw.mp4` lưu camera gốc + timestamp.
- Mask, contour, bbox, tâm, dx/dy, telemetry JSON/CSV và optical flow hiển thị được giữ. Flow chỉ đưa mask đã hoàn tất lên frame hiển thị, không ghi ngược vào model.
- Đường dẫn tương đối của tài nguyên/đầu ra CLI được tính từ `D:/Sam`, không phụ thuộc cwd.
- Camera có thể nhận `--cam-size 640x480`, `--cam-auto-exposure`/`--no-cam-auto-exposure`, `--cam-exposure` theo driver. Không tự thay phơi sáng hiện tại.

Video mẫu nguồn (chỉ đọc), kiểm tra headless:

```powershell
& D:\Sam\.venv\Scripts\python.exe D:\Sam\scripts\camera.py --source D:\segment-anything-2-real-time-main\notebooks\videos\aquarium\aquarium.mp4 --mode offline --init-bbox 600,214,165,72 --init-frame 0 --max-frames 60 --no-display --csv D:\Sam\runtime\aquarium.csv --profile D:\Sam\runtime\aquarium.profile.json --save-video D:\Sam\runtime\aquarium_overlay.mp4
```

## Thuật toán thực sự chạy

`camera.create_tracker -> source_camera_tracker.SourceCameraTracker -> sam2.build_sam.build_sam2_camera_predictor -> SAM2CameraPredictor`.
Tất cả module `sam2` của luồng này nằm trong `vendor/source_camera/sam2`, được sao chép từ nguồn, không gọi `LiveTracker` cũ.

Nguồn triển khai: `demo/demo.py`, `sam2/sam2_camera_predictor.py`, `sam2/build_sam.py`, `sam2/modeling/*`, `sam2/utils/*`, `sam2/configs/*`.
`SOURCE_MANIFEST.json` ghi SHA-256 của 41 file; license Apache-2.0 và license extension đi kèm. Package nguồn được giữ nguyên, không sao chép venv/video/checkpoint.

- OpenCV BGR -> RGB giống demo nguồn, preprocessing resize/normalize do predictor nguồn thực hiện.
- Chọn: `load_first_frame(rgb)` rồi `add_new_prompt(frame_idx=0,obj_id=1,bbox=...)`; prompt điểm cũng được hỗ trợ qua CLI `--init-point x,y`.
- Frame tiếp: `track(rgb)`, persistent state, không tạo model lại. Mask là logits > 0 giống demo; bbox/tâm tính từ mask.
- Nguồn giữ frame đầu, conditioning anchor, tối đa `num_maskmem=7` non-conditioning outputs. Không lưu video vô hạn.
- Bỏ frame ở live: predictor tăng index theo mỗi frame thực sự xử lý đúng API nguồn. Không có Kalman/SAMURAI thời gian thực riêng; chuyển động quá nhanh giữa hai lần inference vẫn có thể làm mất dấu.
- Mask rỗng -> SEARCHING, tiếp tục `track()` với cùng memory, không khởi tạo lại. Mask trở lại -> TRACKING. `returns/reacquired` chỉ đếm mask xuất hiện lại, KHÔNG chứng minh đúng danh tính.
- Reset/reselect: xóa trạng thái và đặt `frame_idx=0` (nguồn `load_first_frame` không tự đặt lại biến này); giữ nguyên model đã nạp.

Các điều chỉnh tích hợp: adapter result dict cho UI, RGB theo demo, reset counter, chọn config Tiny có thật, tắt compile trong config để chạy eager trên Windows. Không chỉnh nội dung package nguồn.

## Model và thiết bị

- Config mặc định: `configs/sam2.1/sam2.1_hiera_t_512.yaml` từ nguồn.
- Checkpoint: `D:/Sam/sam2/checkpoints/sam2.1_hiera_tiny.pt`, SHA-256 trùng checkpoint Tiny ở nguồn; strict load thành công, không ghi đè/copy weights.
- `--image-size 1024` dùng config Tiny 1024 của nguồn. Config khác phải chỉ định `--model-config` và checkpoint phù hợp; image-size phải khớp config.
- Thực đo: torch 2.5.1+cu124, RTX3050 Laptop 6GB, CUDA, BF16. FP16 tùy chọn `--sam-precision fp16`.
- Không cài đè SAM2 editable hiện có. Adapter chọn vendored package trước import; nếu đã import package SAM2 khác thì báo lỗi và yêu cầu process mới.
- Nguồn hardcode CUDA trong predictor; nếu không có CUDA chương trình báo lỗi rõ, chưa có CPU fallback.
- `torch.compile`/VOS và CUDA graph cũ không bật. Extension `_C` của nguồn chưa build: nguồn tự bỏ postprocess lấp lỗ nhỏ. Segmentation/memory vẫn chạy.

## Phần thay thế / giữ lại

THAY: `LiveTracker`, `SAMMemoryTracker`, `NativeMemoryRecovery`, `ObjectContinuityRecovery`, `NativeCropTracker`, async probe, color/prototype recovery, class detector routing trong factory chính.
GIỮ: capture latest-frame, worker inference, camera/video transport, GUI/phím tắt, optical flow hiển thị, CSV/audit, save-video/raw, telemetry.

Các file thuật toán cũ vẫn giữ để bảo toàn lịch sử nhưng không được factory mới gọi. `--recovery current/friend`, `--target-mode auto/person/object`, memory/probe/crop/legacy gating options không còn thay đổi thuật toán. Cùng source predictor dùng cho người hoặc đồ vật. `--no-sam-only` bị từ chối. Không thêm YOLO/NanoSAM.
Nguồn không trả confidence định danh hoặc danh sách ứng viên cần chọn; HUD ghi rõ không có confidence này. W/C vẫn ghi nhãn; phím số vẫn còn handler nhưng không có candidate masks từ nguồn. Không giữ gallery/memory policy cũ dưới tên mới.

## Kết quả kiểm tra

- Syntax tất cả module Python được chuyển và adapter/UI: đạt. Import UI không nạp thuật toán cũ.
- `scripts/check_source_camera.py`: 55 frame sinh kiểm thử, box/point/reset/feedback, memory bound, source hashes và guard không import module cũ: đạt. Mean 65.06 ms, p95 91.40 ms; peak allocated GPU 357.05 MiB (bài nhỏ, không đại diện mọi phiên).
- `aquarium` nguồn: 60 frame offline, mask + CSV + MP4 hoạt động, TRACKING 100% (không phải độ chính xác GT); xử lý khoảng 16.45 SAM Hz.
- Video sinh 120 frame 30 FPS, async + save-video: Live 29.77 FPS, SAM 17.26 Hz, SAM mean 57.75 / p95 71.44 ms, age mean 103.06 ms, display p95 9.86 ms. Đây là bài headless, không phải FPS GUI camera.
- Camera MSMF 640x480: mở/track/reset/chọn lại/đóng/reopen đạt. Một bài ngắn camera ~15 FPS, SAM một mẫu 56.59 ms. Driver báo 30 nhưng FPS thực thay đổi 15–30 giữa các lần thử.
- GUI thật đã mở và hiển thị, chọn box hai lần và C có log; phiên thử kết thúc trước chuỗi W/R/Q nên assertion ban đầu không đạt. Không coi đây là kiểm tra đầy đủ phím bằng tay.
- `check_source_camera_gui.py`: camera thật, cửa sổ/input mô phỏng; main loop + 3 mouse selections + S/R/W/C/Q/cleanup đạt. Live 16.42 FPS trong bài này, camera khoảng 15–20; không đảm bảo 30 FPS camera. Không lưu hình camera trong bài này.
- `check_smooth_video.py`: 7 kiểm tra transport/output đạt. Sửa timestamp SKIPPED dùng nội suy cùng clock camera thay vì index/FPS; tránh audit thời gian bị phóng đại.
- CSV và báo cáo chi tiết: `test_results/source_camera/`. Dữ liệu runtime không commit.

Chưa kiểm chứng: độ đúng danh tính trên bộ clip cùng màu/che khuất có GT, soak dài, tăng tốc compile trên Windows, chọn bằng tay toàn bộ phím. Source SAM2 không đảm bảo phân biệt hai vật giống nhau sau che khuất; không báo confidence/accuracy giả.

## File thay đổi

- `vendor/source_camera/`: package nguồn, configs, licenses, README/setup tham khảo và manifest.
- `scripts/source_camera_tracker.py`: adapter dùng predictor nguồn thật, reset và output contract.
- `scripts/camera.py`: factory mới, CLI/path/default BF16, reset worker, HUD.
- `scripts/realtime_video.py`: HUD nguồn, timestamp frame bỏ qua.
- `scripts/video_playback.py`: tên cửa sổ nguồn; transport cũ giữ nguyên.
- `scripts/recovery/selection.py`: helper bbox độc lập, bỏ import memory cũ từ UI.
- `scripts/eval_video.py`: chỉ đánh giá source engine, không báo A/B legacy giả.
- `scripts/check_source_camera.py`, `scripts/check_source_camera_gui.py`: kiểm tra integration/GPU/camera và UI callbacks.
- `requirements-source-camera.txt`, tài liệu này và `PROGRESS.md`.
