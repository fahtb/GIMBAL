# Chẩn đoán và kiểm thử PERSON/OBJECT — 06/10/2026

## Kết luận chẩn đoán trước khi đổi thuật toán

Video `videos/300651.mp4`: 988 frame, 2720×1530, 59,94005994 FPS. Bbox xe ở frame 0 dùng xuyên suốt: **x,y,w,h = 1910,865,570,305**, không resize.

Đã thêm log/khởi tạo headless rồi chạy **trước khi thay điều kiện tracking**. 30 frame đầu đều TRACKING; toàn bộ 988 frame cũng TRACKING 100%, mất 0, bắt lại 0 (15,24 FPS xử lý ở lượt đầu). Lưu ở `test_results/drone_baseline_30.csv`, `test_results/drone_baseline_full.csv`. Bản code nguyên thủy được lưu ở `backups/pre_object_20261006`; bản chỉ thêm chẩn đoán ở `backups/object_diagnostic_baseline`.

Không tái hiện được lỗi LOST ở frame thứ 2 bằng bbox này. Không có bbox/frame chọn/resize của phiên lỗi cũ nên **không thể khẳng định chính xác điều kiện gây LOST ở phiên đó**. Các số đo tại frame thứ 2 (index 1):

| Điều kiện cũ | Giá trị | Kết quả |
|---|---:|---|
| Điểm IoU mask ≥ 0,5 | 0,96728515625 | Qua |
| Object logit > 0 | 8,78125 | Qua |
| Diện tích mask ≥ 16 px | 97.601 px | Qua |
| Tỷ lệ diện tích bbox trong [0,25; 4] | 1,00544197 | Qua |
| Dịch chuyển bbox ≤ 240,2893 px | 2,1213 px | Qua |

Frame index 2 cũng TRACKING: IoU 0,9682617, logit 8,484375, diện tích mask 97.032 px. Khoảng thời gian là 1/FPS nguồn, không bị thời gian suy luận offline làm tăng.

Nghi vấn có hai phần cần phân biệt:

1. `ReIDTracker` cũ ở TRACKING gọi SAM; không loại xe ngay chỉ vì không phải người. SAM có thể mất vì mask rỗng, điểm IoU thấp, logit ≤ 0, thay đổi bbox quá lớn hoặc thời gian không hợp lệ. CSV mới chỉ rõ từng lý do, kể cả các điều kiện đồng thời sai.
2. Đường **tìm lại** cũ thực sự chỉ dành cho người: YOLO `classes=[0]`; `VisibleHSV` chia thân người; gallery yêu cầu overlap người/không overlap người khác; xác minh bắt lại yêu cầu detection person. Vì vậy sau một lỗi SAM, xe không có đường tìm lại thích hợp. Khi SAM đã dừng và vòng lặp chỉ tìm/chờ, FPS vòng lặp cao không đại diện cho FPS SAM.

## Kiến trúc sau sửa

`camera.py/create_tracker` tạo `TargetTracker`. Mỗi lần chọn lại, detector person cũ quyết định PERSON theo IoU với ROI; nếu không trùng thì OBJECT. Có `--target-mode person` để tái hiện cổng cũ, `--target-mode object` để ép vật thể. `--no-reid` giữ đường SAMURAI nguyên bản.

- PERSON: gọi nguyên `ReIDTracker`, không đổi ngưỡng/điều kiện/memory. Chỉ thêm diagnostics và API detection all-class riêng. Camera vẫn dùng `SearchDetector` cũ cho PERSON.
- OBJECT: `ObjectTracker` gọi SAM mỗi frame TRACKING; bỏ cổng theo người và latch mất một frame của SAM. Histogram toàn ROI so với mẫu chọn ban đầu, score thấp K frame và hysteresis, diện tích sụp là các cổng riêng trong `configs/object.json`.
- `CropSAM` giữ hệ tọa độ crop ổn định cho vật nhỏ; dịch/đổi kích thước crop sẽ re-prompt với dự đoán chuyển động, ghi lý do. Mask/bbox được trả về hệ tọa độ ảnh xử lý đầy đủ.
- `ImageMotion` dùng LK nền ngoài bbox + affine RANSAC và chuyển động riêng của mục tiêu. SAM sửa lệch flow; flow không quyết định danh tính.
- `ObjectSearch`: thread riêng, queue latest-only có giới hạn. Biết lớp COCO thì chỉ tìm lớp đó; chưa biết lớp thì DINOv2-small tìm patch toàn khung, chấm embedding các ROI, loại trùng proposal, kiểm tra similarity/margin/N frame rồi SAM kiểm chứng lại. Offline bơm sự kiện UI trong khi giữ đúng frame chờ worker; realtime bỏ kết quả quá cũ.

DINO dùng implementation/checkpoint [chính thức Meta](https://github.com/facebookresearch/dinov2), cache tại `runtime/torch_hub`. Không âm thầm thay DINO bằng histogram nếu thiếu model; lỗi chỉ cách chạy `prepare_object_models.py`. Nhánh COCO dùng histogram toàn đối tượng để phân biệt các detection cùng lớp, không dùng histogram thân người.

## Bảng trước/sau chạy tuần tự

Lệnh: `.\.venv\Scripts\python.exe scripts\benchmark_object.py`. Mỗi tiến trình chạy xong mới bắt đầu tiến trình tiếp theo; cùng input, bbox, frame khởi tạo, kích thước encoder. “Trước” là cổng PERSON cũ được giữ nguyên qua `--target-mode person`; “Sau” là `auto`. Toàn bộ command, CSV và số liệu máy đọc được nằm trong `test_results/object_benchmark/summary.json`.

| Clip | Nhánh | Frame | TRACKING | Mất | Bắt lại | FPS xử lý |
|---|---|---:|---:|---:|---:|---:|
| 300651.mp4 | Cổng PERSON cũ | 988 | 100% | 0 | 0 | 13,09 |
| 300651.mp4 | Auto → OBJECT | 988 | 100% | 0 | 0 | 9,01 |
| Người + che khuất | Cổng PERSON cũ | 150 | 78% | 1 | 1 | 18,37 |
| Người + che khuất | Auto → PERSON | 150 | 78% | 1 | 1 | 18,61 |

Clip người là fixture 150 frame, frame 40–69 màu đen. Cả hai nhánh mất tại frame 40, SEARCHING tại 41, REACQUIRE tại 70, TRACKING tại 73. Script assert trạng thái, bbox, IoU, diện tích mask và lý do chuyển trạng thái **trùng từng frame**. Clip người thật 120 frame không chèn che khuất cũng đã chạy trước/sau với output trùng từng frame, TRACKING 100%.

Mục tiêu ≥80% TRACKING trên video drone đạt với bbox đã ghi. Không loại frame nào khỏi mẫu số 988. Đây là % trạng thái, không phải điểm accuracy so với ground truth bbox. Đã kiểm tra trực quan các frame rải đều trong `test_results/drone_after_contact.jpg`; bbox vẫn ôm xe. OBJECT chậm hơn do thêm LK/affine/crop/histogram; không tuyên bố tăng FPS.

## Kiểm tra bắt lại và vật nhỏ

- 6 test transport/video (headless, init-frame/bbox, offline/realtime, seek, loop, output alignment).
- 6 test OBJECT (hysteresis score/appearance, collapse, crop roundtrip/recenter, queue không chặn và có giới hạn, distinct-frame streak, unknown-class routing).
- 14 test ReID hiện có vẫn qua.
- `check_object_gpu.py`: SAM CUDA + COCO search thật trên ảnh xe với frame 5–8 đen. Mất frame 5 vì diện tích mask sụp, bắt lại frame 11; 30 frame, 80% TRACKING, 1 mất/1 bắt lại.
- `check_object_gpu.py --unknown`: giấu lớp COCO để buộc DINO thật; đi qua mất → tìm → xác minh SAM → bắt lại, cùng frame 5/11 và 80% TRACKING.
- `check_object_gpu.py --dino-only`: tìm toàn khung, top1 0,9317, top2 0,7806, margin 0,1511, vùng đề xuất IoU 0,4449 với bbox chọn; vùng này sau đó cần SAM phân đoạn, không dùng trực tiếp làm bbox điều khiển.
- Xe nhỏ: `--resize-width 320 --init-bbox 224.7,101.76,67.06,35.88`, 30 frame; OBJECT, crop ngay lúc chọn, TRACKING 100%, mất 0, bắt lại 0, 13,76 FPS. CSV `test_results/drone_small_30.csv`.
- `check_search_worker.py`: capture/inference/person-search threads thật trên video có nhịp live; các trạng thái TRACKING/LOST/SEARCHING/REACQUIRE, phím C xác nhận và shutdown đều PASS. Đây là mô phỏng nguồn live bằng file, không phải thử webcam thật.

## Chạy lại trên xe

```powershell
.\.venv\Scripts\python.exe scripts\camera.py --source videos\300651.mp4 --mode offline --init-frame 0 --init-bbox 1910,865,570,305 --no-display --max-frames 988 --csv runtime\300651_object.csv --save-video runtime\300651_object.mp4
```

Bỏ `--no-display` để xem. Hoặc bỏ cả `--init-frame/--init-bbox` để chọn thủ công bằng S/R, wheel zoom, kéo khung, Enter. P/A/D/Q/Esc giữ chức năng. Console và `<tên>.selections.csv` ghi tọa độ để tái sử dụng. Không dùng bbox trên với `--resize-width` mà chưa quy đổi tọa độ.

Tái hiện 30 frame cổng cũ:

```powershell
.\.venv\Scripts\python.exe scripts\camera.py --source videos\300651.mp4 --mode offline --target-mode person --init-frame 0 --init-bbox 1910,865,570,305 --no-display --max-frames 30 --csv runtime\300651_diagnostic.csv
```

## File thay đổi

- `scripts/camera.py`: CLI, factory/routing, giữ camera workers và phím cũ, selection logging.
- `scripts/video_playback.py`: headless/init, CSV diagnostics, chờ tìm kiếm offline có xử lý phím.
- `scripts/live_tracker.py`: lý do từng cổng SAM, policy cho phép OBJECT dùng hysteresis bên ngoài.
- `scripts/person_reid.py`: diagnostics và `detect_all` riêng; không đổi cổng PERSON.
- `scripts/target_tracker.py`, `scripts/object_tracking.py`, `scripts/object_search.py`: routing, tracking OBJECT, thread tìm lại.
- `configs/object.json`: ngưỡng và cấu hình OBJECT.
- `scripts/prepare_object_models.py`: tải DINO chính thức.
- `scripts/check_video_playback.py`, `scripts/check_object_tracking.py`, `scripts/check_object_gpu.py`, `scripts/benchmark_object.py`: kiểm thử và benchmark tái lập.
- `VIDEO_PLAYBACK.md`, `OBJECT_TRACKING_REPORT.md`: hướng dẫn và báo cáo.

Giới hạn: GUI chuột/phím thật chưa được thao tác bằng tay trong phiên này; test transport và các lượt GPU chạy headless. Tự phân loại phụ thuộc detector ở frame chọn; nếu person detector bỏ sót người thì `auto` chọn OBJECT, có thể ép `--target-mode person`. Hysteresis/embedding không bảo đảm danh tính khi nhiều vật giống nhau; các ngưỡng được ghi trong CSV để đo và điều chỉnh trên dữ liệu mới.
