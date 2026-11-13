# Chạy video với pipeline camera

## Khởi tạo lặp lại và chạy không mở cửa sổ

Ví dụ chiếc xe ở frame 0 của `300651.mp4`, ảnh gốc 2720×1530:

```powershell
.\.venv\Scripts\python.exe scripts\camera.py --source videos\300651.mp4 --mode offline --init-frame 0 --init-bbox 1910,865,570,305 --no-display --max-frames 988 --csv runtime\300651_object.csv --save-video runtime\300651_object.mp4
```

`--init-bbox` nhận **x,y,w,h**, không phải x1,y1,x2,y2. Tọa độ theo ảnh xử lý (sau `--resize-width` nếu có). `--init-frame` đánh số từ 0, không nhỏ hơn `--start-frame`; khi có bbox, bắt đầu chạy ngay tại frame đó. Bỏ `--no-display` để xem và dùng các phím cũ. Chạy 30 frame chẩn đoán bằng `--max-frames 30`.

Chọn thủ công luôn in dòng `SELECTION ... --init-frame ... --init-bbox ...`, ghi bbox vào cột `selected_bbox` của frame CSV và tệp `<tên>.selections.csv`. Tệp selections được append để giữ các lần chọn trước; frame CSV/video được ghi đè.

CSV bổ sung `transition`, `transition_reason`, điểm mask `iou`, `object_logit`, `mask_area`, `sam_reason`, diện tích bbox tương đối, dịch chuyển/ngưỡng, `yolo_person` (`not_run` nếu không gọi YOLO trên frame đó), `thresholds`, `target_mode`, `coco_class`, `crop_box`, lý do sửa `correction_reason`, lý do motion và tìm kiếm. Các điểm SAM trống nghĩa là frame đó không chạy SAM; không diễn giải FPS cao ở trạng thái chờ/tìm kiếm thành FPS SAM.

## PERSON và OBJECT

Mặc định `--target-mode auto`. Khi chọn bbox, YOLO person với thiết lập cũ kiểm tra overlap theo `person_match_iou`; trùng thì PERSON, không trùng thì OBJECT. Có thể ép `--target-mode person` để tái hiện cổng kiểm chứng cũ, hoặc `--target-mode object` để thử nhánh vật thể. `--no-reid` vẫn dùng SAMURAI nguyên bản, tắt routing và bắt lại tự động.

- PERSON dùng nguyên `ReIDTracker`, histogram vùng thân người, gallery và các ngưỡng cũ. Việc thêm chẩn đoán không đổi điều kiện chấp nhận.
- OBJECT dùng histogram toàn vùng chọn làm mẫu bất biến, không có cổng YOLO person/HSV thân người. SAM vẫn tạo mask mỗi frame TRACKING. Mất khi score thấp đủ K frame, histogram thấp đủ K frame hoặc diện tích mask sụp. Vùng hysteresis giữ bộ đếm; vượt ngưỡng cao mới reset. Không coi chỉ một object logit thấp hoặc một bước dịch chuyển lớn là mất.
- `configs/object.json` chứa ngưỡng: score thấp/cao 0,35/0,55; K=3; histogram 0,30/0,45; diện tích sụp dưới 12% diện tích ổn định hoặc dưới 16 px. Có thể thay bằng `--object-config`.
- Mục tiêu có cạnh dưới 48 px **hoặc** dưới 48 px khi quy về encoder dùng crop khoảng 4× bbox (tối thiểu 128 px). Crop có hệ tọa độ ổn định; dịch/tăng giảm crop sẽ khởi tạo lại SAM bằng vị trí dự đoán và ghi rõ `small_target_crop_enter`, `crop_recenter_motion_prediction`, `crop_rescale_size_change`. Mask/bbox được quy về ảnh đầy đủ.
- LK trên nền ngoài bbox ước lượng affine camera; LK cục bộ bổ sung chuyển động riêng của mục tiêu. SAM sửa sai lệch flow; mỗi lần có ghi `flow_drift_corrected_by_sam`. Flow không tự xác nhận danh tính.
- Khi mất, worker riêng tìm YOLO đúng lớp COCO đã nhận lúc chọn. Nếu không nhận được lớp COCO, worker dùng DINOv2-small: đặc trưng patch toàn ảnh → đề xuất nhiều vùng/kích thước → cosine embedding vùng → similarity, margin và N frame nguồn khác nhau liên tiếp → SAM kiểm chứng và khởi tạo lại. Không đổi lớp tự động giữa phiên.
- Offline giữ frame đang tìm kiếm cho đến khi worker trả kết quả, vẫn bơm sự kiện bàn phím. Realtime/camera nhận kết quả mới nhất và từ chối kết quả quá cũ. Hàng chờ có giới hạn, không tích lũy toàn video. DINO dùng CUDA mặc định; có thể đặt `dino_device: cpu` nhưng sẽ chậm hơn.

DINO được tải từ [repo chính thức Meta](https://github.com/facebookresearch/dinov2) bằng `scripts/prepare_object_models.py`, lưu code/weights ở `runtime/torch_hub`. Trên máy này đã tải và kiểm thử. Nếu chuyển máy, chạy:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_object_models.py
```

Kiểm thử và benchmark:

```powershell
.\.venv\Scripts\python.exe scripts\check_object_tracking.py
.\.venv\Scripts\python.exe scripts\check_video_playback.py
.\.venv\Scripts\python.exe scripts\check_reid.py
.\.venv\Scripts\python.exe scripts\check_object_gpu.py
.\.venv\Scripts\python.exe scripts\check_object_gpu.py --unknown
.\.venv\Scripts\python.exe scripts\benchmark_object.py
```

Benchmark chạy tuần tự, cùng frame/bbox/input, xuất `test_results/object_benchmark/summary.json`. Clip người che khuất dùng fixture `test_results/reid_threaded_exit.avi` (150 frame: ảnh người cố định, frame 40–69 đen). Các clip che khuất là kiểm thử có kiểm soát, không phải số đo accuracy trên dữ liệu thực có ground truth.

## Điều khiển và xuất file

Từ PowerShell tại `D:\Sam`:

```powershell
.\.venv\Scripts\python.exe scripts\camera.py --source "videos\thay_gui.mp4" --mode offline --save-video "runtime\ketqua.mp4"
.\.venv\Scripts\python.exe scripts\camera.py --source "videos\thay_gui.mp4" --mode realtime --start-frame 300 --max-frames 600 --resize-width 1280 --save-video "runtime\realtime.mp4"
.\.venv\Scripts\python.exe scripts\camera.py --source 0
```

- Video bắt đầu pause ở `--start-frame` (đánh số từ 0). Nhấn **S/R**, lăn chuột để zoom quanh con trỏ (tối đa 16x), kéo khung và **Enter/Space** xác nhận; **Esc** hủy chọn. Nhấn **P** để chạy/pause.
- Khi pause: **A/D** lùi/tiến đúng một frame. Lùi frame hoặc `--loop` về đầu sẽ bỏ tracking cũ và yêu cầu chọn lại, vì bộ nhớ SAMURAI chỉ đi theo thời gian tăng dần. Loop về `--start-frame` và pause cho phép chọn lại.
- **W/C** vẫn đánh dấu lần bắt lại sai/đúng; **Q/Esc** hoặc đóng cửa sổ để thoát. Các phím có hiệu lực sau khi bước suy luận hiện tại hoàn tất.
- `--mode realtime` (mặc định): phát theo FPS nguồn và bỏ những frame đã trễ. `offline`: xử lý tuần tự mọi frame, không chờ theo FPS và không bỏ frame.
- `--max-frames` giới hạn số lượt xử lý, không tính frame `SKIPPED`, tính chung qua các lần loop/tua/chọn lại. `--resize-width` mặc định không resize; khi dùng sẽ giữ tỷ lệ ảnh. Tọa độ output thuộc ảnh sau resize.
- Các tùy chọn playback/xuất file dành cho nguồn video; nguồn camera vẫn chạy giao diện và capture worker cũ.

Camera và video dùng chung `create_tracker` và `TargetTracker`; PERSON dùng `ReIDTracker`, OBJECT dùng `ObjectTracker`. Camera PERSON vẫn dùng detector worker latest-frame và các cổng cũ.

Thời gian đưa vào tracker là `frame / FPS nguồn`; pause và tốc độ xử lý offline không làm tăng khoảng cách thời gian của tracker. Video không có FPS hợp lệ dùng 30 FPS và in thông báo. Video FPS biến thiên được phát theo FPS danh nghĩa do OpenCV cung cấp.

CSV luôn được ghi: dùng `--csv path.csv` để chỉ định; mặc định cạnh `--save-video`, hoặc `runtime/<tên-video>_frames.csv`. Cột gồm `pass`, `frame`, `time_s`, `status`, bbox `x1/y1/x2/y2`, `center_x/y`, `dx/dy`, `similarity` (ReID top1), `latency_ms`, `age_ms`, `sam_fps`. Giá trị không có để trống; SAM FPS bằng 0 khi frame không chạy SAM. `latency_ms` đo bước tracking; `age_ms` tính cả độ trễ so với lịch phát trong realtime. Pause không làm già frame. Similarity là điểm ReID hiện có, không phải ground truth.

Mỗi frame xử lý/bỏ qua có một dòng CSV và một frame video overlay tương ứng. Frame bỏ qua ghi `SKIPPED` và không có mask/bbox. Pause không ghi lặp; chọn lại trên cùng frame tạo một kết quả mới, tua/loop có thể lặp số frame nguồn. `pass` tăng khi tua lùi hoặc loop. Video xuất không có âm thanh, dùng FPS nguồn; `.avi` dùng MJPG, định dạng khác dùng mp4v. Tệp output hiện có được ghi đè.

Khi EOF/thoát/đạt giới hạn, console in số frame xử lý, số bỏ qua, % TRACKING trên frame xử lý, số lần mất/bắt lại tự động, FPS xử lý trung bình (không tính thời gian pause/chọn ROI, chờ lịch phát hay ghi output). Thống kê loop cộng dồn. Chọn lại thủ công không được tính là bắt lại tự động.

Kiểm tra transport/export không cần GPU:

```powershell
.\.venv\Scripts\python.exe scripts\check_video_playback.py
```
