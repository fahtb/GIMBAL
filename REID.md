# Camera person ReID — 06/10/2026

**Đã có bản v2:** xem [REID_V2.md](REID_V2.md) cho anchor + rolling, partial crop,
hai mức bắt lại, detector thread, phím W/C và kết quả trước/sau. Nội dung bên
dưới là mô tả/đo đạc của bản v1, không phải hành vi mặc định hiện tại.

Đã thêm tự tìm lại **người** vào pipeline camera SAMURAI đang dùng. Bản sao trước
khi sửa nằm ở `backups/pre_reid_20261006/`.

## Chạy

```powershell
cd D:\Sam
& .\.venv\Scripts\python.exe .\scripts\camera.py --source 0 --image-size 512
```

Kéo chuột khoanh người trên khung LIVE bên trái; nên bao đủ phần người đang nhìn
thấy, không chỉ khuôn mặt. **S/R** chọn người mới, **Q/Esc** thoát, giữ nguyên luồng
cũ. Khi mất dấu, chương trình tự tìm; hết 30 giây chưa đủ bằng chứng thì hiện
`WAIT_SELECTION`, nhấn S/R để chọn lại. `--no-reid` trở về chế độ SAMURAI chọn lại
thủ công. `--reid-config đường_dẫn.json` dùng cấu hình khác.

## Kiến trúc thực tế

- SAM 2.1 Hiera Tiny, `samurai_mode: true`, bộ nhớ xét chuyển động và Kalman,
  CUDA Graph 512 tiếp tục tạo mask/box. Không sửa thuật toán SAMURAI trong lần này.
- `HSVRegionsEmbedder.encode(frame, box, mask=None)` là interface thay được bằng
  OSNet/torchreid; hiện chỉ dùng HSV, không cài mạng ReID khác. Crop được chuẩn hóa
  64×128, chia đầu/thân/chân theo tỷ lệ 20/40/40, histogram HS và V, căn bậc hai
  rồi chuẩn hóa L2. Chỉ lấy vùng giữa crop để giảm nền.
- Pipeline dùng **cùng cách crop theo box** ở cả SAM và YOLO. Box của SAM được
  suy ra từ mask. Không trộn descriptor mask-only với descriptor box của detector:
  thử nghiệm thực tế cho thấy sự khác biệt nền làm giảm điểm của đúng người.
- Gallery FIFO tối đa 20 vector, không lưu video/crop. Điểm là trung bình cosine
  của tối đa 3 mẫu giống nhất. Khởi tạo một mẫu khi chọn thành công; mỗi giây kiểm
  tra cập nhật: SAM IoU dự đoán ≥0.8, object logit ≥2, similarity ≥0.92, có detection
  người khớp box ≥0.5 và IoU với mọi người khác ≤0.1. Không cập nhật khi đang mất,
  tìm hoặc xác minh; nếu detector bỏ sót người khác, phép kiểm tra chồng lấn cũng
  có thể bỏ sót nguy cơ đó.
- YOLOv8n chính thức (`yolov8n.pt`), class 0/person, chạy CUDA FP16, toàn khung.
  Trong TRACKING chỉ chạy định kỳ khi kiểm tra gallery; trong SEARCHING tối đa
  một lượt mỗi 0.15 giây nguồn. Detector và SAM dùng cùng worker suy luận; đọc
  camera, giao diện và telemetry vẫn độc lập, chỉ lấy khung mới nhất.
- Máy trạng thái: `TRACKING → LOST → SEARCHING → REACQUIRE → TRACKING`.
  LOST giữ vị trí cuối và vận tốc làm mượt, vẽ dự đoán tối đa 0.7 giây. Đây chỉ là
  dự đoán, không phải vị trí đã xác nhận. Không dùng khoảng cách vị trí để nới
  điều kiện danh tính.
- Bắt lại cần top-1 >0.88, top-1 − top-2 >0.08 qua **3 lượt detection liên tiếp**,
  cùng ứng viên (IoU ≥0.3), cách nhau không quá 0.8 giây. Không có người thứ hai
  thì top-2=0. Không đếm lại cùng timestamp hay cộng streak từ các người khác nhau.
- Sau đó prompt box mới reset bộ nhớ SAMURAI và Kalman; thêm 3 khung kiểm tra SAM,
  appearance, detection và margin trước khi xuất điều khiển. Tracker hiện tại
  chính là LiveTracker tối ưu CUDA Graph; không có tracker optical flow riêng và
  không khởi động lại nhánh optical flow đã bị bỏ ở lần sửa trước.
- `valid=false`, dx/dy=null trong LOST/SEARCHING/REACQUIRE/WAIT_SELECTION hoặc
  kết quả cũ quá hạn. Box dự đoán/ứng viên có nhãn riêng, không phải box điều khiển.

## Cấu hình, overlay và log

Toàn bộ ngưỡng ReID nằm trong `configs/reid.json`; tên khóa tương ứng với các giá
trị mô tả ở trên. `detector_size`, `detector_confidence`, `max_candidates` điều
khiển chi phí/phạm vi detector. `timeout`, `prediction_seconds`, `gallery_interval`,
`search_interval`, `consecutive_frames`, `confirmation_frames` tính theo giây
nguồn hoặc số khung đã thực sự xử lý, không theo tốc độ vẽ UI.

Overlay có trạng thái, top-1/top-2 của lượt tìm/xác minh gần nhất, streak, gallery,
pipeline ms và các bộ đếm. `Reacq OK` là số lần qua kiểm tra tự động; `suspect` là
số lần ứng viên bị loại khi SAM/xác minh tiếp theo không đạt. **Không có ground
truth nên số bắt nhầm thật là `wrong: unlabeled` / `confirmed_wrong=null`**, không
đánh đồng nghi ngờ với sai danh tính đã được chứng minh. Các bộ đếm cộng dồn trong
phiên chạy; gallery và session ID đổi khi người dùng chọn mục tiêu mới.

CSV `runtime/reid_events.csv` ghi `selected`, `lost`, `reacquire_candidate`,
`reacquire_rejected`, `reacquired`, `timeout`, `manual_cancel`, `no_descriptor`.
Mỗi dòng có session, UTC, timestamp nguồn, timestamp mất/bắt lại, thời gian bắt
lại và top-1/top-2. Timestamp nguồn dùng đồng hồ đơn điệu của camera; UTC dùng để
đối chiếu thời điểm thực. Khi thử video mô phỏng, thời gian nguồn có thể khác thời
gian chạy kiểm tra. CSV nối thêm trên đĩa; có thể lưu trữ/xóa các log cũ khi đã dừng
ứng dụng. Nếu file bị khóa, chương trình cảnh báo; sự kiện không ghi được không
được đảm bảo khôi phục. Telemetry giữ đường dẫn `runtime/target.json`.

## Kiểm tra đã chạy trên máy này

- Python 3.11; torch 2.5.1+cu124, torchvision 0.20.1+cu124; CUDA khả dụng trên RTX
  3050 Laptop 6 GB. Cài thêm Ultralytics 8.4.173; không thay torch/torchvision.
  `pip check` đạt. Trọng số YOLOv8n đã tải vào `sam2/checkpoints/yolov8n.pt`.
- `scripts/check_reid.py`: 9 kiểm tra đạt (gallery, người giống màu, đổi ứng viên,
  timeout, timestamp trùng, reset thủ công, xác minh thất bại, đối thủ xuất hiện
  lúc xác minh, CSV thời gian bắt lại).
- `scripts/check_reid_gpu.py`: SAMURAI + YOLO thực trên GPU, 120 khung video người
  `05_default_juggle.mp4`, chèn 15 khung đen 25..39. Tự bắt lại 1 lần, không chọn
  thủ công, similarity cuối khoảng 0.953; thời gian từ mất đến bắt lại **0.967 giây
  theo timeline video**. Pipeline lúc TRACKING trung bình 55.2 ms; đỉnh CUDA
  allocated 312.3 MiB, reserved 534 MiB (không phải toàn bộ VRAM của tiến trình).
- `scripts/check_camera_pipeline.py --reid --source sam2/demo/data/gallery/05_default_juggle.mp4 --box 590 40 850 680`:
  chạy worker thật với video được phát theo nhịp; 139 kết quả trong khoảng 8 giây,
  toàn bộ TRACKING, SAM trung bình 55.5 ms, toàn pipeline 57.3 ms, latency 80.4 ms,
  gallery tăng từ 1 lên 8. Khung bị bỏ được phản ánh bằng sequence/timestamp.
- `check_telemetry.py` đạt cả trường hợp Windows khóa JSON. Kiểm tra cú pháp và
  `camera.py --help` đạt. DSHOW/MSMF camera 0 đều không mở được trong môi trường
  kiểm tra: **chưa xác minh camera/GUI trực tiếp**. Không khẳng định không lag.

Báo cáo nằm ở `test_results/reid_gpu_report.json`, `reid_gpu_events.csv`,
`camera_pipeline_report_512_reid.json`; ảnh sau bắt lại: `reid_recovered.jpg`.

HSV không đủ phân biệt chắc chắn người mặc giống nhau, đổi ánh sáng/góc nhìn
mạnh hoặc chỉ lộ một phần. Cơ chế margin ưu tiên từ chối trường hợp mơ hồ; vẫn có
thể bắt nhầm một người giống màu khi chỉ có người đó trong khung. Chưa có đánh giá
định lượng nhiều người thật/ID-switch; cần dữ liệu có nhãn để đo tỷ lệ bắt nhầm.
Thay backend embedder bằng OSNet là bước nâng cấp tiếp theo, chưa tích hợp ở đây.

API detector theo [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)
và [Predict](https://docs.ultralytics.com/modes/predict/). Phiên bản dependency
được pin trong `requirements-reid.txt`; tham số `quantize=16` được xác minh bằng
code Ultralytics đã cài và kiểm tra GPU.
