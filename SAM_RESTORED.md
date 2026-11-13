# Bản SAMURAI mask đã khôi phục — 05/10/2026

**Cập nhật 06/10:** camera mặc định đã có tự tìm lại người bằng HSV + YOLOv8n,
giữ nguyên tracker SAMURAI mô tả bên dưới. Xem [REID.md](REID.md) cho hành vi hiện
tại. Chính sách mất dấu/chọn lại thủ công của bản này áp dụng với `--no-reid`.

Theo yêu cầu dùng lại bản ở hình 3: `camera.py` được khôi phục từ
`backups/pre_live_motion/camera.py`. Chương trình chạy SAM 2.1 Tiny + SAMURAI,
mask xanh, box xanh, tâm đỏ. Không khởi chạy worker optical flow; không còn
ngưỡng mất mục tiêu hoặc hết hạn xác nhận riêng của optical flow.

Đóng cửa sổ cũ rồi chạy:

```powershell
cd D:\Sam
& .\.venv\Scripts\python.exe .\scripts\camera.py --source 0 --image-size 512
```

Kéo chuột khoanh vật thể bên trái; **R/S** chọn lại, **Q/Esc** thoát.
Cửa sổ có tên **SAMURAI MASK**. Giữ nguyên độ phân giải model 512 của hình 3,
checkpoint, `samurai_mode: true`, cơ chế nhớ/Kalman và các ngưỡng tracking của
bản đó. Không có phím M hay tham số `--view` trong bản khôi phục.

## Thay đổi để giảm thời gian xử lý

- CUDA Graph phát lại encoder ở cùng kích thước đầu vào, giảm chi phí gọi GPU.
- Ở 512, CUDA Graph attention dùng cho đúng dạng tensor khi bộ nhớ đã đầy.
  Mỗi lần đều chép nội dung bộ nhớ/position mới. Hình dạng hoặc dtype khác dùng
  phép tính gốc, không padding/cắt bớt khung nhớ hay đổi lựa chọn SAMURAI.
- CUDA Graph cho decoder ở bước tracking; bước người dùng khoanh box vẫn
  chạy như gốc. Output được sao chép trước khi lưu lịch sử để các lần replay
  không ghi đè điểm số của khung trước.
- Lấy box bằng biên hàng/cột của cùng mask dương thay vì tạo mảng tọa độ pixel
  lớn trên GPU. Kết quả box được kiểm tra tương đương, kể cả mask rỗng.
- Cache embedding của bước không có prompt mới; lưu các điểm số lịch sử cần
  kiểm tra bằng Python ở CPU một lần, giữ nguyên giá trị. Tránh lặp phép tính
  cố định và đồng bộ GPU cho từng điểm số của từng khung cũ.
- Vẽ overlay ở kích thước cửa sổ và cache mỗi kết quả. Capture, inference và
  telemetry vẫn độc lập, chỉ giữ khung mới nhất, không queue video tăng dần.

Log khởi động báo CUDA graph enabled khi sử dụng được. Để so sánh bằng encoder/
attention gốc, thêm `--no-cuda-graph`. Không thay đổi PyTorch hoặc cài thư viện mới.
SAM vẫn cần thời gian suy luận; FPS trong cửa sổ là FPS mask thực. Không cam kết
30 FPS/zero-latency trên camera, và không dùng lại số đo 30 FPS của optical flow.

## Kiểm tra

`scripts/check_sam_restore.py` chạy bản gốc và bản tối ưu xen kẽ trên GPU,
90 khung tĩnh/video, so sánh từng pixel mask, box, trạng thái mất và số khung
nhớ. Kết quả/tốc độ nằm ở `test_results/sam_restore_report.json`.
Lượt A/B cuối: 90/90 khung cho mask giống từng pixel, box và trạng thái khớp;
mean suy luận gốc 66.35 ms, tối ưu 53.35 ms (1.24 lần), p95 tối ưu 58.13 ms.
Graph attention thực sự replay 53 lần, decoder 83 lần; không chỉ bật cờ mà
không sử dụng. Đây là số đo cùng phiên trên video/ảnh tĩnh, không phải cam kết
FPS camera thật. Các thử nghiệm cùng máy khác thời điểm có biến động tốc độ.
`scripts/check_camera_pipeline.py` kiểm tra camera giả lập từ video, chạy đúng
worker SAM mới. `scripts/check_telemetry.py` kiểm tra khóa file Windows.

Lượt pipeline cuối trên video người `05_default_juggle.mp4`, chọn cả người:
146 kết quả suy luận trong lượt 8 giây, mean 54.37 ms (~18.4 FPS suy luận),
mean độ trễ 77.97 ms. Peak CUDA allocated 300.08 MiB, reserved 524 MiB
(không gồm driver/process khác). Kết quả ở
`test_results/camera_pipeline_report_512.json`, ảnh cuối ở
`test_results/restored_pipeline_last.jpg`. Đã kiểm tra ảnh có mask bám người.
Chưa đo GUI/camera thật của người dùng; không dùng số 30 FPS của optical flow
làm số đo cho bản SAM mask này. Các lần thử trước khi cache prompt/score có
độ trễ cao hơn đáng kể, không còn là số liệu của bản cuối.

Telemetry vẫn ở `D:\Sam\runtime\target.json`, với dx/dy từ kết quả SAMURAI;
không phải tọa độ do optical flow ước lượng. LOST/STALE/STOPPED xuất valid=false.
Giữ xử lý ghi nền và thử lại khi file bị khóa.

Các file chính: `scripts/camera.py`, `scripts/live_tracker.py`,
`scripts/cuda_backbone.py`, `sam2/sam2/modeling/sam2_base.py`.
Bản trước lần khôi phục nằm ở `backups/pre_restore_sam_20261005`.
`LIVE_MOTION.md` và các test `check_live_*` là lịch sử thử nghiệm cũ,
không phải kiến trúc đang được `camera.py` sử dụng.
