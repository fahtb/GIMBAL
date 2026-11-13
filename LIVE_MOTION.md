# Bản camera cập nhật theo chuyển động trực tiếp

**Lưu trữ lịch sử:** từ 05/10/2026, camera đã khôi phục bản SAMURAI mask theo
yêu cầu người dùng. Hướng dẫn hiện tại ở [SAM_RESTORED.md](SAM_RESTORED.md).
Luồng optical flow và tham số `--view` bên dưới không còn chạy trong camera.py.

Chạy từ PowerShell, đóng cửa sổ cũ trước:

```powershell
cd D:\Sam
& .\.venv\Scripts\python.exe .\scripts\camera.py --source 0 --image-size 512 --view live
```

Kéo chuột khoanh vật thể bên trái. Đợi lần SAMURAI đầu tiên xác nhận; sau đó
khung bao vàng/tâm đỏ bên phải cập nhật theo ảnh camera mới bằng optical flow.
**R/S** chọn lại; **M** đổi giữa live box và mask SAMURAI có độ trễ; **Q/Esc** thoát.
Đây là thay đổi kiến trúc so với bản chỉ hiển thị khung SAM đã xử lý. Không vẽ
nguyên mask/box cũ lên ảnh mới để làm giả tốc độ tracking.

## Phân công thuật toán

- **SAM 2.1 Tiny + SAMURAI CUDA FP16** vẫn xử lý các khung mới nhất bằng state
  liên tục, bộ nhớ chuyển động giới hạn 24 khung và một khung điều kiện. Giữ
  `samurai_mode: true`, Kalman cập nhật theo khoảng thời gian nguồn. Không thay
  model bằng YOLO, NanoSAM hay tracker khác rồi gọi đó là SAMURAI.
- **Lucas–Kanade optical flow** chạy riêng trên ảnh xám rộng tối đa 640 pixel.
  Chỉ dùng corner nằm trong mask SAM đã erosion. Kiểm tra forward/backward,
  sai số quang học, số điểm, RANSAC similarity transform, scale và độ dịch chuyển.
  Khung bao được tính lại từ chuyển động quan sát trên khung mới; không phải
  giữ nguyên tọa độ hay dự đoán vận tốc mù.
- SAM đến trễ: khởi tạo lần đầu bằng cách theo flow qua ring tối đa 45 ảnh xám.
  Các lần sau tái sử dụng biến đổi đã kiểm tra, đối chiếu box SAM với lịch sử
  flow ở đúng thời điểm (IoU >=0.60), rồi đưa điểm/mask tham chiếu về hiện tại.
  Mask được biến đổi chỉ dùng chọn điểm nội bộ; không hiển thị thành mask SAM
  mới và không đưa ngược vào bộ nhớ SAMURAI.
- Không có xác nhận SAM quá 1.2 giây, ảnh bị ngắt >0.2 giây, thiếu điểm tin cậy,
  SAM báo mất hoặc hai luồng bất đồng: khóa LOST, bỏ sai lệch tâm và yêu cầu
  người dùng chọn lại. Không tự nhận lại một vật thể giống nhau.

`Live updates` là tốc độ worker chuyển động nhận/xử lý khung; `flow` là thời gian
xử lý flow; `Latency` tính từ lúc nguồn công bố ảnh tới kết quả chuyển động;
`image age` tính tới lúc GUI hiển thị. `SAM check age` là tuổi bằng chứng SAM,
khác độ trễ live box. `SAM inference` vẫn có thể hàng trăm ms; không được gọi
đó là segmentation 30 FPS. Khung bên phải luôn là đúng ảnh gắn với kết quả flow,
và ẩn box nếu ảnh cũ >200 ms hoặc bằng chứng SAM hết hạn.

## Đo và kiểm tra ngày 04/10/2026

Trên RTX 3050/Python 3.11 hiện tại, dùng ảnh blocks thực trong repo và dịch
chuyển đã biết, phát theo nhịp 30 FPS trong 7 giây, chạy SAMURAI thật trên CUDA:

| Phép đo | SAM chạy tự nhiên | SAM cố ý giữ mỗi lượt >=400 ms để mô phỏng log người dùng |
|---|---:|---:|
| Mẫu tracking hợp lệ | 178 | 168 |
| Median cập nhật live | 29.93 FPS | 29.97 FPS |
| Mean độ trễ live | 18.19 ms | 11.72 ms |
| P95 độ trễ live | 26.83 ms | 24.97 ms |
| P95 sai số tâm trên ảnh 1280x720 | 3.56 px | 3.78 px |
| Mất mục tiêu khi che toàn ảnh | đúng khung 180 | đúng khung 180 |

Hai lần chạy có tải khác nhau; không kết luận model chậm làm flow nhanh hơn.
Cột 400 ms có chèn thời gian đợi sau suy luận thật để kiểm tra tách luồng,
không mô phỏng mọi dạng tranh chấp CPU/GPU. Đây là độ trễ phần mềm sau khi ảnh
được công bố, **chưa bao gồm camera exposure/USB, GUI và màn hình thực**.
Không cam kết zero-latency hoặc 30 FPS trên mọi camera/cảnh.

Bài optical flow riêng dùng mask ground truth, delay anchor 400 ms: kiểm tra
dịch chuyển, vật thể tương tự, che khuất, khóa mất, chọn lại, gap, quá hạn SAM,
SAM lệch box và bộ nhớ hữu hạn đều PASS. Mask ground truth này chỉ ở test,
không dùng trong chương trình camera. Video blocks gốc có tay che khối Y:
flow khóa LOST sớm khi thiếu bằng chứng, không tự nhảy sang khối khác.

Chạy lại:

```powershell
& .\.venv\Scripts\python.exe .\scripts\check_live_motion.py
& .\.venv\Scripts\python.exe .\scripts\check_live_pipeline.py
& .\.venv\Scripts\python.exe .\scripts\check_live_sustained.py
& .\.venv\Scripts\python.exe .\scripts\check_live_sustained.py --sam-delay-ms 400
& .\.venv\Scripts\python.exe .\scripts\check_telemetry.py
```

JSON đo nằm trong `test_results/live_*report*.json`. Không đánh đồng các phép
thử này với đánh giá ID-switch trên dataset hoặc camera thật nhiều giờ.
Người/vật thể biến dạng nhiều, thiếu texture, mờ chuyển động, che khuất hay
vật thể giống hệt nhau vẫn có thể mất tracking; ngưỡng hiện tại ưu tiên dừng
khi thiếu bằng chứng hơn là đổi danh tính.

## Telemetry và file

Đích mặc định mới: **`D:\Sam\runtime\target.json`**, được tạo lúc chạy, tránh
tiếp tục ghi vào file `target.json` cũ đang báo WinError 5. Có thể đổi bằng
`--telemetry <path>`. Ghi nền, latest-only, tối đa khoảng 20 lần/giây. Khóa file
vẫn có thể ngăn ghi nhưng không làm dừng preview/tracking; kiểm tra khóa và
hồi phục Windows đã PASS. Khi lỗi kéo dài, consumer phải timeout dữ liệu cũ.

`estimated=true`: tọa độ là flow trên ảnh mới, được SAM xác nhận gần đây;
`sam_age_ms` là tuổi bằng chứng tại khung nguồn. `dx/dy` là pixel, dương phải/
xuống; có normalized values và `valid`. Consumer phải kiểm tra `valid`,
`timestamp_unix`, `source_age_ms`, `sam_age_ms`; không điều khiển từ dữ liệu cũ.
Không có motor/gimbal được điều khiển.

- `scripts/live_motion.py`: flow, lịch sử hữu hạn, xác nhận và worker riêng.
- `scripts/camera.py`: giao diện live/SAM, kết nối workers, telemetry mới.
- `scripts/check_live_motion.py`: kiểm tra định lượng với chuyển động đã biết.
- `scripts/check_live_pipeline.py`: video gốc + CUDA SAMURAI + flow.
- `scripts/check_live_sustained.py`: toàn pipeline trên nguồn ảnh dịch chuyển
  30 FPS, có chế độ mô phỏng model chậm.
- `backups/pre_live_motion/`: bản sao trước thay đổi camera/live tracker/docs.
