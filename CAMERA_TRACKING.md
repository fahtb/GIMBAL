# Tracking camera với SAMURAI

**05/10/2026: đã khôi phục SAMURAI mask theo hình 3, bỏ optical flow khỏi camera.**
Xem [SAM_RESTORED.md](SAM_RESTORED.md) cho lệnh chạy, CUDA Graph và kiểm tra mới.
Các mô tả/số đo bên dưới là lịch sử phát triển, không phải cấu hình hiện hành.

## Bản sửa giảm trễ (04/10/2026)

Camera hiện mặc định `--image-size 512`: giảm kích thước encoder thực sự từ
1024 xuống 512, vẫn cùng checkpoint SAM 2.1 Tiny, SAMURAI và 24 khung nhớ.
Để giữ chi tiết bản gốc, dùng `--image-size 1024`; mức giữa là 768. Giảm độ
phân giải có thể làm mất vật thể nhỏ/biên mask mảnh. Bản mới bổ sung optical
flow riêng theo LIVE_MOTION.md; vẫn không có YOLO thay thế SAMURAI.

Lần so sánh cùng video/cùng phiên: 1024 mean 916 ms, 768 mean 918 ms,
512 mean 412 ms (nhanh hơn ~2.2 lần so với 1024 trong phép đo này).
Các số đo khác phiên trước vì tải/điều kiện chạy thay đổi; không dùng chúng để
cam kết FPS camera. Cả ba giữ 39 khung đầu rồi LOST. Mean mask IoU với 1024,
chỉ trên khung cả hai còn tracking: 768=0.979, 512=0.960. Đây là mức tương đồng
với output model gốc, **không phải độ chính xác so với ground truth**.
Chi tiết: `test_results/resolution_benchmark.json`; tái đo bằng
`& .\.venv\Scripts\python.exe .\scripts\benchmark_sizes.py`.

Kiểm tra cuối ở 512: toàn bộ kiểm tra mất mục tiêu, giới hạn nhớ, chọn lại và
bỏ khung PASS (`test_results/streaming_report_512.json`). Mean suy luận tuần tự
374 ms; GPU allocated ổn định 193.77 MiB, peak 322.96 MiB. Pipeline video có
nhịp nguồn: 7 kết quả chạy model, mean inference 281.77 ms (~3.55 FPS), latency
309.27 ms (`test_results/camera_pipeline_report_512.json`). Vẫn chưa đạt 30 FPS;
tuổi ảnh tracking còn gồm thời gian chờ kết quả kế tiếp. Camera thật/GUI chưa
được đo lại trong phiên sửa lỗi này.

Giao diện cache ảnh mask theo kết quả suy luận để không tô lại toàn bộ ảnh ở
mỗi lượt hiển thị. Ghi JSON chuyển sang worker riêng, chỉ giữ bản mới nhất,
tự thử lại nếu Windows đang khóa file. Đã tái hiện WinError 5 và kiểm tra hồi
phục bằng `scripts/check_telemetry.py`: ứng dụng không thoát vì khóa JSON.
Nếu khóa kéo dài, consumer phải timeout theo timestamp (file có thể chưa cập nhật).
Trên wheel Windows không có Flash Attention, cho SDPA chọn kernel sẵn có ngay;
không cài Flash Attention hoặc giả định cảnh báo biến mất nghĩa là tăng tốc.

Bản trước sửa: `backups/pre_fps_fix`. Các số đo 1024 trong phần kiểm tra phía
dưới là lịch sử của phiên trước, không phải cam kết tốc độ mặc định 512 hiện tại.

Chạy trong PowerShell từ `D:\Sam`:

```powershell
& .\.venv\Scripts\python.exe .\scripts\camera.py --source 0
```

Chờ model tải xong. Kéo chuột trái khoanh mục tiêu trên cửa sổ LIVE bên trái;
khung được chọn là khung mới nhất lúc nhả chuột. Nhấn **R** (hoặc S) rồi kéo
lại để đổi/nhận lại mục tiêu. **Q**, **Esc** hoặc đóng cửa sổ để thoát.
Nếu camera khác, thay `--source 0` bằng `--source 1`. Có thể truyền đường dẫn
video vào `--source` để phát mô phỏng tốc độ nguồn.

Ở chế độ `--view sam`, ảnh bên trái là preview mới nhất; bên phải là đúng khung đã suy luận với mask,
box, tâm mục tiêu, trạng thái, FPS suy luận, thời gian suy luận, độ trễ từ lúc
đọc khung tới kết quả và tuổi của kết quả. FPS này không phải FPS camera hoặc
FPS toàn pipeline. Khi kết quả quá một giây sẽ ẩn mask và báo STALE.

## Thuật toán và giới hạn

- Dùng checkpoint **sam2.1_hiera_tiny.pt**, config gốc
  `configs/samurai/sam2.1_hiera_t.yaml`, `samurai_mode: true`, CUDA FP16.
  Camera override `model.image_size=512`, có thể đổi bằng `--image-size`.
- `LiveTracker` gọi backbone và `SAM2Base.track_step` từng khung mới trên cùng
  state. Box ban đầu là hai điểm SAM nhãn 2/3; các bước sau không prompt lại.
- Vẫn dùng chọn mask theo SAMURAI/Kalman và bộ nhớ chọn theo IoU, object score,
  motion score. Bộ nhớ giữ một khung điều kiện và tối đa 24 khung gần nhất.
  Đây là cửa sổ hữu hạn: không giữ vô hạn các khung tốt trong quá khứ như repo
  offline có thể làm. Chất lượng sau che khuất dài có thể giảm.
- Thay vòng quét tất cả index bằng quét các key còn giữ; không truy cập khung đã
  xóa. Chỉ số model tăng theo khung **đã xử lý**. Kalman tích phân theo timestamp
  thực tế, đơn vị 1/30 giây; covariance tăng theo dt. Positional encoding của SAM
  vẫn theo bước đã xử lý vì model không được huấn luyện với timestamp liên tục.
  Khoảng cách > `--max-gap 1.0` giây khóa LOST thay vì tiếp tục dự đoán mù.
- Capture thread chỉ giữ khung mới nhất; inference thread giữ một kết quả;
  main thread hiển thị. Không có queue video tăng dần. Các dict nhớ đều hữu hạn.
  PyTorch dùng 4 luồng CPU, OpenCV dùng 1 để giảm tranh chấp CPU.
- Gate bảo thủ: mask >=16 pixel, predicted IoU >=0.5, object logit >0, giới hạn
  thay đổi diện tích và dịch chuyển theo thời gian. Kết quả không đạt bị loại,
  không đưa vào bộ nhớ, khóa LOST. **Không tự tìm lại**: người dùng khoanh lại
  để xác nhận danh tính. Dự đoán IoU không phải xác suất đúng danh tính; vật thể
  giống hệt nhau/cắt ngang vẫn có thể gây drift trước khi gate phát hiện.
- Không dùng YOLO, NanoSAM hay mô hình ReID. Có optical flow cập nhật live box
  giữa các lần SAM xác nhận, chi tiết ở LIVE_MOTION.md. Không tuyên bố đảm
  bảo danh tính trong mọi cảnh; cần bộ video có nhãn ID để đo ID-switch.

## Xuất sai lệch tâm

`runtime/target.json` được thay thế bằng kết quả live motion mới nhất, không tạo log vô hạn.
`dx`, `dy` tính bằng pixel trên ảnh nguồn: dương sang phải và xuống dưới.
`dx_normalized`, `dy_normalized` chia theo nửa chiều rộng/cao.
Khi LOST/STALE/STOPPED: `valid=false`, sai lệch bằng null.
Consumer tương lai phải kiểm tra `valid`, `source_age_ms` và thời điểm file/
`timestamp_unix`, có timeout riêng nếu chương trình bị kill. Timestamp `captured`
là đồng hồ monotonic cục bộ sau khi OpenCV đọc ảnh, không phải timestamp phần
cứng/exposure. Chưa có giao tiếp hay điều khiển motor/gimbal.

## Kiểm tra lại

```powershell
& .\.venv\Scripts\python.exe .\scripts\check_streaming.py
```

Script chạy video blocks có sẵn trong repo, chọn khối chữ Y, lưu ảnh và số liệu
trong `test_results`; kiểm tra giới hạn bộ nhớ, ảnh trống, khóa mất mục tiêu,
chọn lại, gap dài và khung bị bỏ. Kiểm tra ảnh tĩnh lặp lại phục vụ đo bộ nhớ,
không đại diện cho tracking chuyển động. Xem `test_results/streaming_report.json`
cho kết quả đo trên máy này. Chưa xác minh tương tác GUI/camera bằng camera thật.

Kết quả đo ngày 04/10/2026: 120 khung video, TRACKING 39 khung đầu, LOST từ
khung index 39 (bị che/diện tích thay đổi), sau đó khóa mất mục tiêu. Đây không
phải accuracy 39/120: chủ động không tự nhận lại sau mất. Ảnh index 30 xác nhận
mask vẫn trên khối Y giữa nhiều khối chữ. Thời gian suy luận sau 5 khung đầu,
chỉ tính các bước thật sự chạy model: mean 245.48 ms (~4.07 FPS), p50 248.33 ms,
p95 253.14 ms. Lần khởi tạo đầu 1332.06 ms; UI có warmup trước khi cho chọn.
Peak tensor CUDA 613.72 MiB, allocator reserved peak 682 MiB; tensor đang giữ
sau các bước ảnh tĩnh 30/50/79 là 301.020/301.024/301.024 MiB. Đây là số liệu
PyTorch, không bao gồm CUDA context, driver hay phần mềm khác. Không tuyên bố
real-time. Windows torch hiện fallback attention vì thiếu Flash Attention.
Working set RAM toàn tiến trình tại ba mốc là 1565.840/1565.898/1565.945 MiB.
Đây là bài kiểm tra hữu hạn, chưa phải soak test chạy nhiều giờ.

`scripts/check_camera_pipeline.py` kiểm tra riêng các thread capture/inference
trên video phát theo nhịp nguồn, bao gồm bỏ khung, latency và sequence tăng.
Kết quả nằm ở `test_results/camera_pipeline_report.json`. Không mở GUI trong
kiểm tra này. Không cần cài thêm dependency; các gói hiện tại đã đủ.
Lần đo sau giới hạn luồng CPU: 9 kết quả có chạy model, mean inference 198.40 ms,
mean latency 223.35 ms, bỏ 25 khung trước các kết quả này. Mẫu ngắn và kết thúc
tracking khi che khuất; không ngoại suy thành FPS/độ chính xác cho mọi cảnh.
Kiểm tra Kalman riêng xác nhận dt=1 và dt=4 dịch chuyển đúng theo vận tốc,
covariance vẫn xác định dương. Hai backend DirectShow/MSMF không mở được camera
index 0 trong phiên này; lý do phần cứng/quyền riêng tư chưa xác định.

Môi trường đã kiểm tra: Windows/Python 3.11.0, torch 2.5.1+cu124,
torchvision 0.20.1+cu124, CUDA tensor chạy thành công trên RTX 3050 Laptop 6 GB,
OpenCV 5.0.0; `pip check` không báo xung đột. Script thêm `D:\Sam\sam2` vào
import path để tránh thư mục repo bên ngoài che package SAM 2 đã cài editable.
Checkpoint tải từ URL chính thức trong script download gốc của repo.

## Model bổ sung / tài liệu tham khảo

- [SAMURAI chính thức](https://github.com/yangchris11/samurai): API offline không
  hỗ trợ webcam sẵn. Adapter này bổ sung streaming, giữ model/thuật toán.
- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/): tên chính thức
  YOLO26 (không phải tên API tự đặt YOLOv26), có ví dụ `YOLO("yolo26n.pt")`.
  Chưa cài/tích hợp. Detector chỉ đề xuất vị trí/lớp, không chứng minh đúng
  mục tiêu đã chọn; cần xác minh ID riêng trước khi dùng để nhận lại tự động.
- [NanoSAM NVIDIA](https://github.com/NVIDIA-AI-IOT/nanosam): distilled SAM với
  ResNet18 encoder, MobileSAM decoder, TensorRT engines và quy trình export riêng.
  Có thể nhanh ở segmentation độc lập, nhưng không cung cấp memory encoder,
  object pointers hay motion-aware memory tương thích trực tiếp với SAM 2.1.
  Ghép thêm tốn VRAM, engine build và kiểm thử Windows/TensorRT; thay encoder
  âm thầm sẽ không còn pipeline SAMURAI này. Chưa có vai trò đủ lợi ích để thêm.
- Link TikTok được cung cấp không truy cập được trong phiên làm việc; không
  đưa ra mô tả hoặc kết luận về nội dung clip.

## File và sao lưu

- `scripts/live_tracker.py`: adapter model/state/gate/memory.
- `scripts/camera.py`: capture, inference worker, UI, telemetry.
- `scripts/check_streaming.py`: kiểm tra CUDA và streaming bằng video.
- `scripts/check_camera_pipeline.py`: kiểm tra hai worker bằng video theo nhịp nguồn.
- `CAMERA_TRACKING.md`, `README.md`: hướng dẫn, kiến trúc, kết quả đo và giới hạn.
- `sam2/sam2/modeling/sam2_base.py`: hỗ trợ key bộ nhớ đã bị evict.
- `sam2/sam2/utils/kalman_filter.py`: dt biến đổi, mặc định offline vẫn 1.
- `backups/pre_camera/`: bản trước sửa của hai file lõi và README. Không có Git
  trong PATH và workspace không có `.git`, nên dùng bản sao dự phòng.
- `sam2/checkpoints/sam2.1_hiera_tiny.pt`: checkpoint chính thức tải thêm.

Khôi phục hai file lõi bằng Copy-Item từ `backups/pre_camera` vào đường dẫn
gốc tương ứng. Các file demo và config có sẵn được giữ nguyên.
