# Giữ danh tính bằng memory SAM — 10/10/2026

Nhánh `feature/friend-recovery`; snapshot trước sửa: `d30516f`. Chỉ SAM 2.1 Tiny
và SAMURAI, không thêm model. `D:\Gimbal` không bị sửa. Chỉ lấy hành vi duy trì
propagation từ `scripts/demo.py` của bạn làm tham chiếu; chưa đối chiếu core
`sam2/` của bạn vì giới hạn đọc trước đó. Không tuyên bố đã sao chép một thuật toán
recovery nội bộ chưa được kiểm chứng.

```powershell
cd D:\Sam
& .\.venv\Scripts\python.exe .\scripts\camera.py --source 0 --sam-only --recovery friend
```

Đóng cửa sổ cũ và chọn lại mục tiêu một lần để tạo memory sạch. Không tiếp tục
dùng state đã nhận nhầm trước khi sửa. Phím S/R, W/C, Q/Esc và các phím video giữ
nguyên. `--recovery current` chạy lại policy cũ để đối chiếu. Camera/video,
SMALL/crop, flow, mask/viền, HUD, CSV, save-video và CUDA Graph vẫn được giữ.

## Thay đổi

- Policy mới `scripts/recovery/native_memory.py` tiếp tục gọi SAM bằng memory
  của đối tượng đã chọn, kể cả khi mất dấu. Không dùng mask ứng viên làm prompt
  mới để biến một vật cùng màu thành mục tiêu.
- Giữ conditioning anchor ban đầu. Khi cần xác minh, chạy thêm SAM trên **cùng
  frame và cached encoder features**, chỉ dùng anchor memory, không box/point/
  mask prompt. So mask này với mask temporal; không có frame tương lai.
- Màu chỉ được giữ trong log để chẩn đoán, không tham gia gate/ranking identity
  mới. Dùng SAM quality, similarity với anchor (không max trên gallery trôi dạt),
  hình dạng PCA bất biến theo xoay, diện tích và giới hạn chuyển động tăng theo
  thời gian mất. Đây là kiểm tra bổ sung, không phải khả năng ReID tuyệt đối của SAM.
- SAM point decoding chỉ kiểm tra ứng viên cạnh tranh để **từ chối** khi thiếu
  margin; mask sinh từ point không được thay identity memory. Nhánh SMALL/tile
  vẫn tìm kiếm nhưng phải được unprompted anchor memory xác nhận trước probation.
- Memory đóng băng trong SEARCHING/probation. Thử lại qua 3 quan sát liên tiếp
  và 5 khung probation; chỉ ghi memory sau xác minh cuối. Kalman bị cập nhật bởi
  quan sát bị loại được phục hồi. Probe anchor không thay đổi stream index/motion.
- Probe độc lập ở đầu/cuối thử lại, khi ứng viên nhảy vị trí và định kỳ 3 frame
  tracking; không lặp tất cả phép giải mã ở mọi frame probation. Các frame giữa
  vẫn chạy SAM với memory nguyên vẹn, kiểm tra quality/shape/anchor prototype.
- Thay crop được kiểm tra lại bằng anchor; thất bại phục hồi memory cửa sổ cũ.
  W loại memory gần đây có nguy cơ nhiễm và quay về anchor gốc để tìm tiếp.
- CSV thêm `recovery_policy`, `anchor_similarity`, `shape_ratio`,
  `anchor_mask_iou`, `memory_frozen`, `color_used_for_acceptance`,
  `motion_distance`, `motion_limit`, `lost_seconds`; không xóa cột cũ.
  Thời gian SAM trên HUD của policy mới tính cả xác minh, tránh báo FPS quá cao.

Ngưỡng mới trong `scripts/sam_memory.json`: native_shape_ratio=.4,
native_area_min=.15, native_area_max=6, native_mask_agreement=.55,
native_memory_interval=3, native_ambiguity_points=4. Không hạ ngưỡng similarity
để ép bắt lại. Các kiểm tra này có thể từ chối mục tiêu biến dạng/đổi tỷ lệ quá
mạnh; ưu tiên giữ mất dấu thay vì chuyển sang vật khác.

## Kiểm chứng và giới hạn

- Unit tests: memory/probation mới 6; memory cũ 5; SAM-only 4; smooth UI/flow 7;
  video playback 6; SMALL crop 2; tile 2. Tất cả đã qua. Gate replay cho thấy
  score SAM/Lab cao có thể khiến gate cũ gọi seed dù hình dạng khác; policy mới
  từ chối. Đây là replay tín hiệu, không phải phép đo camera.
- GPU A/B, hai fixture tự vẽ: vật trắng khác ở xa và chuyển vào vị trí vật gốc;
  mỗi policy 60 frame/fixture. Cả cũ và mới có 0 khóa nhầm; vật gốc quay lại được
  bám sau 0.200 s (cũ), 0.233 s (mới) **theo timeline nguồn**. Vì baseline không
  mắc lỗi trên fixture này, chưa chứng minh giảm tỷ lệ nhầm trong cảnh bút/giày
  của người dùng. Xem `test_results/native_memory/same_color.csv`, `summary.json`.
- Pipeline realtime headless trên clip tự tạo, giữ flow/CSV/save-video: phiên
  đo đầy đủ có 1 recovery, Live khoảng 23.5 FPS, SAM updates khoảng 3.4 FPS;
  bắt lại được hiển thị ở thời gian nguồn 6.53 s sau khi vật trở lại ở 3.00 s.
  **Chưa đạt mục tiêu Live >=30 hay recovery <=2 s.** Thử OpenCV một thread
  riêng không cải thiện (22.2 FPS), không áp cấu hình thử vào mặc định.
  Anchor probe/giải mã cạnh tranh tăng chi phí SAM; dense optical flow và ghi
  video có các đỉnh thời gian đáng kể trong profile. Không bỏ các tính năng này
  để làm đẹp số liệu. Xem `live_profile.json`, `live.csv`, `live_overlay.avi`.
- Chưa kiểm chứng trực tiếp camera hoặc clip bút/giày trong ảnh. Chưa có bảng
  20 lượt tự nhiên hay nhãn ID-switch trên xe đèo/300651 cho bản này. Không đọc
  hoặc thay các checkpoint/video của người dùng trong đợt sửa này; model hiện
  có chỉ được nạp để chạy kiểm tra.
- SAM vẫn có thể nhầm hai vật gần như giống hệt nhau. Bộ nhớ không phải dấu vân
  tay nhận dạng duy nhất; không bảo đảm tuyệt đối từ một ảnh chọn ban đầu.

File sửa: `camera.py`, `live_tracker.py`, `sam_memory.json`, `sam_tiles.py`,
`video_playback.py`, `PROGRESS.md`. File mới: module `recovery/`,
`check_native_memory.py`, `check_native_memory_gpu.py`, tài liệu này.
