# Kế hoạch ghép recovery — Bước 1, 10/10/2026

**Trạng thái:** chỉ phân tích và ghi tài liệu; chưa sửa/chạy code, chưa baseline
hay ghép. Dừng chờ duyệt. `D:\Gimbal` chỉ đọc. Không đọc `sam2/`, `.venv`/`venv`,
checkpoint hoặc nội dung video. Không có AGENTS.md tại các thư mục gốc/entry đã
kiểm tra. Các phần nằm sau API predictor được đánh dấu CHƯA XÁC MINH, không suy
diễn từ chất lượng video người dùng báo.

## a. Kiến trúc và tương thích

**Hiện tại:** `scripts/camera.py:create_tracker` mặc định `--sam-only=True` →
`NativeCropTracker(SAMMemoryTracker(LiveTracker))`. Không đi qua YOLO/ReID cũ.
Camera/video realtime giữ inference worker, `realtime_video.py`, `mask_display.py`
và optical flow; offline qua `video_playback.py`. Checkpoint mặc định Tiny SAM 2.1,
encoder 512; CUDA Graph và memory hữu hạn. README cũ còn mô tả YOLO, không dùng
làm bằng chứng cho đường chạy mặc định hiện nay.

**Bạn:** không có README ở gốc; README tại `D:\Gimbal\samurai_source\README.md`
là README SAMURAI upstream. Entry rõ nhất là `D:\Gimbal\scripts\demo.py`:
1. OpenCV mở file, đọc FPS/kích thước/số frame; bbox từ dòng đầu txt, dạng xywh.
2. Thêm `D:\Gimbal\sam2` vào import path; build `build_sam2_video_predictor` với
   `configs/samurai/sam2.1_hiera_s.yaml`, mặc định `checkpoints/sam2.1_hiera_small.pt`.
3. CUDA FP16; `init_state(video_path, offload_video_to_cpu=True,
   offload_state_to_cpu=True)`; prompt duy nhất ở frame 0/object 0.
4. `propagate_in_video(state)` chạy liên tục; mỗi output đọc frame tương ứng,
   mask >0, nếu mask rỗng thì không vẽ đối tượng nhưng vẫn xuất frame và tiếp tục
   predictor. Không tự prompt lại một vật khác tại tầng script.
5. Ghi video đồng bộ; cuối cùng đóng images loader nếu có `close()`.

`demo_before_lazy.py` và `samurai_source/scripts/demo.py` là biến thể cũ, giữ toàn
bộ ảnh trong `loaded_frames`; không port loader này. `main_inference*.py` là
entry benchmark dataset, cấu hình base_plus hoặc tham số model; không chứng minh
đó là entry tạo kết quả bạn đang tham chiếu. Chưa được cung cấp lệnh đã chạy/tên
tác giả; mặc định small không chứng minh checkpoint thực tế đã dùng.

Phụ thuộc entry: Python stdlib, NumPy, OpenCV, PyTorch và SAMURAI/SAM2 local.
README yêu cầu Python >=3.10, torch >=2.3.1, torchvision >=0.18.1; còn liệt kê
matplotlib/tikzplotlib/jpeg4py/lmdb/pandas/scipy/loguru cho công cụ benchmark.
Không cài thêm các gói này ở Bước 1. Hai phía cùng tên họ SAM 2.1/SAMURAI, nhưng
small và tiny không đổi weights lẫn nhau; config phải khớp kiến trúc. Chưa xác
minh revision core, nội dung YAML, patches loader/predictor, key/shape checkpoint
hay môi trường cài đặt của bạn. Giữ Tiny hiện tại cho A/B đầu tiên để không trộn
ảnh hưởng đổi model vào ảnh hưởng thuật toán. Chưa cần copy checkpoint.

## b. Cơ chế mất/bắt lại: bằng chứng và khoảng trống

**Quan sát chắc chắn ở code bạn:** propagation tiếp tục sau mask rỗng, cùng
predictor/state/object ID; không latch mất dấu, không histogram chọn lại, không
reset box tự động. Ngưỡng duy nhất ở tầng vẽ là mask-logit >0/không rỗng.
Đây là hành vi tham chiếu đáng giữ, chưa đủ chứng minh có một thuật toán recovery
riêng để sao chép. `init_state` có lời bình lazy nhưng nội dung loader chưa đọc;
offload CPU không tự chứng minh RAM hữu hạn. Entry không gọi reverse propagation
hoặc prompt tương lai, nhưng chưa thể chứng nhận toàn bộ core là causal.

**CHƯA XÁC MINH bên bạn:** object-score/IoU/diện tích để quyết định mất; anchor,
lọc memory xấu, cửa sổ nhớ; sinh ứng viên; chấm điểm danh tính, motion, margin,
probation. Tất cả được ủy quyền vào `sam2/`, hiện bị loại trừ khỏi phạm vi đọc.
Không gán ngưỡng SAMURAI mặc định từ trí nhớ cho bản local của bạn.

**Điểm cần tái hiện bên hiện tại:**
- `sam_memory.py:assess`: score = .45 SAM prototype + .25 Lab + .25 mask IoU
  dự đoán + .05 stability; similarity dùng max gallery (24 mẫu, anchor giữ lại).
- TRACKING: logit >0; IoU .35/.55; SAM similarity >=.65; Lab >=.4; diện tích
  anchor ratio .02..12, collapse so gần nhất >=.12; 3 frame xấu mới mất, ngoại
  lệ mask rỗng/probation. Memory ghi khi IoU >=.7, logit >=1 và identity đủ tốt.
- SEARCH: IoU >=.65, stability >=.8 (SMALL .75), SAM similarity >=.8,
  Lab >=.6 (SMALL .8), logit >0 và diện tích hợp lệ.
- **Lỗ hổng cụ thể:** `sam_memory.py` và `sam_tiles.py` cho nhánh strong >=.9
  được nhận sau 2 frame mà **không kiểm tra margin**. Chỉ nhánh normal >=.83
  kiểm margin .04/3 frame; single >=.77/5 frame. Đây là nguy cơ, chưa phải kết
  luận thực nghiệm về nguyên nhân nhầm màu.
- Vị trí chỉ trừ tối đa .03 trong một đường proposal; không phải gate bắt buộc
  đồng nhất cho raw propagated mask và mọi đường tile/crop. Diện tích cho phép
  rất rộng; chưa có xác minh contour/shape độc lập ở gate chung.
- Probation 5 frame và rollback đã có. Snapshot thấy lưu outputs, pinned quality,
  active prompt, recent area; cần audit bổ sung Kalman/trạng thái temporal khi
  candidate bị rút lại. Không được mặc định rollback hiện tại đã đầy đủ.

## c. So sánh và quyết định có điều kiện

| Chức năng | Hiện tại | Bạn (phạm vi đã đọc) | Quyết định, lý do/rủi ro |
|---|---|---|---|
| Phát hiện mất | Nhiều quality/identity gate, 3 bad frames; vẫn gọi SAM với continue_lost | Mask rỗng chỉ ngừng vẽ; predictor tiếp tục | GIỮ gate an toàn và propagation; đối chiếu core trước khi THAY. Mask không rỗng không chứng minh đúng vật |
| Memory | Anchor + top-quality pinned 4 + recent 24; lọc trước commit | State predictor giữ qua toàn video; policy bên trong chưa biết | GIỮ bounded memory; chỉ port khác biệt có bằng chứng. Không mang cả state offline vào live |
| Sinh ứng viên | Raw SAM, heatmap SAM/negative background, Kalman grid, Lab points, tiles native | Không sinh lại ở entry, tiếp tục native propagation | GIỮ SMALL/tiles; BỔ SUNG ưu tiên giả thuyết propagation gốc nếu core xác nhận; không mặc định reset vào đỉnh màu |
| Điểm/xác minh | SAM+Lab+IoU+stability; motion penalty yếu | Chưa thấy gate riêng ở entry | BỔ SUNG gate chung SAM-quality + motion/size/shape/elapsed; màu chỉ phụ trợ. Đây là sửa của dự án hiện tại, chưa gọi là thuật toán lấy từ bạn |
| Chống vật cùng màu | Strong bỏ qua margin; max gallery dễ chấp nhận tương tự | Không có histogram re-lock tại entry; core chưa biết | THAY strong acceptance để mọi nhánh phải đủ margin; so các giả thuyết khác nhau và không khóa khi mơ hồ. Rủi ro tăng thời gian chờ |
| Chấp nhận/rút lại | Mask-prompt, 5 probation; snapshot memory một phần | Không re-init/probation ở entry | GIỮ probation; BỔ SUNG rollback đầy đủ và log theo attempt; không commit mẫu nghi ngờ |
| Live/UI/output | Async, camera+video, SMALL/crop, mask/contour, flow, HUD, CSV/save-video, CUDA graph | Video đồng bộ, không có live UI | GIỮ toàn bộ hiện tại; không port main loop/video writer của bạn |

## d. Kế hoạch online và các cổng kiểm chứng

1. **Cổng đọc bổ sung trước khi ghép:** cần người dùng cho phép ngoại lệ đọc đúng
   các phần liên quan của `D:\Gimbal\sam2\sam2\build_sam.py`,
   `sam2_video_predictor.py`, `modeling\sam2_base.py`,
   `configs\samurai\sam2.1_hiera_s.yaml`, loader được import trong `utils\misc.py`;
   và đoạn tương ứng bên D:\Sam nếu cần diff. Không quét toàn bộ sam2, không đọc
   weights. Nếu vẫn cấm, chỉ có thể port hành vi entry đã chứng minh và sửa gate
   hiện tại; không thể tuyên bố đã ghép cơ chế recovery nội bộ của bạn.
2. Sau duyệt: tạo `feature/friend-recovery`, commit snapshot hiện trạng trước
   sửa code. Git đang ở master với nhiều thay đổi chưa commit; không reset chúng.
   Chỉ stage scripts/config/docs đã rà soát, không `git add .`; loại media,
   checkpoint, venv, runtime và secret. Ownership git cần dùng scoped
   `git -c safe.directory=D:/Sam ...`, không đổi global config. Chưa tạo nhánh/
   commit trong Bước 1 chỉ đọc này. Core sam2 đang untracked và chưa đọc nên chưa
   thể tuyên bố snapshot git bao gồm toàn bộ model code.
3. Bước 2 mới dựng/annotate hai fixture: mục tiêu vắng nhưng distractor cùng màu
   còn đó; mục tiêu quay lại với/không distractor. Đóng băng seed/box/timeline và
   nhãn ID/visibility. Log mọi acceptance: SAM similarity, Lab, logit, IoU,
   stability, diện tích/aspect/shape, motion residual, elapsed, top2/margin,
   proposal source, memory commit và kết quả probation. Đo baseline trước sửa.
4. Module dự kiến `scripts/recovery/`: interface `update(frame,timestamp,
   sam_observation,context)` → trạng thái/điểm/quyết định; `try_reacquire(...)`
   tạo probation, commit/rollback theo transaction. Adapter không sở hữu UI,
   capture hoặc tải AI khác. Cờ `--recovery current|friend` tại factory; ban đầu
   current, chỉ đổi friend mặc định sau khi đạt kiểm thử. Cả hai vẫn sam-only.
5. Native propagation chỉ dùng frame <=t, prompt từ lúc chọn trước đó, memory
   giới hạn. Không reverse, không future-conditioned memory; tách batch ứng viên
   trong cùng frame (causal) khỏi batch nhiều frame. Crop/tile phải chuyển đúng
   tọa độ, giữ timestamp nguồn khi bỏ frame, không dùng index xử lý làm thời gian.
   Search chia budget; không chặn display. Tile sweep 4K trước đây được ghi nhận
   khoảng 7.5 s tối thiểu trong PROGRESS, là nút thắt phải đo, không hứa <=2 s.
6. Regress camera/video, full mask/contour, flow, HUD/keys/CSV/save-video,
   CUDA graph/SMALL. `scripts/eval_cases.json` đã khai báo mountain=test2,
   car300651=test3, beach=test1 và person; chưa mở video hay đánh giá ở bước này.
   Cần visibility/ID labels (hiện null). Báo bảng A/B: đúng/nhầm trên mọi sự kiện,
   median/p90 và failure censored, %TRACKING, Live FPS riêng SAM FPS, age và memory.
   Chỉ nghiệm thu khi cùng-màu false reacquire <=10% sự kiện, recovery >=baseline
   và >=80% trong <=2 s, Live FPS >=30; không dùng FPS video xuất làm Live FPS.
7. Header phần port ghi nguồn entry/file chính xác và tên tác giả khi được cung
   cấp; hiện chưa có tên nên không bịa. Phân biệt logic port và gate tự bổ sung;
   không copy secret hay đường dẫn máy tác giả vào runtime config.

**Chờ duyệt kế hoạch và phạm vi đọc core giới hạn ở mục d.1. Chưa thực hiện Bước 2–4.**
