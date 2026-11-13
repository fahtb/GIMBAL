# ReID v2: đo trước, sửa crop và giảm thời gian bắt lại

File thay đổi/tạo mới:

| File | Vai trò |
|---|---|
| `scripts/person_reid.py` | State machine, hai mức, low gain, nhãn và gallery |
| `scripts/reid_features.py` | HSV vùng nhìn thấy, anchor/rolling, interface/fallback |
| `scripts/reid_diagnostics.py` | CSV từng frame, bọc đo baseline |
| `scripts/search_detector.py` | Thread detector latest-frame |
| `scripts/camera.py` | Kết nối thread, W/C, HUD, camera settings |
| `configs/reid.json`, `configs/camera.json` | Ngưỡng ReID và thiết lập camera |
| `scripts/check_reid.py`, `scripts/check_reid_gpu.py` | Unit/GPU tests cập nhật |
| `scripts/check_search_worker.py` | Kiểm tra thread và label qua pipeline thật |
| `scripts/benchmark_reid_cycles.py`, `scripts/summarize_reid_cycles.py` | 20 lượt trước/sau và bảng tổng hợp |
| `REID.md`, `REID_V2.md` | Tài liệu và liên kết phiên bản |

Không sửa `live_tracker.py`, thuật toán SAMURAI gốc, checkpoint hay CUDA Graph.

Đóng cửa sổ cũ, chạy lại từ PowerShell:

```powershell
cd D:\Sam
& .\.venv\Scripts\python.exe .\scripts\camera.py --source 0 --image-size 512
```

S/R và kéo chuột chọn người như trước; Q/Esc thoát. **W** đánh dấu lần bắt lại
đang hiển thị là nhầm; **C** đánh dấu đúng. Bấm lặp cùng nhãn không cộng nhiều
lần; đổi W↔C sửa nhãn của cùng attempt. Nhãn gắn với attempt ID hiển thị, không
được áp nhầm sang lần bắt lại mới khi worker đang chạy. Khi W, hệ thống quay về
tìm kiếm và bỏ rolling có nguy cơ nhiễm, giữ anchor. Không có gimbal thật được điều
khiển; `control_gain` trong JSON là hệ số để bộ điều khiển sau này áp dụng.

## Đo trước khi sửa

Đã sao lưu code/config vào `backups/pre_reid_v2/`. Baseline dùng nguyên code đó,
chỉ bọc đo CSV; không sửa ngưỡng trước khi đo. Bộ thử tái hiện mất dấu do crop:
240 lượt kiểm tra dưới ngưỡng, similarity 0.826–0.868 so với T_mid=0.88. Detector
đã thấy người ngay khi trở lại. Detector median 10.67 ms, p90 13.45 ms; khâu chờ
chính là appearance không đạt và lịch detector 0.15 s, không phải detector chậm.
Trong các lượt thành công, từ ứng viên đầu đến khởi tạo lại mất 0.4 s theo timeline,
sau đó xác minh thêm 0.3 s.

Code cũ chỉ `gallery.clear()` trong nhánh chọn thủ công, không trong nhánh tự bắt
lại. Chưa có log phiên người dùng nên chưa thể kết luận nguyên nhân cụ thể của
20→4. Bản mới log `selected`/session, `gallery_added`, attempt, gallery size và
nhãn W/C. Gallery chỉ thay anchor khi chọn thủ công; tự bắt lại giữ cả hai bank.
Test đã kiểm tra giữ nguyên 20 mẫu qua mất/bắt lại. Hiện gallery không tăng khi
chỉ nhìn cùng một hình vì đã bỏ mẫu trùng.

## Những thay đổi đang chạy

- **Appearance:** `VisibleHSV` lưu histogram theo các dải dọc crop, không lưu ảnh.
  Crop chạm trên/dưới hoặc quá rộng được đánh dấu; khi so với mẫu toàn người, chỉ
  so phần hiện diện trên/dưới/giữa. Tìm trong tập extent hữu hạn, tối thiểu khoảng
  35% thân người; không tìm một mảng màu nhỏ tùy ý. Cạnh trái/phải dùng vùng giữa
  crop, vẫn giữ các vùng dọc nhìn thấy. Đây là heuristic bbox, không phải pose
  estimation: cúi/vặn người quá mạnh vẫn có thể sai căn chỉnh.
- **Gallery:** 1 anchor bất biến + tối đa 19 rolling FIFO, thêm khi SAM tin cậy,
  có person detection khớp, ít overlap và đủ khác các mẫu cũ. `score_top_k=1`
  nghĩa là max; tăng thành k để dùng top-k mean. Mẫu trùng bị chặn khi cosine
  ≥`duplicate_similarity=0.985`. Tính sẵn template vùng để không tăng chi phí
  tìm kiếm quá nhiều: đo so một crop partial với 20 mẫu khoảng 0.185 ms, chưa tính
  detector/trích đặc trưng.
- **Hai mức:** giữ T_mid=`similarity=0.88`, N=`consecutive_frames=3`; T_high=
  `high_similarity=0.96`, `high_frames=1`. Cả hai vẫn cần margin 0.08 để không
  chọn ngẫu nhiên giữa hai người giống nhau. `single_candidate_decay=0` mặc định:
  chưa hạ ngưỡng theo thời gian. Nếu bật, chỉ hạ khi đúng một ứng viên, sau
  `decay_after`, không thấp hơn `minimum_similarity` (mặc định cũng 0.88).
- **Hai pha:** khi gate đạt, khởi tạo SAMURAI và Lucas–Kanade optical flow trên
  đúng khung/box đó. HUD hiện `TRACKING provisional - low gain`; trạng thái máy
  vẫn là REACQUIRE, `valid=true`, `provisional=true`, `control_gain=0.25`.
  SAMURAI tiếp tục cập nhật bộ nhớ và được kiểm tra bằng ReID/detector thêm 3
  khung. Optical flow làm bằng chứng chuyển động để phát hiện bất đồng với SAM,
  không quyết định danh tính; thiếu texture thì dùng bằng chứng SAM/ReID. Không
  chạy lại cơ chế giữ/xóa viền của MotionTracker cũ và không đưa mask flow vào SAM.
  Thất bại trở lại SEARCHING, dx/dy=null, gain=0. Khi xác minh xong gain=1.
- **Detector:** YOLOv8n person CUDA FP16, input 320. SEARCHING có thread
  `person-search` riêng lấy từng khung mới nhất, không bị timer 0.15 s hay SAM
  chặn. Một slot input/output, không tích lũy video. Worker chỉ dùng detections
  cùng chính xác frame/timestamp; frame bị bỏ không được đếm vào streak. Nếu máy
  không theo kịp camera, không đảm bảo xử lý mọi frame vật lý. SAM/YOLO giữ nguyên
  checkpoint, không có model mới được tải trong lần này.
- **OSNet interface:** `EmbeddingBackend.encode/compare`; inject adapter vào
  `ReIDTracker(..., embedder=adapter)`. `HSVFallbackBackend` lưu song song HSV và
  descriptor adapter, không trộn không gian vector; adapter lỗi thì cảnh báo và
  dùng HSV. OSNet/torchreid chưa được cài hoặc dùng; cần hiệu chỉnh score của
  backend đó trước khi triển khai thực tế.

Ngưỡng và tham số nằm trong `configs/reid.json`. `search_interval` chỉ còn để
đọc tương thích config cũ, không throttle đường tìm kiếm v2. `timeout=30` vẫn
kết thúc tìm kiếm để chờ S/R.

## CSV và HUD

`runtime/reid_search_frames.csv`: mỗi frame mà state machine xử lý lúc mất/tìm/
xác minh, ghi UTC, source timestamp, số ứng viên, top-1/top-2, T_mid thực tế,
T_high, margin yêu cầu/thực tế, streak/N, region đã so, detector ms và lý do:
`no_candidates`, `below_threshold`, `insufficient_margin`, `insufficient_streak`,
`provisional`, `sam_verification`, `confirmed`, `verification_failed:*`, timeout.

`runtime/reid_events.csv`: các mốc lost → first_candidate → provisional →
reacquired, cùng session/attempt trong note. **First candidate là người đầu tiên
detector nhìn thấy, chưa chắc là đúng mục tiêu.** Ground truth xuất hiện thật chỉ
biết trong bộ thử có kịch bản/nhãn. `label_correct`/`label_wrong` ghi nhãn C/W;
bộ đếm HUD là nhãn người dùng, không còn `wrong: unlabeled`. Số tự xác minh thành
công và số nghi ngờ sai vẫn được giữ riêng trong telemetry.

Các file CSV nối thêm giữa các lần chạy. Không mở khóa độc quyền bằng Excel khi
đang đo; nếu ghi lỗi, chương trình cảnh báo và tiếp tục, nên dữ liệu đo phiên đó
có thể thiếu. Không có hàng đợi log vô hạn. Bộ tổng hợp dùng session ID trong
report để không trộn dữ liệu giữa các lần đo.

## Camera và cách đọc độ trễ

`configs/camera.json`: backend `auto` thử DSHOW rồi MSMF, hoặc chọn `dshow`/
`msmf`; yêu cầu MJPG, 640×480, 30 FPS, buffer=1. Chương trình in accepted/readback
để biết driver có thực sự hỗ trợ không. Buffer phần mềm luôn một frame mới nhất;
không hứa buffer phần cứng là 1 nếu driver từ chối.

`auto_exposure=null` giữ thiết lập camera. Để khóa phơi sáng, đặt
`auto_exposure=false`, `manual_exposure_value=0`, `exposure=-6` (giá trị exposure
phải phù hợp camera). Để bật lại đặt true với `auto_exposure_value=1`. Những giá
trị bật/tắt này dựa trên code backend [DSHOW](https://github.com/opencv/opencv/blob/4.x/modules/videoio/src/cap_dshow.cpp)
và [MSMF](https://github.com/opencv/opencv/blob/4.x/modules/videoio/src/cap_msmf.cpp);
set thành công/readback không thay thế kiểm tra ảnh thực tế.

FPS đo từ số frame đọc được/thời gian, in mỗi 2 giây và hiện trên HUD. Latency là
từ khi `cap.read()` trả frame đến khi worker hoàn tất; age là tuổi frame tại lúc
vẽ. `age-latency` là thời gian kết quả chờ sau hoàn tất, **không phải độ trễ cảm
biến/USB**. Khóa phơi sáng và MJPG chưa xác minh được: DSHOW/MSMF camera 0 đều
không mở trong môi trường kiểm tra.

## Kết quả trước/sau

**Đây là 20 lượt mô phỏng từ một ảnh của video thật, không phải người thật rời
camera 20 lần.** Cùng ảnh, box chọn, tịnh tiến ra ngoài khung và trở lại; 10 lượt
toàn người, 10 lượt phóng lớn/cắt thân dưới. Timeline 10 Hz, chạy không chờ nhịp
thực, quan sát tối đa 6 giây sau trở lại. Mỗi lượt seed lại cùng anchor để hai
phiên bản so công bằng. Bộ thử chỉ có một người, chưa đánh giá real ID-switch.

| Chỉ số | Trước | Sau |
|---|---:|---:|
| Bắt lại thành công trong 6 giây nguồn | 12/20 (60%) | 20/20 (100%) |
| Trung vị / p90, chỉ các lượt thành công | 0.7 / 0.7 s | 0.3 / 0.3 s |
| p90 toàn bộ lượt, tính cả lượt chưa bắt lại | >6 s | 0.3 s |
| Nhầm trong fixture một người (proxy overlap) | 0/12 | 0/20 |
| Tỷ lệ nhầm danh tính với người gây nhiễu thật | Chưa đo | Chưa đo |

Sau sửa, detector median/p90 9.34/11.06 ms. Người xuất hiện được phát hiện ngay;
high gate vào provisional ở frame trở lại đầu tiên, vẫn tốn thời gian tính
detector+SAM, **không phải latency 0 ms**; xác minh còn 3 frame/0.3 s nguồn.
Xem [bảng đủ 20 lượt và phân rã thời gian](test_results/REID_COMPARISON.md),
`test_results/reid_cycles_before/` và `reid_cycles_after/` cho CSV/report gốc.

Kiểm tra bổ sung:

- 14 unit tests đạt: anchor không bị xóa, giữ 20 mẫu qua recovery, mẫu trùng,
  partial crop, margin, đổi ứng viên, timeout, low gain, W/C dedup/correction,
  fallback backend, không cấp control khi xác minh thất bại.
- `check_reid_gpu.py`: video người chuyển động + 15 khung che kín, SAM/YOLO GPU
  thực, bắt lại thành công; TRACKING trung bình khoảng 48.4 ms; CUDA allocated
  peak khoảng 306 MiB (không phải tổng VRAM tiến trình).
- `check_search_worker.py`: video phát theo nhịp, **thread capture/SAM/detector
  thật**, có rời/trở lại, xác minh low gain và lệnh C qua worker; latency trung
  bình khoảng 73.6 ms trong lần đo cuối; shutdown không còn thread detector.
- `check_telemetry.py`, kiểm tra cú pháp, `pip check` đạt. Chưa xác minh GUI,
  thao tác phím vật lý hoặc camera trực tiếp.

Chạy lại bộ so sánh:

```powershell
& .\.venv\Scripts\python.exe .\scripts\benchmark_reid_cycles.py --variant before
& .\.venv\Scripts\python.exe .\scripts\benchmark_reid_cycles.py --variant after
& .\.venv\Scripts\python.exe .\scripts\summarize_reid_cycles.py
```

HSV vẫn có thể nhận nhầm người giống màu khi người thật vắng mặt. Chấm điểm
partial/max cải thiện recall nhưng không biến histogram màu thành danh tính
chắc chắn. Cần phiên camera có W/C và nhiều người để đo tỷ lệ nhầm thực tế.
