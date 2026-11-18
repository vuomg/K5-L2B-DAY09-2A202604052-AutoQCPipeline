# Báo cáo Day 09 — Nguyễn Quốc Vương, 2A202604052

## 1. Kết luận trước
Pipeline kết luận **Release** (`08_summary.json`, mã thoát 0). **224 ảnh vào → 203 ảnh ra dataset** (`submission/export/clean_v1`): 21 ảnh bị loại ở cửa vào (8 trùng từng byte, 3 trùng gần đúng, 5 rò rỉ tập test, 2 ảnh hỏng, 3 ảnh quá nhỏ).
Sau sửa tự động + 80 ảnh người duyệt, `07_error_list_v3` còn **0 lỗi bắt buộc** và **19/203 ảnh (9,4%)** còn cảnh báo chưa duyệt, dưới ngưỡng Hold 10%.
Lưu ý: 9,4% chỉ cách ngưỡng 0,6 điểm phần trăm, tức thêm 2 ảnh có cảnh báo là thành Hold. Release này **mong manh**, không phải "sạch hoàn toàn".

## 2. Lỗi và phân việc
Số lỗi lấy từ `01_intake` + `02_error_list_v1` (trước khi sửa). Mẫu số: 224 ảnh ở cửa vào, 203 ảnh ở QC.

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| DUP_IMAGE_EXACT | 8 ảnh | máy (loại ảnh) | trùng từng byte (md5), không có rủi ro |
| DUP_IMAGE_NEAR | 3 ảnh | máy (loại ảnh) | phash ≤ 8; trùng gần đúng làm lệch phân bố |
| LEAK_TEST | 5 ảnh | máy (loại ảnh) | trùng `data/test` (test_009/033/039/044/085): giữ lại thì điểm test bị thổi phồng |
| CORRUPT_IMAGE / BAD_SIZE | 2 / 3 ảnh | máy (loại ảnh) | không mở được / cạnh ngắn < 250 px |
| BAD_CLASS_NAME | 1 tệp (`data.yaml`, 2 tên sai) | máy (`rename_classes`) | tên lớp đã cố định trong `cfg["classes"]` |
| BAD_FORMAT / BAD_CLASS_ID / BOX_ZERO / DUP_BOX | 3 / 6 / 4 / 7 dòng | máy (`drop_line`, 20 dòng) | dòng hỏng không có cách sửa đúng duy nhất; xoá an toàn hơn đoán |
| BOX_OUT_OF_IMAGE | 8 dòng | máy (`clip_box`) | cắt về [0,1], giữ lớp; box nằm hẳn ngoài ảnh thì xoá |
| MISSING_LABEL_FILE | 6 ảnh | **người** (`needs_label`) | máy không được bịa nhãn; file rỗng nghĩa là "ảnh nền", khác "chưa gán" |
| POSSIBLE_MISSING / WRONG_CLASS / LOOSE_BOX / FALSE_POS | 38 / 16 / 14 / 27 ảnh | **người** (hàng đợi) | ý kiến của yolo26l chỉ là cảnh báo, model cũng có thể sai |
| EMPTY_LABEL / BOX_TINY / BOX_HUGE / CLASS_RARE | 12 / 4 / 3 / 1 | **người** (hàng đợi) | là ảnh nền hay thiếu nhãn? cần mắt người |

Máy ghi 35 thay đổi trong `03_fixes.csv` (20 `drop_line`, 8 `clip_box`, 6 `needs_label`, 1 `rename_classes`), mỗi dòng có cột trước → sau.

**Hàng đợi** (`05_review_queue.csv`, 108 ảnh > ngân sách 80): điểm ưu tiên = tổng trọng số mã × (1 + conf của model), cộng thêm điểm cho dòng đã bị máy xoá/cắt.
- Đầu hàng là `img_0020` (11,5 điểm: đã bị xoá dòng + MISSING + WRONG_CLASS), tiếp theo là 6 ảnh `NEEDS_LABEL` (10 điểm mỗi ảnh). Thứ tự này hợp lý: ảnh chưa có nhãn thì chỉ người mới gán được, và nếu không ai gán thì cả lô bị Rework.
- Với 80 ảnh, người xem phủ **84/93 cảnh báo ngữ nghĩa** (72/81 ảnh) trong `04_error_list_v2`.
- 28 ảnh không được duyệt có điểm từ 2,0 trở xuống. Nhóm lớn nhất là ảnh chỉ có POSSIBLE_FALSE_POS (9 ảnh), tiếp theo là ảnh chỉ bị máy xoá dòng (5) hoặc cắt box (4).

## 3. Calibration (2 ngưỡng)
Số liệu lấy từ `submission/calibration/calibration_report.md` (39 ảnh calib đã chèn lỗi). P / R ghi theo dạng precision / recall.

| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | **8** | NEAR 1,00/0,50 → 1,00/0,67; EXACT giữ 1,00/1,00 | Từ 4 đến 14 cho cùng P/R trên calib. Chọn 8 ở giữa vùng ổn định, xa vùng báo nhầm (≥ 16, xem bên dưới). Đổi lại, vẫn bỏ sót ảnh trùng bị biến đổi mạnh. |
| `second_opinion.conf` | 0.25 | **0.75** | WRONG_CLASS 0,56/0,83 → 1,00/0,83; MISSING 0,50/1,00 → 0,64/0,88; LOOSE_BOX 0,29/0,40 → 0,50/0,20 | Tăng precision để người duyệt bớt tốn công vào cảnh báo nhầm. Cái giá: MISSING bỏ sót 1/8 ca, LOOSE_BOX bỏ sót thêm 1/5 ca. Lên 0.8 thì MISSING recall rơi xuống 0,75 mà không được gì thêm, nên dừng ở 0.75. FALSE_POS không đổi (0,31/1,00) vì dùng `fp_conf` cố định. |

Tác động trên lô raw: cảnh báo vòng 1 giảm từ 273 xuống 123, hàng đợi từ 145 xuống 108 ảnh, kết luận từ Hold 29,3% thành Release 9,4%.

**Chỗ calibration đánh lừa:** trên calib, `phash_dist = 16` có F1 cao nhất cho DUP_IMAGE_NEAR (0,91 so với 0,80). Nhưng trên lô raw nó loại **31 ảnh** (so với 21 ở mức 8). Mở xem 3 cặp chỉ bị coi là trùng ở mức 16:
- `img_0011` (tennis) ↔ `img_0008` (trượt tuyết): **khác hẳn**;
- `img_0177` (lướt sóng) ↔ `img_0113` (người ngồi bàn): **khác hẳn**;
- `img_0084` ↔ `img_0074`: trùng thật (ảnh bị cắt/lệch).

Vậy ở mức 16 thì 2/3 cặp là báo nhầm, và ảnh bị loại nhầm sẽ mất khỏi dataset. Calib chỉ có 6 cặp NEAR nên 1 cặp đúng thêm đã đẩy F1 lên cao, trong khi phân bố của lô raw khác. Ngoài ra, ngay sát mức 16 là mức 20, nơi EXACT recall rơi xuống 0,33, nên vùng này rất dễ đổ. Tương tự, conf 0.75 được chọn trên chỉ 8 ca MISSING và 6 ca WRONG_CLASS: mỗi ca sai làm recall đổi 12–17 điểm.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
**(a) Gate chặn đúng.** Chạy `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb` (không có người duyệt):
```
kết luận: REWORK — còn 6 lỗi bắt buộc trên 6 ảnh/tệp (MISSING_LABEL_FILE×6) -> phải làm lại
Bước 5/5  KHÔNG xuất: còn lỗi bắt buộc (Rework) — trả về bước 3
exit=20
```
Thư mục `work/exp_nb` **không được tạo**, tức không có dataset nào được xuất. Đây chính là 6 ảnh `NEEDS_LABEL`: máy không bịa nhãn, nên không có người gán thì cổng chặn lại.

**(b) Điểm mù có bằng chứng:**
1. **Ảnh trùng bị biến đổi mạnh:** cặp `img_0084`/`img_0074` (cắt/lệch) có khoảng cách phash > 8 nên lọt qua. Trên calib, NEAR recall chỉ đạt 0,67 (bỏ sót 2/6 cặp).
2. **Box lỏng nhẹ:** LOOSE_BOX recall trên calib chỉ **0,20** (bắt được 1/5 ca). Box lệch ít (IoU với dự đoán ≥ 0,5) không bao giờ bị cờ. Không có check cú pháp nào bắt được loại lỗi này.
3. **Nhầm lớp mà model lớn cũng nhầm:** WRONG_CLASS recall 0,83, bỏ sót 1/6 ca chèn lỗi. Nếu yolo26l cũng gọi truck là car thì nhãn sai vẫn khớp với "ý kiến thứ hai" và không bị cờ.
4. **Cảnh báo được thả qua:** 19 ảnh còn cảnh báo (chủ yếu là 18 POSSIBLE_FALSE_POS, precision trên calib chỉ 0,31) đã được phát hành mà chưa ai xem, vì tỉ lệ dưới 10%. Gate đếm theo tỉ lệ, không theo mức độ nghiêm trọng.

## 5. Train: raw vs clean
yolo26n.pt, 20 epoch, imgsz 640, batch 16, seed 0 và 1, Colab T4; cùng val (`gold_val`, 40 ảnh) và cùng test (100 ảnh). Bảng do `scripts/summarize_results.py` sinh:

| biến thể | #ảnh train | số seed | map50 (TB ± độ lệch) | map50_95 (TB ± độ lệch) | precision (TB ± độ lệch) | recall (TB ± độ lệch) |
|---|---|---|---|---|---|---|
| raw | 224 | 2 | 0.726 ± 0.000 | 0.512 ± 0.004 | 0.764 ± 0.061 | 0.649 ± 0.028 |
| clean | 203 | 2 | 0.699 ± 0.007 | 0.484 ± 0.005 | 0.758 ± 0.030 | 0.599 ± 0.037 |

Chênh mAP50 (clean − raw) = **−0,027**.

AP50 theo lớp (trung bình 2 seed): car 0,738 → 0,749 · person 0,823 → 0,807 · **truck 0,648 → 0,538** · bus 0,695 → 0,701.

**Đọc kết quả:**
- Chênh −0,027 lớn hơn độ lệch giữa seed (0,000 và 0,007). Nhưng độ lệch chuẩn tính từ **2 seed** gần như không đáng tin, và test chỉ có 100 ảnh.
- Gần như toàn bộ chênh lệch nằm ở **truck**, lớp hiếm (40 box sau pipeline). Riêng clean đã dao động 0,495 ↔ 0,581 giữa 2 seed, lệch 0,086, lớn hơn cả chênh lệch tổng.
- **Kết luận trung thực:** clean *chưa* được chứng minh tốt hơn raw. Số liệu cũng *chưa đủ* để nói pipeline làm hại dữ liệu.

**Giả thuyết và cách kiểm:**
1. **Raw được thổi phồng do rò rỉ test:** raw chứa 5 ảnh trùng tập test (test_009/033/039/044/085). *Kiểm:* chấm lại cả 2 model trên 95 ảnh test còn lại.
2. **Truck dồn vào ít ảnh, không phải mất mẫu:** đếm box hợp lệ thì truck *tăng* 32 → 40 (raw → clean). Nhưng riêng `img_0160` đã chiếm +10 box, do người duyệt bổ sung các truck bị thiếu (3 → 13 box). Một số box car cũng được đổi thành truck (`img_0078`, `img_0220`). Kết quả là số ảnh chứa truck vẫn rất ít, nên AP truck nhạy với từng ảnh. Ngoài ra, ảnh rò rỉ `img_0101` (= test_039) chứa truck: raw đã "thấy trước" ảnh này, clean thì không, nên giả thuyết 1 tác động mạnh nhất đúng vào lớp truck. *Kiểm:* chấm AP truck trên test không có test_039; đếm số *ảnh* chứa truck ở raw và clean.
3. **Nhiễu:** *Kiểm:* thêm seed (≥ 5) và báo khoảng tin cậy.

Chi tiết phân tích: `submission/results/ANALYSIS.md`.

## 6. Quyết định và ghi chú AI
**(a) Quyết định cuối: Release `clean_v1`**, kèm 2 điều kiện:
- Ưu tiên người duyệt 19 ảnh còn cảnh báo trước khi dùng cho sản phẩm thật, vì Release đang chỉ cách ngưỡng 0,6 điểm phần trăm.
- Tách 5 ảnh rò rỉ test ra khỏi mọi phép so sánh trong tương lai.

Nếu ở lô sau tỉ lệ cảnh báo vượt 10% (Hold), cần tăng ngân sách duyệt (hàng đợi 108 > 80) thay vì nới ngưỡng `conf`.

**(b) Có dùng AI (Claude Code):**
- **Prompt chính:** "hoàn thành 4 module của repo yêu cầu". AI đọc docstring và test, viết 4 module, chạy sweep, chạy thử nhiều cặp ngưỡng trên raw.
- **Cách kiểm:**
  - `pytest` 79/79;
  - chạy `pipeline.run` trên dữ liệu thật và đọc `03_fixes.csv` (chỉ còn 6 lỗi `MISSING_LABEL_FILE` sau sửa, đúng thiết kế);
  - **mở bằng mắt** các cặp ảnh bị coi là trùng ở `phash_dist = 16`;
  - chạy `--budget 0` để thấy cổng ra Rework.
- **AI sai:**
  - Lần đầu AI đề xuất `conf = 0.6` vì chỉ nhìn lưới sweep thô (0.6, 0.7, 0.8); sweep mịn hơn mới thấy 0.75 tốt hơn.
  - Sau đó AI nói 0.75 "tốt hơn ở mọi chỉ số", nhưng so với giá trị khởi đầu thì **LOOSE_BOX recall giảm 0,40 → 0,20**. AI đã so với 0.6 chứ không so với khởi đầu, phải đối chiếu lại `calibration_report.md` mới thấy.

**(Thưởng) CVAT — lọc box nhỏ.**
- **Thiết lập:** CVAT local 2.81 (`localhost:8080`). Tạo project `Day09-QC` từ `cvat/labels.json` và task `review-top20` gồm 25 ảnh: 20 ảnh đầu `05_review_queue.csv` và 5 ảnh có `BOX_TINY`. Import nhãn sau autofix theo định dạng YOLO 1.1 (88 box).
- **Bộ lọc `width < 20`:** khớp **23 box trên 7 ảnh**:

  | Ảnh | Số box khớp |
  |---|---|
  | img_0020 | 4 |
  | img_0026 | 6 |
  | img_0030 | 4 |
  | img_0051 | 3 |
  | img_0069 | 2 |
  | img_0078 | 1 (car) |
  | img_0161 | 3 |

  22/23 box là **person** đứng xa: rộng 9–20 px nhưng cao 22–84 px.
- **Ảnh chụp:** [`submission/cvat/filter_width20.png`](cvat/filter_width20.png) (job 23, `img_0020`, bộ lọc `width < 20` → *Items: 4*).
  - Ba box nhỏ (15,0 × 50,7 / 16,8 × 55,3 / 19,5 × 54,3 px) là người thật ở phía sau người mặc áo hồng.
  - Box `PERSON 7` (16,4 × 33,2 px) nằm trong cửa kính xe buýt, có thể là hành khách bên trong hoặc hình phản chiếu. Đây là ca cần người quyết định giữ hay xoá (đúng loại ca nên tick `can_review`), và pipeline không cờ box này.
- **Đối chiếu với `BOX_TINY`:** pipeline cờ 7 box trên 5 ảnh:

  | Ảnh | Dòng bị cờ |
  |---|---|
  | img_0026 | 6 |
  | img_0030 | 9 |
  | img_0051 | 5 |
  | img_0069 | 2 |
  | img_0161 | 2, 3, 4 |

  Cả 7 box đều nằm trong tập CVAT, ngược lại thì không: 16 box rộng 12–20 px chỉ CVAT bắt được, vì pipeline dùng ngưỡng 12 px (cạnh ngắn) còn slide dùng 20 px. Riêng `img_0020` (4 box) và `img_0078` (1 box) nằm ở đầu hàng đợi nhưng pipeline không cờ `BOX_TINY`. Các box này là người và xe ở xa, có thật trong ảnh, nên **không** dùng *Remove filtered shapes* xoá hàng loạt. Bộ lọc `width < 20` hợp để *xem*, không hợp để *xoá*.
- **`can_review`:** người gán tick thuộc tính này khi không chắc chắn (vật bị che, quá nhỏ, khó phân biệt car/truck) để người thứ hai xem lại. Pipeline làm việc tương tự bằng cảnh báo WARN: `BOX_TINY` và `POSSIBLE_*` được dồn vào `05_review_queue.csv` và xếp ưu tiên theo trọng số × conf, thay vì để máy tự sửa. Nhãn import từ YOLO không mang thuộc tính này, nên mọi box đều có `can_review = false`.
