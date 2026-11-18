# Phân tích train: raw vs clean

**Cấu hình:** yolo26n.pt, 20 epoch, imgsz 640, batch 16, seed 0 và 1, Colab T4. Val: `gold_val` (40 ảnh). Test: `data/test` (100 ảnh, nhãn người). Hai biến thể chỉ khác dữ liệu train.

| biến thể | #ảnh train | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|---|
| raw | 224 | 0.726 ± 0.000 | 0.512 ± 0.004 | 0.764 ± 0.061 | 0.649 ± 0.028 |
| clean | 203 | 0.699 ± 0.007 | 0.484 ± 0.005 | 0.758 ± 0.030 | 0.599 ± 0.037 |

| AP50 theo lớp | raw s0 | raw s1 | clean s0 | clean s1 | TB raw → clean |
|---|---|---|---|---|---|
| car | 0.751 | 0.724 | 0.747 | 0.751 | 0.738 → 0.749 |
| person | 0.813 | 0.832 | 0.818 | 0.796 | 0.823 → 0.807 |
| truck | 0.657 | 0.639 | 0.495 | 0.581 | **0.648 → 0.538** |
| bus | 0.681 | 0.709 | 0.714 | 0.687 | 0.695 → 0.701 |

## 1. Chênh mAP50 so với độ lệch giữa seed
- clean − raw = **−0,027**. Độ lệch chuẩn giữa seed là 0,000 (raw) và 0,007 (clean).
- Nhìn bề ngoài, chênh lệch lớn gấp khoảng 4 lần độ lệch. Nhưng độ lệch chuẩn ước từ **2 seed** gần như không có giá trị thống kê: raw ra 0,000 chỉ là do 2 lần chạy tình cờ trùng nhau.
- Theo từng lớp, nhiễu lớn hơn nhiều: AP truck của clean lệch 0,086 giữa 2 seed, lớn hơn cả chênh lệch tổng.
- **Kết luận:** với 2 seed và 100 ảnh test, *chưa kết luận được* bên nào tốt hơn. Car và bus nhích lên, person giảm nhẹ, đều trong mức ±0,02. Phần thua đến gần như hoàn toàn từ truck.

## 2. `raw` thật sự train trên bao nhiêu ảnh?
`train_images = 224` là số ảnh được *đưa vào*, nhưng Ultralytics tự bỏ hoặc biến đổi một phần. Dưới đây là ước tính bằng cách áp quy tắc kiểm nhãn của Ultralytics lên `data/raw`. Cần đối chiếu với log cảnh báo ở ô 6 của notebook.
- **2 ảnh hỏng** (`img_0010`, `img_0136`): không mở được, bị bỏ.
- **Khoảng 9 file nhãn** bị coi là hỏng nên **bỏ cả ảnh**: 4 file có id lớp > 3, 3 file có toạ độ âm, 1 file có toạ độ > 1, 1 file sai số cột. Chỉ một dòng hỏng cũng làm mất toàn bộ các box đúng còn lại của ảnh đó.
- **6 ảnh không có file nhãn** được coi là ảnh nền. Model bị dạy rằng các vật trong ảnh đó "không phải vật", tức nhãn âm sai.
- Box tràn nhẹ (cx ± w/2 > 1 nhưng từng giá trị ≤ 1) không bị Ultralytics bắt.

→ Raw thực tế chỉ học từ khoảng **213 ảnh**, trong đó còn 8 cặp ảnh trùng, 3 ảnh trùng gần đúng và 5 ảnh rò rỉ test. Clean học từ 203 ảnh đã sạch lỗi bắt buộc.

## 3. Raw có trùng tập test không? Bị thổi phồng bao nhiêu?
**Có.** Pipeline phát hiện 5 ảnh `LEAK_TEST` (`01_intake.csv`):

| ảnh raw | = ảnh test | lớp trong nhãn raw |
|---|---|---|
| img_0049 | test_033 | 4 person |
| img_0101 | test_039 | 1 truck |
| img_0167 | test_044 | 3 car, 1 person |
| img_0174 | test_085 | 1 car, 7 person |
| img_0221 | test_009 | 4 car |

Raw được train trên 5% đề thi nên điểm của raw bị **thổi phồng**. Mức thổi phồng chưa đo được vì cần file trọng số của model.

**Cách kiểm:** trên Colab, chấm lại `best.pt` của cả 2 biến thể trên 95 ảnh test còn lại (bỏ 5 ảnh trên), rồi so lại chênh lệch. Riêng `img_0101` = test_039 chứa truck, mà truck là lớp hiếm nhất, nên tác động rò rỉ có thể lớn nhất ở đúng lớp clean thua.

## 4. Lớp nào được lợi / bị hại?
- **car (+0,011)** và **bus (+0,006)** nhích lên. Có thể là nhờ các sửa lỗi bắt buộc trong `03_fixes.csv`: 8 box tràn được cắt, 7 box trùng bị xoá, 6 dòng sai id lớp bị xoá. Ở raw, các lỗi này làm Ultralytics bỏ cả ảnh. Tuy nhiên mức nhích chỉ khoảng 0,01, nằm trong nhiễu. Chưa tách được theo lớp: muốn kiểm thì lọc `03_fixes.csv` theo id lớp.
- **person (−0,016):** lớp nhiều box nhất (565). Thay đổi nằm trong nhiễu.
- **truck (−0,110), bị hại nhất.** Không phải do mất mẫu: box truck tăng 32 → 40, vì người duyệt bổ sung truck bị thiếu (riêng `img_0160` từ 3 lên 13 box) và đổi một số car thành truck (`img_0078`, `img_0220`). Nguyên nhân khả dĩ:
  - (a) mất ảnh rò rỉ test_039 có truck;
  - (b) truck dồn vào rất ít ảnh nên AP nhạy với từng ảnh;
  - (c) nhiễu seed (clean lệch 0,086 giữa 2 seed).

## 5. Clean còn lỗi gì? Đây có phải chặn trên chưa?
**Chưa phải chặn trên.**
- Hàng đợi người duyệt có **108 ảnh > ngân sách 80**, nên 28 ảnh không được xem.
- `07_error_list_v3` còn **19/203 ảnh có cảnh báo chưa duyệt**: 18 POSSIBLE_FALSE_POS, 3 MISSING, 1 WRONG_CLASS, 1 LOOSE_BOX, 4 EMPTY_LABEL, 4 BOX_TINY, 3 BOX_HUGE.
- Trên calib, `second_opinion` chỉ bắt được **20%** box lỏng (LOOSE_BOX recall 0,20) và 83% ca nhầm lớp. Clean chắc chắn còn box lỏng và nhầm lớp mà cả máy lẫn người chưa thấy.
- Chặn trên thật là biến thể `oracle` (nhãn người cho toàn bộ ảnh của clean), chưa chạy.

**Việc tiếp theo:**
1. Chấm lại trên test bỏ 5 ảnh rò rỉ.
2. Chạy ≥ 5 seed và thêm biến thể `oracle`.
3. Tăng ngân sách duyệt để phủ 28 ảnh còn lại.
