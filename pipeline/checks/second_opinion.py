"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401
from ..model_cache import get_preds  # noqa: F401

NAME = "second_opinion"


def _label_rows(ds, stem):
    """[(số dòng gốc, (lớp, cx, cy, w, h))] các dòng hợp lệ — giữ số dòng để báo FALSE_POS."""
    out = []
    for no, text in ds.read_label_lines(stem) or []:
        try:
            b = parse_yolo_line(text)
        except ValueError:
            continue
        if 0 <= b[0] < len(ds.classes):
            out.append((no, b))
    return out


def run(ds, cfg, preds=None):
    c = cfg["second_opinion"]
    conf_t, min_area, fp_conf = c["conf"], c["min_area_px"], c["fp_conf"]
    iou_missing, iou_match, iou_loose = c["iou_missing"], c["iou_match"], c["iou_loose"]
    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    issues = []
    for img in ds.image_files():
        stem = img.stem
        if stem not in preds or not ds.label_path(stem).exists():
            continue
        rows = preds[stem] or []
        labels = _label_rows(ds, stem)
        size = ds.image_size(stem)
        W, H = size if size else (0, 0)

        def add(code, msg, line=None, **extra):
            issues.append(Issue(NAME, WARN, code, stem, msg, line, extra))

        # --- model thấy gì mà nhãn không khớp
        for r in rows:
            cls, conf, box = int(r[0]), float(r[1]), tuple(r[2:6])
            if conf < conf_t:
                continue
            if size and box[2] * W * box[3] * H < min_area:
                continue
            cands = [(iou_xywh(box, lb[1:]), lb[0]) for _, lb in labels]
            cands = [(iou, lc) for iou, lc in cands if iou >= iou_missing]
            if not cands:
                add("POSSIBLE_MISSING", f"model thấy lớp {_name(ds, cls)} (conf={conf:.2f}) nhưng nhãn không có box nào",
                    conf=round(conf, 4), cls=cls, box=[round(v, 4) for v in box])
                continue
            same = [iou for iou, lc in cands if lc == cls]
            if same:
                best = max(same)
                if best < iou_loose:
                    add("POSSIBLE_LOOSE_BOX", f"box lớp {_name(ds, cls)} lệch/lỏng so với model (IoU={best:.2f})",
                        conf=round(conf, 4), iou=round(best, 4), cls=cls)
                continue
            best, lc = max(cands)
            if best >= iou_match:
                add("POSSIBLE_WRONG_CLASS", f"nhãn ghi {_name(ds, lc)} nhưng model nói {_name(ds, cls)} "
                    f"(conf={conf:.2f}, IoU={best:.2f})", conf=round(conf, 4), iou=round(best, 4),
                    label_cls=lc, pred_cls=cls)

        # --- nhãn có box mà model không thấy vật nào ở đó
        fp_boxes = [tuple(r[2:6]) for r in rows if float(r[1]) >= fp_conf]
        for no, lb in labels:
            best = max((iou_xywh(lb[1:], pb) for pb in fp_boxes), default=0.0)
            if best < iou_missing:
                add("POSSIBLE_FALSE_POS", f"box lớp {_name(ds, lb[0])} không khớp vật nào model thấy (IoU tốt nhất={best:.2f})",
                    line=no, iou=round(best, 4), cls=lb[0])
    return issues


def _name(ds, cls):
    return ds.classes[cls] if 0 <= cls < len(ds.classes) else str(cls)
