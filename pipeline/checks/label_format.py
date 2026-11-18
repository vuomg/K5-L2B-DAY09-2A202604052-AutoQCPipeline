"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401

NAME = "label_format"


def run(ds, cfg):
    c = cfg["label_format"]
    tol, min_px, max_frac, dup_iou = c["overflow_tol"], c["min_box_px"], c["max_box_frac"], c["dup_box_iou"]
    n_cls = len(ds.classes)
    issues = []

    for lab in ds.label_files():
        stem = lab.stem
        size = None                     # (W, H) — chỉ mở ảnh khi cần
        seen = []                       # (số dòng, box) các dòng hợp lệ phía trước, để bắt DUP_BOX
        for ln_no, text in ds.read_label_lines(stem) or []:
            def add(sev, code, msg, **extra):
                issues.append(Issue(NAME, sev, code, stem, msg, ln_no, extra))

            try:
                cls, x, y, w, h = parse_yolo_line(text)
            except ValueError as e:
                add(ERROR, "BAD_FORMAT", f"dòng không đúng định dạng YOLO ({e}): {text.strip()!r}")
                continue
            if not 0 <= cls < n_cls:
                add(ERROR, "BAD_CLASS_ID", f"id lớp {cls} ngoài 0..{n_cls - 1}", class_id=cls)
                continue
            if w <= 0 or h <= 0:
                add(ERROR, "BOX_ZERO", f"box không có diện tích (w={w}, h={h})")
                continue

            # tràn mép ảnh: lấy mức tràn lớn nhất trong 4 mép
            overflow = max(0.0, -(x - w / 2), (x + w / 2) - 1, -(y - h / 2), (y + h / 2) - 1)
            if overflow > tol:
                add(ERROR, "BOX_OUT_OF_IMAGE", f"box tràn ra ngoài ảnh {overflow:.4f}", overflow=round(overflow, 6))

            box = (x, y, w, h)
            for prev_no, (pc, pb) in seen:
                if pc == cls:
                    iou = iou_xywh(pb, box)
                    if iou > dup_iou:
                        add(ERROR, "DUP_BOX", f"trùng box ở dòng {prev_no} (cùng lớp, IoU={iou:.3f})",
                            iou=round(iou, 4), dup_of=prev_no)
                        break
            seen.append((ln_no, (cls, box)))

            if size is None:
                size = ds.image_size(stem) or False
            if size:
                W, H = size
                short = min(w * W, h * H)
                if short < min_px:
                    add(WARN, "BOX_TINY", f"box quá nhỏ: cạnh ngắn {short:.1f}px < {min_px}px", short_px=round(short, 2))
            if w * h > max_frac:
                add(WARN, "BOX_HUGE", f"box chiếm {w * h:.0%} ảnh (> {max_frac:.0%})", area_frac=round(w * h, 4))
    return issues
