"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401

DROP_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
CLIP_CODES = {"BOX_OUT_OF_IMAGE"}
REMOVE_CODES = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}          # + mọi mã bắt đầu bằng DUP_IMAGE_
DECIMALS = 6


def _is_remove(code):
    return code in REMOVE_CODES or code.startswith("DUP_IMAGE_")


def _fx(image, line, action, before, after):
    return {"image": image, "line": "" if line is None else line, "action": action, "before": before, "after": after}


def _clip(text):
    """Cắt box về [0,1]. Trả dòng mới, hoặc None nếu box không còn diện tích (nằm hẳn ngoài ảnh)."""
    c, x, y, w, h = parse_yolo_line(text)
    x1, x2 = max(0.0, x - w / 2), min(1.0, x + w / 2)
    y1, y2 = max(0.0, y - h / 2), min(1.0, y + h / 2)
    nx, ny = round((x1 + x2) / 2, DECIMALS), round((y1 + y2) / 2, DECIMALS)
    nw, nh = round(x2 - x1, DECIMALS), round(y2 - y1, DECIMALS)
    if nw <= 0 or nh <= 0:          # so SAU khi làm tròn: tránh ghi ra box "0.000000" (BOX_ZERO ở vòng sau)
        return None
    return f"{c} {nx:.{DECIMALS}f} {ny:.{DECIMALS}f} {nw:.{DECIMALS}f} {nh:.{DECIMALS}f}"


def _remove_image(ds, stem):
    removed = []
    for p in [ds.image_path(stem), ds.label_path(stem)]:
        if p is not None and p.exists():
            p.unlink()
            removed.append(p.name)
    return removed


def _rename_classes(ds, cfg):
    if not ds.yaml_path.exists():
        return None
    data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
    old = ds.yaml_names()
    new = {i: n for i, n in enumerate(cfg["classes"])}
    if old == new:
        return None
    data["names"] = new
    ds.yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return _fx("data.yaml", None, "rename_classes",
               ", ".join(f"{k}:{v}" for k, v in sorted(old.items())),
               ", ".join(f"{k}:{v}" for k, v in new.items()))


def _fix_lines(ds, stem, codes_by_line):
    """Sửa một file nhãn: gom mọi dòng lỗi, ghi lại MỘT lần (số dòng trong issue là số dòng gốc)."""
    path = ds.label_path(stem)
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out_lines, fixes = [], []
    for no, text in enumerate(raw, 1):
        codes = codes_by_line.get(no)
        if not codes or not text.strip():
            out_lines.append(text)
            continue
        new = None
        if codes & DROP_CODES:
            action = "drop_line"
        else:                                           # chỉ còn BOX_OUT_OF_IMAGE
            try:
                new = _clip(text)
            except ValueError:                          # dòng đã không parse được: issue cũ, xoá cho an toàn
                new = None
            action = "clip_box" if new is not None else "drop_line"
        if action == "clip_box":
            if new.split() == text.split():
                out_lines.append(text)
                continue
            out_lines.append(new)
            fixes.append(_fx(stem, no, "clip_box", text.strip(), new))
        else:
            fixes.append(_fx(stem, no, "drop_line", text.strip(), "(đã xoá)"))
    if fixes:
        path.write_text("".join(l + "\n" for l in out_lines), encoding="utf-8")
    return fixes


def fix(ds, issues, cfg):
    errors = [i for i in issues if i.severity == ERROR]
    fixes = []

    # 1) xoá ảnh trước: issue dòng của ảnh đã xoá sẽ bị bỏ qua êm
    removed = set()
    for i in errors:
        if _is_remove(i.code) and i.image not in removed:
            removed.add(i.image)
            gone = _remove_image(ds, i.image)
            if gone:
                fixes.append(_fx(i.image, None, "remove_image", ", ".join(gone), f"(đã xoá: {i.code})"))

    # 2) sửa tên lớp trong data.yaml (một lần)
    if any(i.code == "BAD_CLASS_NAME" for i in errors):
        f = _rename_classes(ds, cfg)
        if f:
            fixes.append(f)

    # 3) sửa dòng nhãn — gom theo file
    per_file = defaultdict(lambda: defaultdict(set))
    for i in errors:
        if i.image in removed or i.line is None:
            continue
        if i.code in DROP_CODES or i.code in CLIP_CODES:
            per_file[i.image][i.line].add(i.code)
    for stem in sorted(per_file):
        fixes.extend(_fix_lines(ds, stem, per_file[stem]))

    # 4) ảnh chưa có nhãn: KHÔNG bịa, chuyển người gán
    seen = set()
    for i in errors:
        if i.code == "MISSING_LABEL_FILE" and i.image not in removed and i.image not in seen:
            seen.add(i.image)
            if not ds.label_path(i.image).exists():
                fixes.append(_fx(i.image, None, "needs_label", "(không có file)", "chuyển người gán"))
    return fixes
