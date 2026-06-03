import json
from pathlib import Path
import cv2
from paddleocr import LayoutDetection


def _detresult_to_boxes(res):
    """
    يحاول استخراج boxes من DetResult مهما كان شكل dict.
    """
    if hasattr(res, "to_dict"):
        d = res.to_dict()
    elif hasattr(res, "json"):
        j = res.json
        d = j() if callable(j) else j
    else:
        d = {}

    # حالات مختلفة شائعة
    if isinstance(d, dict):
        if "boxes" in d:
            return d.get("boxes", [])
        if "res" in d and isinstance(d["res"], dict):
            return d["res"].get("boxes", [])
        if "result" in d and isinstance(d["result"], dict):
            return d["result"].get("boxes", [])
        if "pred" in d and isinstance(d["pred"], dict):
            return d["pred"].get("boxes", [])

    return []


def layout_only(image_path: str, model_name="PP-DocLayoutV2"):
    """
    Layout Detection فقط.
    يرجع قائمة عناصر: label, score, bbox
    """
    model = LayoutDetection(model_name=model_name)
    output = model.predict(image_path, batch_size=1, layout_nms=True)

    items = []
    for res in output:  # res: DetResult
        boxes = _detresult_to_boxes(res)

        for b in boxes:
            coord = b.get("coordinate") or b.get("bbox") or b.get("box") or [0, 0, 0, 0]
            items.append({
                "label": b.get("label", "unknown"),
                "score": float(b.get("score", 0.0)),
                "bbox": [float(x) for x in coord]  # [xmin, ymin, xmax, ymax]
            })

    return items


def draw_layout(image_path, items, out_path="layout_annotated.png"):
    """
    يرسم مربعات حمراء + label أزرق فوق كل مربع.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Can't read image: {image_path}")

    h, w = img.shape[:2]

    for it in items:
        x1, y1, x2, y2 = it["bbox"]
        label = it["label"]

        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w - 1, int(x2)))
        y2 = max(0, min(h - 1, int(y2)))

        # مربع أحمر
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # label أزرق
        text_pos = (x1, max(20, y1 - 6))
        cv2.putText(
            img, label, text_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 0, 0), 2, cv2.LINE_AA
        )

    cv2.imwrite(out_path, img)
    return out_path


if __name__ == "__main__":
    HERE = Path(__file__).parent
    img_path = HERE / "output2.png"   # غيّر الاسم إذا لزم

    print("USING IMAGE:", img_path)

    items = layout_only(str(img_path))
    print("DETECTED ITEMS:", len(items))

    # لو صفر، اطبع تحذير واضح
    if len(items) == 0:
        print("WARNING: No layout boxes detected. Check image quality/format.")

    # حفظ JSON
    json_path = HERE / "layout_only.json"
    json_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved layout json ->", json_path)

    # رسم المربعات
    out_img = HERE / "layout_annotated2.png"
    draw_layout(str(img_path), items, str(out_img))
    print("Saved annotated image ->", out_img)
