import os
import time
import tempfile
import cv2
import numpy as np
import easyocr

from PIL import Image, ImageOps

# ==========================================
# Debug Settings
# ==========================================

DEBUG_MODE = True
DEBUG_DIR = "ocr_debug"

if DEBUG_MODE:
    os.makedirs(DEBUG_DIR, exist_ok=True)

# ==========================================
# Load EasyOCR once
# ==========================================

print("=" * 60)
print("🚀 Loading EasyOCR...")
reader = easyocr.Reader(
    ['ar', 'en'],
    gpu=False
)
print("✅ EasyOCR Loaded")
print("=" * 60)


# ==========================================
# Image Enhancement
# ==========================================

def preprocess_image(rgb):

    print("🛠 Preprocessing image...")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    gray = cv2.fastNlMeansDenoising(gray)

    return gray


# ==========================================
# OCR
# ==========================================

def extract_text(image_file):

    print("\n")
    print("=" * 70)
    print("🚀 extract_text() started")
    print("=" * 70)

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    )

    for chunk in image_file.chunks():
        temp.write(chunk)

    temp.close()

    try:

        # --------------------------------------
        # Load image
        # --------------------------------------

        pil = Image.open(temp.name)

        pil = ImageOps.exif_transpose(pil)

        pil = pil.convert("RGB")

        rgb = np.array(pil)

        print(f"📷 Image Shape : {rgb.shape}")

        if DEBUG_MODE:

            cv2.imwrite(
                os.path.join(DEBUG_DIR, "0_original.jpg"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            )

        # --------------------------------------
        # Enhance image
        # --------------------------------------

        enhanced = preprocess_image(rgb)

        if DEBUG_MODE:

            cv2.imwrite(
                os.path.join(DEBUG_DIR, "1_enhanced.jpg"),
                enhanced
            )

        # --------------------------------------
        # Detect Text
        # --------------------------------------

        print("\n🔍 Detecting text regions...")

        start = time.time()

        detections = reader.readtext(
            enhanced,
            detail=1,
            paragraph=False
        )

        elapsed = time.time() - start

        print(f"⏱ Detection Time : {elapsed:.2f} sec")

        print(f"📄 Number of detections : {len(detections)}")

        debug_img = rgb.copy()

        extracted = []

        # --------------------------------------
        # Loop over detections
        # --------------------------------------

        for idx, det in enumerate(detections):

            print("\n" + "-" * 50)

            box = det[0]
            text = det[1]
            confidence = det[2]

            print(f"Detection {idx}")
            print(f"Text       : {text}")
            print(f"Confidence : {confidence:.3f}")
            print(f"Box        : {box}")

            if confidence < 0.20:

                print("❌ Ignored (Low confidence)")
                continue

            pts = np.array(box).astype(int)

            x = pts[:, 0]
            y = pts[:, 1]

            x1 = max(0, np.min(x))
            y1 = max(0, np.min(y))

            x2 = min(rgb.shape[1], np.max(x))
            y2 = min(rgb.shape[0], np.max(y))

            crop = rgb[y1:y2, x1:x2]

            if crop.size == 0:

                print("❌ Empty crop")
                continue

            crop = cv2.resize(
                crop,
                None,
                fx=2,
                fy=2,
                interpolation=cv2.INTER_CUBIC
            )

            gray = cv2.cvtColor(
                crop,
                cv2.COLOR_RGB2GRAY
            )

            gray = cv2.copyMakeBorder(
                gray,
                20,
                20,
                20,
                20,
                cv2.BORDER_CONSTANT,
                value=255
            )

            if DEBUG_MODE:

                cv2.imwrite(
                    os.path.join(
                        DEBUG_DIR,
                        f"crop_{idx}.jpg"
                    ),
                    gray
                )

            cv2.polylines(
                debug_img,
                [pts],
                True,
                (0, 255, 0),
                2
            )

            cv2.putText(
                debug_img,
                str(idx),
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

            print("🔍 Running OCR on cropped region...")

            try:

                result = reader.readtext(
                    gray,
                    detail=0,
                    paragraph=True
                )

            except Exception as e:

                print("❌ OCR Error")
                print(e)

                continue
            print("OCR Result:")
            print(result)

            if len(result) == 0:
                print("⚠️ No text recognized in this crop.")
                continue

            final_text = " ".join(result).strip()

            if final_text == "":
                print("⚠️ Empty text after join.")
                continue

            print(f"✅ Final Text: {final_text}")

            extracted.append({
                    "text": final_text,
                    "confidence": confidence,
                    "x": x1,
                    "y": y1
                })

        # ======================================
        # حفظ صورة الـ Debug
        # ======================================

        if DEBUG_MODE:

            debug_path = os.path.join(
                DEBUG_DIR,
                "2_detected_boxes.jpg"
            )

            cv2.imwrite(
                debug_path,
                cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
            )

            print(f"\n🖼 Debug image saved : {debug_path}")

        # ======================================
        # ترتيب النصوص من أعلى لأسفل ثم من اليسار لليمين
        # ======================================

        extracted = sorted(
            extracted,
            key=lambda x: (x["y"], x["x"])
        )

        texts = []

        print("\n" + "=" * 60)
        print("Detected Texts")
        print("=" * 60)

        for item in extracted:

            print(item["text"])

            texts.append(item["text"])

        # ======================================
        # إنشاء النص النهائي
        # ======================================

        if len(texts) > 0:

            combined = " ".join(texts)

            tts_text = f"النص المكتشف هو: {combined}"

        else:

            print("❌ No text detected.")

            tts_text = (
                "لم أتمكن من العثور على نص واضح."
            )

        print("\n")
        print("=" * 60)
        print("FINAL RESULT")
        print("=" * 60)

        print(tts_text)

        return {
            "tts_text": tts_text,
            "extracted_texts": texts
        }

    except Exception as e:

        import traceback

        print("\n")
        print("=" * 60)
        print("CRITICAL ERROR")
        print("=" * 60)

        traceback.print_exc()

        return {
            "tts_text": "حدث خطأ أثناء قراءة النص.",
            "extracted_texts": [],
            "error": str(e)
        }

    finally:

        if os.path.exists(temp.name):
            os.remove(temp.name)

        print("🗑 Temporary image deleted.")
        print("=" * 60)