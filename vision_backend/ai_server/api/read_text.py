import os
import tempfile
import cv2
import numpy as np
import easyocr
from PIL import Image, ImageOps
from paddleocr import LayoutDetection

# --- 🛠️ إعدادات التنقيب عن المشاكل (Debug Settings) ---
DEBUG_MODE = True 
DEBUG_DIR = "ocr_debug_crops"

if DEBUG_MODE and not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)

# 1. تحميل المحركات (مرة واحدة فقط عند تشغيل السيرفر)
print("--- 🚀 System Launch: Loading AI Engines ---")
layout_engine = LayoutDetection(model_name="PP-DocLayoutV2")
ocr_reader = easyocr.Reader(['ar', 'en'], gpu=False) # اجعلها True إذا كنت تملك GPU

def extract_text(image_file):
    # 2. استقبال وحفظ الملف المؤقت
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    for chunk in image_file.chunks():
        temp.write(chunk)
    temp.close()

    try:
        # 3. المعالجة الأساسية (السر في النجاح)
        pil_img = Image.open(temp.name)
        # تصحيح الدوران من الهاتف + التأكد من صيغة RGB
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        img_np = np.array(pil_img)
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        print(f"📸 Image Received. Shape: {img_np.shape}")

        # 4. تحليل الهيكل (Layout Detection)
        layout_res = layout_engine.predict(img_cv2)
        
        # استخراج البلوكات بشكل مرن
        blocks = []
        if layout_res and len(layout_res) > 0:
            raw_data = layout_res[0]
            if isinstance(raw_data, dict):
                blocks = raw_data.get('boxes', raw_data.get('res', raw_data.get('result', [])))
            else:
                blocks = raw_data

        print(f"🔍 Layout Stage: Found {len(blocks)} blocks.")

        all_segments = []

        # 5. معالجة البلوكات وقصها
        for i, block in enumerate(blocks):
            # استخدام المفتاح السحري 'coordinate' الذي اكتشفناه في التيرمنال عندك
            bbox = block.get('coordinate', block.get('bbox', None))
            label = block.get('label', 'text')
            
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                crop = img_np[y1:y2, x1:x2]
                
                if crop.size > 0:
                    # حفظ القصاصة للمعاينة إذا كان وضع الديبيج مفعل
                    if DEBUG_MODE:
                        crop_filename = os.path.join(DEBUG_DIR, f"block_{i}_{label}.jpg")
                        cv2.imwrite(crop_filename, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

                    # قراءة النص من القصاصة
                    res = ocr_reader.readtext(crop)
                    txt = " ".join([r[1] for r in res if r[2] > 0.15])
                    
                    if txt.strip():
                        print(f"✅ Found Text in Block {i}: {txt}")
                        all_segments.append(txt)

        # 6. 💡 الخطة البديلة (Fallback): إذا فشل الـ Layout، اقرأ الصورة كاملة فوراً
        if not all_segments:
            print("⚠️ Layout yielded no text. Starting FULL IMAGE SCAN...")
            full_results = ocr_reader.readtext(img_np)
            full_txt = " ".join([r[1] for r in full_results if r[2] > 0.15])
            
            if full_txt.strip():
                print(f"🚀 Full Scan Success: {full_txt}")
                all_segments.append(full_txt)

        # 7. تجهيز الرد النهائي
        if all_segments:
            combined_text = " . ".join(all_segments)
            tts_output = f"النص المكتشف هو: {combined_text}"
        else:
            tts_output = "لم أتمكن من قراءة أي نص، حاول تحسين الإضاءة."

        return {
            "tts_text": tts_output,
            "extracted_texts": all_segments
        }

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        return {"tts_text": "حدث خطأ فني أثناء القراءة.", "extracted_texts": []}
    
    finally:
        if os.path.exists(temp.name):
            os.remove(temp.name)