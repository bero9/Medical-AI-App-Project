from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
import torch
import math
from PIL import Image, ImageOps

# =========================================================
# تحميل YOLO مرة واحدة فقط
# =========================================================
model = YOLO("yolov8n.pt")

# =========================================================
# تحميل MiDaS مرة واحدة فقط (Depth Estimation)
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

midas_model_type = "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", midas_model_type)
midas.to(device).eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

# =========================================================
# ترجمات عربية
# =========================================================
ARABIC_LABELS = {
    'person': 'شخص',
    'bicycle': 'دراجة هوائية',
    'car': 'سيارة',
    'motorcycle': 'دراجة نارية',
    'airplane': 'طائرة',
    'bus': 'حافلة',
    'train': 'قطار',
    'truck': 'شاحنة',
    'boat': 'قارب',
    'traffic light': 'إشارة مرور',
    'fire hydrant': 'صنبور إطفاء',
    'stop sign': 'إشارة توقف',
    'parking meter': 'عداد موقف',
    'bench': 'مقعد',
    'bird': 'طائر',
    'cat': 'قطة',
    'dog': 'كلب',
    'horse': 'حصان',
    'sheep': 'خروف',
    'cow': 'بقرة',
    'elephant': 'فيل',
    'bear': 'دب',
    'zebra': 'حمار وحشي',
    'giraffe': 'زرافة',
    'backpack': 'حقيبة ظهر',
    'umbrella': 'مظلة',
    'handbag': 'حقيبة يد',
    'tie': 'ربطة عنق',
    'suitcase': 'حقيبة سفر',
    'frisbee': 'قرص طائر',
    'skis': 'زلاجات',
    'snowboard': 'لوح تزلج',
    'sports ball': 'كرة رياضية',
    'kite': 'طائرة ورقية',
    'baseball bat': 'مضرب بيسبول',
    'baseball glove': 'قفاز بيسبول',
    'skateboard': 'لوح تزلج',
    'surfboard': 'لوح ركوب الأمواج',
    'tennis racket': 'مضرب تنس',
    'bottle': 'زجاجة',
    'wine glass': 'كأس نبيذ',
    'cup': 'كأس',
    'fork': 'شوكة',
    'knife': 'سكين',
    'spoon': 'ملعقة',
    'bowl': 'وعاء',
    'banana': 'موزة',
    'apple': 'تفاحة',
    'sandwich': 'شطيرة',
    'orange': 'برتقالة',
    'broccoli': 'بروكلي',
    'carrot': 'جزرة',
    'hot dog': 'هوت دوغ',
    'pizza': 'بيتزا',
    'donut': 'دونات',
    'cake': 'كيك',
    'chair': 'كرسي',
    'couch': 'كنبة',
    'potted plant': 'نبتة مزروعة',
    'bed': 'سرير',
    'dining table': 'طاولة طعام',
    'toilet': 'مرحاض',
    'tv': 'تلفاز',
    'laptop': 'حاسوب محمول',
    'mouse': 'فأرة',
    'remote': 'ريموت',
    'keyboard': 'كيبورد',
    'cell phone': 'هاتف محمول',
    'microwave': 'ميكروويف',
    'oven': 'فرن',
    'toaster': 'محمصة',
    'sink': 'مغسلة',
    'refrigerator': 'ثلاجة',
    'book': 'كتاب',
    'clock': 'ساعة',
    'vase': 'مزهرية',
    'scissors': 'مقص',
    'teddy bear': 'دبدوب',
    'hair drier': 'مجفف شعر',
    'toothbrush': 'فرشاة أسنان'
}

# =========================================================
# قراءة صورة الموبايل مع تصحيح EXIF
# =========================================================
def read_image_with_exif(path):
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)  # يصحح دوران الموبايل
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

# =========================================================
# تحديد الاتجاه
# =========================================================
def get_direction(x_center, img_width):
    if x_center < img_width * 0.33:
        return "على يسارك"
    elif x_center > img_width * 0.66:
        return "على يمينك"
    else:
        return "أمامك"

# =========================================================
# MiDaS raw inverse depth (بدون تطبيع)
# =========================================================
def compute_depth_map(img_rgb):
    inp = transform(img_rgb).to(device)
    with torch.no_grad():
        pred = midas(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
    depth_raw = pred.cpu().numpy().astype("float32")
    return depth_raw  # inverse depth خام

# =========================================================
# تقدير مسافة الشخص بالمتر (Pinhole Model)
# Z = f * H / h
# =========================================================
def estimate_person_distance_m(bbox, focal_px, person_height_m=1.7):
    x1, y1, x2, y2 = bbox
    h_px = (y2 - y1)
    if h_px <= 1:
        return None
    return float((focal_px * person_height_m) / h_px)

# =========================================================
# الدالة الرئيسية
# =========================================================
def analyze_image(image_file):
    # حفظ الصورة مؤقتًا
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    for chunk in image_file.chunks():
        temp.write(chunk)
    temp.close()

    # قراءة الصورة (مع EXIF)
    try:
        img = read_image_with_exif(temp.name)
    except Exception:
        img = cv2.imread(temp.name)

    if img is None:
        os.remove(temp.name)
        return {"tts_text": "تعذر قراءة الصورة", "obstacles": []}

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    # 1) حساب خريطة العمق الخام
    depth_map = compute_depth_map(img_rgb)

    # 2) تشغيل YOLO
    results = model(img_rgb, conf=0.4)

    obstacles = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            if cls_name not in ARABIC_LABELS:
                continue

            label_ar = ARABIC_LABELS[cls_name]

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            x_center = (x1 + x2) / 2
            direction = get_direction(x_center, w)

            # خذ العمق من مركز البوكس (أثبت)
            cx1 = int(x1 + 0.25*(x2-x1)); cx2 = int(x1 + 0.75*(x2-x1))
            cy1 = int(y1 + 0.25*(y2-y1)); cy2 = int(y1 + 0.75*(y2-y1))
            patch = depth_map[cy1:cy2, cx1:cx2]
            if patch.size == 0:
                patch = depth_map[y1:y2, x1:x2]
                if patch.size == 0:
                    continue

            depth_val = float(np.median(patch))
            depth_val = max(depth_val, 1e-3)  # حماية

            obstacles.append({
                "class_en": cls_name,
                "class": label_ar,
                "direction": direction,
                "depth_val": depth_val,
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })

    os.remove(temp.name)

    if not obstacles:
        return {"tts_text": "لا توجد عوائق أمامك", "obstacles": []}

    # =====================================================
    # 3) معايرة بالمتر (لو في شخص)
    # =====================================================
    persons = [o for o in obstacles if o["class_en"] == "person"]
    person_ref = None
    if persons:
        person_ref = max(persons, key=lambda o: (o["bbox"][2]-o["bbox"][0])*(o["bbox"][3]-o["bbox"][1]))

    scale_Zref = None
    depth_ref = None

    if person_ref:
        # focal من FOV تقريبي للموبايل (≈70°)
        FOV_DEG = 70.0
        focal_px = (w / 2) / math.tan(math.radians(FOV_DEG / 2))

        scale_Zref = estimate_person_distance_m(
            person_ref["bbox"],
            focal_px=focal_px,
            person_height_m=1.7
        )
        depth_ref = max(person_ref["depth_val"], 1e-3)

    # =====================================================
    # 4) احسب مسافة كل شيء
    # - إذا في شخص: متر حقيقي تقريبًا
    # - إذا ما في شخص: متر تقريبي (approx) اعتمادًا على عمق الصورة
    # =====================================================
    # fallback scale لو ما في شخص:
    if scale_Zref is None:
        # نثبت نقطة مرجعية تقريبية: اعتبر median عمق الصورة = 2.0m
        global_med = float(np.median(depth_map))
        global_med = max(global_med, 1e-3)
        scale_Zref = 2.0
        depth_ref = global_med
        approx_mode = True
    else:
        approx_mode = False

    for o in obstacles:
        d_obj = max(o["depth_val"], 1e-3)
        o["distance_m"] = float(scale_Zref * (depth_ref / d_obj))
        o["approx"] = approx_mode  # True إذا كانت تقديرية

    # الأقرب أولاً (inverse depth أكبر = أقرب)
    obstacles.sort(key=lambda x: x["depth_val"], reverse=True)

    # =====================================================
    # 5) بناء نص TTS
    # =====================================================
    tts_messages = []
    for o in obstacles:
        meters = round(o["distance_m"], 1)
        if o["approx"]:
            tts_messages.append(f'{o["class"]} {o["direction"]} يبعد تقريبًا {meters} متر')
            o["distance"] = f"~{meters} m"
        else:
            tts_messages.append(f'{o["class"]} {o["direction"]} يبعد حوالي {meters} متر')
            o["distance"] = f"{meters} m"

    tts_text = ". ".join(tts_messages)

    # تنظيف
    for o in obstacles:
        o.pop("depth_val", None)
        o.pop("class_en", None)

    return {
        "tts_text": tts_text,
        "obstacles": obstacles
    }
