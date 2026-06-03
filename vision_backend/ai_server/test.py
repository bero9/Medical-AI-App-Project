import cv2
import numpy as np
import easyocr
import os
from PIL import Image, ImageOps
from paddleocr import LayoutDetection

# --- Configuration ---
IMAGE_PATH = "photo_2026-06-02_17-17-49.jpg" 

def run_local_test():
    print("--- 🚀 Starting Standalone AI System Test (Final Fix) ---")
    
    try:
        # 1. Loading Engines
        print("[1/5] Initializing Layout Engine (PP-DocLayoutV2)...")
        layout_engine = LayoutDetection(model_name="PP-DocLayoutV2")
        
        print("[2/5] Initializing OCR Engine (EasyOCR)...")
        # Using CPU as per your previous terminal output
        ocr_reader = easyocr.Reader(['ar', 'en'], gpu=False) 

        # 2. Image Loading & Pre-processing
        print(f"[3/5] Reading image from: {IMAGE_PATH}")
        if not os.path.exists(IMAGE_PATH):
            print(f"❌ Error: File '{IMAGE_PATH}' not found!")
            return

        pil_img = Image.open(IMAGE_PATH)
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        img_np = np.array(pil_img)
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 3. Running Layout Detection
        print("[4/5] Analyzing document layout...")
        layout_res = layout_engine.predict(img_cv2)
        
        blocks = []
        if layout_res and len(layout_res) > 0:
            raw_data = layout_res[0]
            if isinstance(raw_data, dict):
                blocks = raw_data.get('boxes', raw_data.get('res', raw_data.get('result', [])))
            else:
                blocks = raw_data

        print(f"✅ Detection completed: Found ({len(blocks)}) blocks.")

        # 4. Running OCR on detected blocks
        print("[5/5] Extracting text from blocks...")

        for i, block in enumerate(blocks):
            # 💡 THE FIX: Using the exact key found in your terminal: 'coordinate'
            bbox = block.get('coordinate', block.get('bbox', block.get('box', None)))
            label = block.get('label', 'unknown')
            
            if bbox is None:
                print(f"   ⚠️ Warning: Still no coordinates found for Block {i+1}.")
                continue

            # Mapping the coordinates [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, bbox)
            
            # Crop the block from the original image
            crop = img_np[y1:y2, x1:x2]
            
            # Skip if crop is invalid
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                print(f"   ⚠️ Warning: Block {i+1} has invalid dimensions.")
                continue

            # Process crop with EasyOCR
            print(f"   🔎 Processing Block {i+1} [{label}]...")
            ocr_results = ocr_reader.readtext(crop)
            
            # Extract text with confidence > 20%
            text_content = " ".join([res[1] for res in ocr_results if res[2] > 0.20])

            if text_content.strip():
                print(f"   ✅ Block {i+1} [{label}]: {text_content}")
            else:
                print(f"   ❌ Block {i+1} [{label}]: No text detected inside this area.")

        print("\n--- ✅ System Test Finished Successfully ---")

    except Exception as e:
        print(f"\n❌ A Critical Error Occurred!")
        print(f"Details: {str(e)}")

if __name__ == "__main__":
    run_local_test()