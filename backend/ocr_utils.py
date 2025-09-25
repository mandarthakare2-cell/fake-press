
# ocr_utils.py
# Improved OCR pipeline: deskew, CLAHE, adaptive threshold, multi-pass, confidence filtering.
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image
import re

def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def deskew_image(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))
    if coords.shape[0] == 0:
        return cv_img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = cv_img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(cv_img, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def apply_clahe(cv_img):
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def denoise(cv_img):
    # fastNlMeansDenoisingColored parameters can be tuned
    return cv2.fastNlMeansDenoisingColored(cv_img, None, 10, 10, 7, 21)

def adaptive_thresh(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # gaussian adaptive threshold
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 25, 15)
    return th

def upscale(cv_img, scale=2):
    h, w = cv_img.shape[:2]
    return cv2.resize(cv_img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

def words_from_tesseract(cv_img, config='--oem 1 --psm 6', lang='eng', conf_threshold=20):
    data = pytesseract.image_to_data(cv_img, output_type=Output.DICT, config=config, lang=lang)
    words = []
    confs = []
    boxes = []
    n = len(data.get('text', []))
    for i in range(n):
        txt = data['text'][i].strip()
        conf = data['conf'][i]
        try:
            conf = float(conf)
        except:
            try:
                conf = int(conf)
            except:
                conf = -1.0
        if txt != '' and conf >= conf_threshold:
            words.append(txt)
            confs.append(conf)
            boxes.append((data['left'][i], data['top'][i], data['width'][i], data['height'][i]))
    return ' '.join(words), confs, boxes

def extract_phone_numbers(text):
    phones = re.findall(r'\\b\\d{10}\\b', text)
    return phones

def run_ocr_improved(pil_image, debug=False):
    \"\"\"Return best_text, details dict. Details include candidate passes and chosen confidence.\"\"\"
    cv_img_orig = pil_to_cv2(pil_image)

    # create preprocessing variants
    variants = []
    # 1. original (resized moderately)
    variants.append(('orig', cv_img_orig))
    # 2. deskewed
    try:
        desk = deskew_image(cv_img_orig)
        variants.append(('deskew', desk))
    except Exception:
        pass
    # 3. CLAHE + denoise
    try:
        clahe = apply_clahe(cv_img_orig)
        den = denoise(clahe)
        variants.append(('clahe_denoise', den))
    except Exception:
        pass
    # 4. adaptive threshold of clahe
    try:
        th = adaptive_thresh(clahe)
        # convert single channel thresh back to BGR for tesseract (it accepts gray as well)
        variants.append(('clahe_thresh', th))
    except Exception:
        pass
    # 5. upscaled original
    try:
        up = upscale(cv_img_orig, scale=2)
        variants.append(('upscaled', up))
    except Exception:
        pass

    candidates = []
    for name, var in variants:
        try:
            # if single-channel image (thresh), supply it directly; else pass BGR
            if len(var.shape) == 2:
                img_for_tess = var
            else:
                img_for_tess = cv2.cvtColor(var, cv2.COLOR_BGR2RGB)
            text, confs, boxes = words_from_tesseract(img_for_tess, conf_threshold=15)
            avg_conf = float(np.mean(confs)) if len(confs) > 0 else 0.0
            candidates.append({'variant': name, 'text': text, 'avg_conf': avg_conf, 'words': len(confs)})
            if debug:
                print(f\"Variant={name} words={len(confs)} avg_conf={avg_conf}\")
        except Exception as e:
            if debug:
                print('Tesseract failed on variant', name, str(e))
    # attempt EasyOCR as a fallback (optional)
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
        res = reader.readtext(cv2.cvtColor(cv_img_orig, cv2.COLOR_BGR2GRAY))
        easy_text = ' '.join([r[1] for r in res])
        candidates.append({'variant': 'easyocr', 'text': easy_text, 'avg_conf': (sum([r[2] for r in res])/len(res)) if len(res)>0 else 0.0, 'words': len(res)})
    except Exception:
        # easyocr not available or failed — ignore
        pass

    # pick best candidate by (avg_conf * words) heuristic
    best = None
    best_score = -1.0
    for c in candidates:
        score = c['avg_conf'] * max(1, c['words'])
        if score > best_score:
            best_score = score
            best = c

    best_text = best['text'] if best else ''
    phones = extract_phone_numbers(best_text)
    details = {'candidates': candidates, 'chosen': best, 'phones': phones}
    return best_text, details
