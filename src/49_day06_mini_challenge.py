import csv

import cv2

from day06_ocr_utils import (
    PROJECT_ROOT,
    load_rules,
    preprocess_for_ocr,
    run_ocr,
)

from day06_ocr_validation import (
    clean_ocr_text,
    validate_pattern,
)


# ==================================================
# 49_day06_mini_challenge.py  (Day06 Mini Challenge)
#
# 본 실습(SN 13자리 숫자)과 같은 흐름을
# 새로운 제품 4장의 Ch.-B. 코드에 다시 적용한다.
#
#   ROI 선택 → original 전처리 → 4배 확대 → OCR
#   → 문자열 정리 → 형식 검사 → 실제 값과 비교
#
# 교재 54번 코드에 다음을 추가했다.
#   1. 선택한 ROI 좌표를 CSV에 저장하고, 다시 실행하면 재사용
#      (매번 마우스로 다시 고르지 않아도 되고, 결과를 재현할 수 있음)
#   2. original이 실패한 이미지만 gray, clahe를 한 번씩 다시 시도 (교재 55번)
#   3. 교재의 느슨한 형식 검사 외에, Ch.-B. 실제 모양에 맞춘
#      엄격한 형식 검사도 함께 기록
# ==================================================


# ========================================
# 경로 설정
# ========================================

PRODUCT_DIR = PROJECT_ROOT / "samples" / "day06" / "mini_challenge" / "product"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "day06" / "mini_challenge"
REPORT_PATH = PROJECT_ROOT / "reports" / "day06_mini_challenge_results.csv"
ROI_PATH = PROJECT_ROOT / "configs" / "day06_mini_challenge_rois.csv"


# ========================================
# 기존 OCR 설정 가져오기 → Ch.-B.에 맞게 변경
# ========================================

rules = load_rules()

rules["preprocess_mode"] = "original"      # 본 실습에서 가장 좋았던 전처리로 시작
rules["ocr_psm"] = 7                        # 한 줄 문자
rules["whitelist"] = (                      # SN: 숫자만 → Ch.-B.: 영문 대문자 + 숫자 + -
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789-"
)
rules["expected_pattern"] = r"^[A-Z0-9-]+$"     # 교재 형식: 영문·숫자·- 로만 이루어졌는가

# 추가: 실제 Ch.-B. 모양 (예: NA679903-8A)
#   영문 2자 + 숫자 6자 + - + 숫자 1자 + 영문 1자
STRICT_PATTERN = r"^[A-Z]{2}[0-9]{6}-[0-9][A-Z]$"

# original이 틀렸을 때 한 번 더 시도할 전처리 (교재 55번)
RETRY_MODES = ["gray", "clahe"]


# ========================================
# 공통 함수
# ========================================

def read_text(roi, mode):
    """지정한 전처리로 OCR을 실행하고 정리된 문자열을 돌려준다."""
    rules["preprocess_mode"] = mode
    processed = preprocess_for_ocr(roi, rules)
    ocr_input = cv2.resize(processed, None, fx=4.0, fy=4.0,
                           interpolation=cv2.INTER_CUBIC)
    text = clean_ocr_text(run_ocr(ocr_input, rules)).upper()
    return text, ocr_input


def load_saved_rois():
    """이전에 선택해 둔 ROI가 있으면 읽는다."""
    if not ROI_PATH.exists():
        return {}
    with ROI_PATH.open(encoding="utf-8-sig") as fp:
        return {
            row["file"]: (int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"]))
            for row in csv.DictReader(fp)
        }


def save_rois(rois):
    ROI_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ROI_PATH.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.writer(fp)
        writer.writerow(["file", "x", "y", "w", "h"])
        for name, (x, y, w, h) in rois.items():
            writer.writerow([name, x, y, w, h])


# ========================================
# Mini Challenge 이미지 찾기
# ========================================

image_paths = sorted(
    path for path in PRODUCT_DIR.iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
) if PRODUCT_DIR.exists() else []

if not image_paths:
    raise FileNotFoundError(f"Mini Challenge 이미지가 없습니다: {PRODUCT_DIR}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

saved_rois = load_saved_rois()
if saved_rois:
    print(f"저장된 ROI를 사용합니다: {ROI_PATH}")
    print("(다시 고르고 싶으면 이 파일을 지우고 실행하세요)")

rows = []


# ========================================
# 4장 하나씩 검사
# ========================================

for image_path in image_paths:

    image = cv2.imread(str(image_path))
    if image is None:
        print("읽기 실패:", image_path.name)
        continue

    print()
    print("=" * 60)
    print("Image:", image_path.name)

    # ------------------------------------
    # Ch.-B. ROI: 저장된 값이 있으면 재사용, 없으면 마우스로 선택
    # ------------------------------------
    if image_path.name in saved_rois:
        x, y, w, h = saved_rois[image_path.name]
    else:
        print("Ch.-B.: 오른쪽의 코드 값만 마우스로 선택하고 Enter를 누르세요.")
        x, y, w, h = cv2.selectROI(
            f"Select Ch.-B. ROI - {image_path.name}",
            image,
            fromCenter=False,
            showCrosshair=True,
        )
        cv2.destroyAllWindows()
        if w == 0 or h == 0:
            print("ROI 선택이 취소되었습니다.")
            continue
        saved_rois[image_path.name] = (int(x), int(y), int(w), int(h))

    roi = image[int(y):int(y + h), int(x):int(x + w)]
    print(f"ROI: x={x}, y={y}, w={w}, h={h}")

    # ------------------------------------
    # 1차: original
    # ------------------------------------
    ocr_text, ocr_input = read_text(roi, "original")
    cv2.imwrite(str(OUTPUT_DIR / f"{image_path.stem}_chb_ocr_input.jpg"), ocr_input)
    cv2.imwrite(str(OUTPUT_DIR / f"{image_path.stem}_chb_roi.jpg"), roi)

    print("OCR 결과 (original):", ocr_text)

    # ------------------------------------
    # 이미지에서 직접 확인한 정답 입력
    # ------------------------------------
    expected_text = input("이미지에 실제로 보이는 Ch.-B. 값을 입력하세요: ").strip().upper()

    format_ok = validate_pattern(ocr_text, rules["expected_pattern"])
    strict_format_ok = validate_pattern(ocr_text, STRICT_PATTERN)
    exact_match = ocr_text == expected_text

    # ------------------------------------
    # original이 틀렸을 때만 gray, clahe를 한 번씩 다시 시도 (교재 55번)
    # 최종 판정을 바꾸지 않고, "도움이 되었는가"만 기록한다.
    # ------------------------------------
    retry_results = {}
    if not exact_match:
        for mode in RETRY_MODES:
            retry_text, retry_input = read_text(roi, mode)
            retry_results[mode] = retry_text
            cv2.imwrite(str(OUTPUT_DIR / f"{image_path.stem}_chb_{mode}.jpg"), retry_input)
            mark = "정답" if retry_text == expected_text else "틀림"
            print(f"  다시 시도 ({mode}): {retry_text}  → {mark}")

    rows.append({
        "file": image_path.name,
        "expected_text": expected_text,
        "ocr_text": ocr_text,
        "format_ok": format_ok,
        "strict_format_ok": strict_format_ok,
        "exact_match": exact_match,
        "gray_text": retry_results.get("gray", ""),
        "clahe_text": retry_results.get("clahe", ""),
    })

    print("Format OK        :", format_ok)
    print("Strict Format OK :", strict_format_ok)
    print("Exact Match      :", exact_match)


# ========================================
# ROI와 결과 저장
# ========================================

if not rows:
    raise RuntimeError("검사 결과가 없습니다.")

save_rois(saved_rois)

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
with REPORT_PATH.open("w", newline="", encoding="utf-8-sig") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


# ========================================
# 요약
# ========================================

success = sum(r["exact_match"] for r in rows)

print()
print("=" * 60)
print(f"OCR 성공 (Exact Match) : {success} / {len(rows)}")
print(f"형식 통과 (교재 Regex)  : {sum(r['format_ok'] for r in rows)} / {len(rows)}")
print(f"형식 통과 (엄격 Regex)  : {sum(r['strict_format_ok'] for r in rows)} / {len(rows)}")

for r in rows:
    if not r["exact_match"]:
        print(f"  실패: {r['file']}  OCR={r['ocr_text']}  실제={r['expected_text']}  "
              f"gray={r['gray_text']}  clahe={r['clahe_text']}")

print()
print("Saved:", REPORT_PATH)
print("ROI  :", ROI_PATH)
print("Day06 Mini Challenge Complete")
