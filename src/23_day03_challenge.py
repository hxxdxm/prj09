from pathlib import Path
import csv

import cv2
import numpy as np


# ==================================================
# 23_day03_challenge.py  (Day03 Mini Challenge)
#
# 검사 문제 : 주황색 병뚜껑 존재 여부 (색상 객체 검사)
# 정상(OK)  : 병 입구에 주황색 뚜껑이 있음
# 불량(NG)  : 뚜껑이 없어 투명한 병목이 보임
#
# 처리 흐름
#   ROI → HSV → 주황색 Mask → Morphology → Contour
#   → Feature 측정 → Rule → OK / NG
#
# Calibration : ok_01, ng_01  (기준값을 정하는 이미지)
# Test        : 나머지 이미지 (정한 기준을 확인만 하는 이미지)
# ==================================================


# --------------------------------------------------
# 1. 경로와 Parameter
# --------------------------------------------------

input_dir = Path("samples/day03/challenge")
output_dir = Path("outputs/day03/challenge")
report_path = Path("reports/day03_challenge_results.csv")

output_dir.mkdir(parents=True, exist_ok=True)
report_path.parent.mkdir(parents=True, exist_ok=True)

# 검사 ROI (이미 ROI로 자른 사진을 쓴다면 USE_ROI = False)
#
# 사진마다 병 위치가 좌우로 약 80px 달라서,
# 5장의 병뚜껑 위치를 모두 포함하도록 여유를 두었다.
# 픽셀 대신 "이미지 크기에 대한 비율"로 적어서
# 사진 해상도가 달라도 같은 위치를 자른다.
#   1265 × 1012 사진 기준 픽셀 좌표: (380, 110, 930, 340)
USE_ROI = True
ROI_RATIO = (0.300, 0.109, 0.735, 0.336)   # x1, y1, x2, y2 (0~1)

# 주황색 HSV 범위 (OpenCV: H 0~179, S·V 0~255)
# 병뚜껑 측정값: H 약 7~9, S 약 220~255, V 약 180~210
ORANGE_LOWER = (5, 120, 80)
ORANGE_UPPER = (20, 255, 255)

KERNEL_SIZE = 5            # Morphology Kernel 크기
MIN_AREA_RATIO = 0.01      # ROI의 1%보다 작은 덩어리는 Noise로 보고 제외

CALIBRATION_OK = "ok_01"
CALIBRATION_NG = "ng_01"


# --------------------------------------------------
# 2. 이미지 한 장의 Feature를 측정하는 함수
# --------------------------------------------------

def measure(image):

    # 2-1. ROI 자르기
    if USE_ROI:
        h, w = image.shape[:2]
        rx1, ry1, rx2, ry2 = ROI_RATIO
        x1, y1 = int(rx1 * w), int(ry1 * h)
        x2, y2 = int(rx2 * w), int(ry2 * h)
        roi = image[y1:y2, x1:x2]
    else:
        roi = image

    # 2-2. HSV로 바꾸고 주황색 Mask 만들기
    #      색상이 검사 기준이므로 Gray가 아니라 HSV를 사용한다.
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)

    # 2-3. Morphology: 작은 점 지우기(Open) → 작은 구멍 메우기(Close)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 2-4. Contour 찾기
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    roi_area = roi.shape[0] * roi.shape[1]
    valid = [c for c in contours if cv2.contourArea(c) >= roi_area * MIN_AREA_RATIO]

    # 2-5. Feature 계산
    orange_ratio = (mask > 0).sum() / roi_area
    largest = max(valid, key=cv2.contourArea) if valid else None
    largest_area_ratio = cv2.contourArea(largest) / roi_area if largest is not None else 0.0

    features = {
        "orange_ratio": orange_ratio,
        "largest_area_ratio": largest_area_ratio,
        "valid_count": len(valid),
    }

    return roi, mask, largest, features


# --------------------------------------------------
# 3. 이미지 목록 읽기
# --------------------------------------------------

patterns = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
image_files = sorted({p for pat in patterns for p in input_dir.glob(pat)})

if not image_files:
    raise RuntimeError(f"{input_dir}에 이미지가 없습니다.")

results = {}

for path in image_files:
    image = cv2.imread(str(path))
    if image is None:
        print("읽기 실패:", path.name)
        continue
    results[path.stem] = (path, *measure(image))


# --------------------------------------------------
# 4. Calibration: ok_01, ng_01로 기준값 정하기
#
# 기준값 = 두 이미지 orange_ratio의 가운데
# Test 이미지는 기준값을 정하는 데 사용하지 않는다.
# --------------------------------------------------

if CALIBRATION_OK not in results or CALIBRATION_NG not in results:
    raise RuntimeError(
        f"Calibration 이미지({CALIBRATION_OK}, {CALIBRATION_NG})가 필요합니다."
    )

ref_ok = results[CALIBRATION_OK][4]["orange_ratio"]
ref_ng = results[CALIBRATION_NG][4]["orange_ratio"]
threshold = (ref_ok + ref_ng) / 2

print(f"Calibration OK ({CALIBRATION_OK}) orange_ratio : {ref_ok:.4f}")
print(f"Calibration NG ({CALIBRATION_NG}) orange_ratio : {ref_ng:.4f}")
print(f"Threshold                          : {threshold:.4f}")
print()


# --------------------------------------------------
# 5. Rule 적용
#
# OK : orange_ratio >= threshold
# NG : orange_ratio <  threshold
# --------------------------------------------------

rows = []

print(f"{'file':10} {'role':12} {'expected':8} {'decision':8} {'correct':7} "
      f"{'orange':>7} {'margin':>8} {'count':>5}")

for stem, (path, roi, mask, largest, f) in results.items():

    expected = "OK" if stem.startswith("ok") else "NG" if stem.startswith("ng") else "?"
    decision = "OK" if f["orange_ratio"] >= threshold else "NG"
    correct = (expected == decision) if expected != "?" else ""
    role = "calibration" if stem in (CALIBRATION_OK, CALIBRATION_NG) else "test"
    margin = f["orange_ratio"] - threshold

    rows.append({
        "file": path.name, "role": role, "expected": expected, "decision": decision,
        "correct": correct, "orange_ratio": round(f["orange_ratio"], 4),
        "margin": round(margin, 4), "largest_area_ratio": round(f["largest_area_ratio"], 4),
        "valid_count": f["valid_count"],
    })

    print(f"{stem:10} {role:12} {expected:8} {decision:8} {str(correct):7} "
          f"{f['orange_ratio']:7.4f} {margin:+8.4f} {f['valid_count']:5d}")

    # 5-1. 결과 이미지: ROI | 주황색 Mask | 판정 표시
    view = roi.copy()
    painted = view.copy()
    painted[mask > 0] = (0, 255, 0)
    view = cv2.addWeighted(view, 0.6, painted, 0.4, 0)
    if largest is not None:
        cv2.drawContours(view, [largest], -1, (0, 255, 0), 2)
    color = (0, 200, 0) if decision == "OK" else (0, 0, 255)
    cv2.putText(view, f"{decision} {f['orange_ratio']:.3f}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    side = cv2.hconcat([roi, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), view])
    cv2.imwrite(str(output_dir / f"{stem}_result.jpg"), side)


# --------------------------------------------------
# 6. CSV 저장과 요약
# --------------------------------------------------

with open(report_path, "w", newline="", encoding="utf-8-sig") as fp:
    writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

tests = [r for r in rows if r["role"] == "test" and r["expected"] != "?"]

print()
print("Test 이미지 수 :", len(tests))
if tests:
    print("Test 일치     :", sum(r["correct"] for r in tests), "/", len(tests))
    closest = min(tests, key=lambda r: abs(r["margin"]))
    print(f"가장 기준에 가까운 Test 이미지: {closest['file']} (margin {closest['margin']:+.4f})")
print("CSV  :", report_path)
print("결과 :", output_dir)