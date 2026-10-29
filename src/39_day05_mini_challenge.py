from pathlib import Path
from itertools import product
import csv
import json

import cv2
import numpy as np

from day05_logo_features import (
    prepare_reference,
    measure_target,
)


# ==================================================
# 39_day05_mini_challenge.py  (Day05 Mini Challenge)
#
# 5일차에 광천김으로 한 과정을 다른 제품에 다시 적용한다.
#
#   ① Reference 준비
#   ② Calibration OK / NG 측정
#   ③ Feature별 OK / NG 비교
#   ④ Threshold 후보 시험 → Rule 결정
#   ⑤ Rule을 JSON으로 고정 (Freeze)
#   ⑥ Test에 같은 Rule 적용
#   ⑦ TP / TN / FP / FN 계산
#
# 공통 측정 함수는 day05_logo_features.py를 그대로 재사용한다.
# ==================================================


# --------------------------------------------------
# 0. 경로
# --------------------------------------------------

ROOT = Path("samples/day05/mini_challenge")

REFERENCE_PATH = ROOT / "reference/reference_label.jpg"
CALIBRATION_DIR = ROOT / "calibration"
TEST_DIR = ROOT / "test"

OUTPUT_DIR = Path("outputs/day05/mini_challenge")
RULE_PATH = Path("configs/day05_mini_challenge_rule.json")
CALIBRATION_CSV = Path("reports/day05_mini_challenge_calibration.csv")
TEST_CSV = Path("reports/day05_mini_challenge_test.csv")

for folder in (OUTPUT_DIR, RULE_PATH.parent, CALIBRATION_CSV.parent):
    folder.mkdir(parents=True, exist_ok=True)

IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

FEATURES = [
    "template_score",
    "sift_good_matches",
    "homography_inliers",
    "inlier_ratio",
]


# --------------------------------------------------
# 공통 함수
# --------------------------------------------------

def list_images(folder):
    return sorted({p for pattern in IMAGE_PATTERNS for p in folder.glob(pattern)})


def measure_folder(folder, actual, reference_data):
    """폴더 안의 이미지를 모두 측정한다. actual = 'OK' 또는 'NG'."""
    rows = []
    for path in list_images(folder):
        try:
            result = measure_target(path, reference_data)
        except cv2.error as error:
            # 예: 이미지가 Reference보다 작으면 Template Matching이 실패한다.
            print(f"  측정 실패: {path.name} → {error.err}")
            continue
        result["file"] = path.name
        result["actual"] = actual
        rows.append(result)
    return rows


def save_csv(rows, path, extra_columns=()):
    columns = ["file", "actual"] + list(extra_columns) + FEATURES + ["homography_found"]
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in row.items() if k in columns})


def apply_rule(row, rule):
    """고정된 Rule로 OK / NG를 판정하고, NG이면 이유를 함께 돌려준다."""
    if rule["require_homography"] and row["homography_found"] != 1:
        return "NG", "NG_NO_HOMOGRAPHY"
    for feature, threshold in rule["thresholds"].items():
        if row[feature] < threshold:
            return "NG", f"NG_LOW_{feature.upper()}"
    return "OK", ""


def save_preview(row, decision, reason, folder):
    """판정 결과 이미지: 찾은 로고 위치(사각형)와 판정을 표시한다."""
    image = row["target_image"].copy()
    color = (0, 200, 0) if decision == "OK" else (0, 0, 255)
    if row["polygon"] is not None:
        cv2.polylines(image, [np.int32(row["polygon"])], True, color, 4)
    text = f"{row['actual']} -> {decision} {reason}"
    cv2.putText(image, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)
    cv2.putText(image,
                f"SIFT {row['sift_good_matches']}  Inlier {row['homography_inliers']}  "
                f"Ratio {row['inlier_ratio']:.3f}  Template {row['template_score']:.3f}",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imwrite(str(folder / f"{Path(row['file']).stem}_result.jpg"), image)


# --------------------------------------------------
# ① Reference 준비  (TODO 1)
# --------------------------------------------------

reference_data = prepare_reference(REFERENCE_PATH)
print("Reference:", REFERENCE_PATH)
print("Reference Keypoints:", len(reference_data["keypoints"]))
print()


# --------------------------------------------------
# ② Calibration OK / NG 측정  (TODO 2)
# --------------------------------------------------

calibration = (
    measure_folder(CALIBRATION_DIR / "ok", "OK", reference_data)
    + measure_folder(CALIBRATION_DIR / "ng", "NG", reference_data)
)

ok_rows = [r for r in calibration if r["actual"] == "OK"]
ng_rows = [r for r in calibration if r["actual"] == "NG"]

if not ok_rows or not ng_rows:
    raise RuntimeError(
        f"Calibration 이미지가 부족합니다. OK {len(ok_rows)}장, NG {len(ng_rows)}장\n"
        f"{CALIBRATION_DIR}/ok, {CALIBRATION_DIR}/ng 폴더를 확인하세요."
    )

save_csv(calibration, CALIBRATION_CSV)

print("=" * 60)
print(f"Calibration 측정  (OK {len(ok_rows)}장 / NG {len(ng_rows)}장)")
print("=" * 60)
print(f"{'file':16} {'actual':6} {'template':>8} {'sift':>6} {'inlier':>7} {'ratio':>7} {'H':>3}")
for r in calibration:
    print(f"{r['file']:16} {r['actual']:6} {r['template_score']:8.3f} "
          f"{r['sift_good_matches']:6d} {r['homography_inliers']:7d} "
          f"{r['inlier_ratio']:7.3f} {r['homography_found']:3d}")
print()


# --------------------------------------------------
# ③ Feature별 OK / NG 비교  (TODO 3)
#
# gap = OK 최솟값 - NG 최댓값
#   gap > 0 : 한 기준값으로 OK와 NG를 완전히 나눌 수 있다.
#   gap ≤ 0 : 두 분포가 겹친다.
# --------------------------------------------------

print("=" * 60)
print("Feature 비교  (gap > 0 이면 완전히 분리됨)")
print("=" * 60)
print(f"{'feature':20} {'OK min~max':>18} {'NG min~max':>18} {'gap':>9}")

feature_gap = {}
for feature in FEATURES:
    ok_values = [r[feature] for r in ok_rows]
    ng_values = [r[feature] for r in ng_rows]
    gap = min(ok_values) - max(ng_values)
    feature_gap[feature] = gap
    print(f"{feature:20} {min(ok_values):8.3f} ~ {max(ok_values):7.3f} "
          f"{min(ng_values):8.3f} ~ {max(ng_values):7.3f} {gap:+9.3f}")
print()


# --------------------------------------------------
# ④ Threshold 후보 시험 → Rule 결정  (TODO 4)
#
# 광천김 실습과 같은 형식의 복합 Rule을 사용한다.
#   Homography 성공 AND SIFT >= a AND Inlier Ratio >= b
#
# 후보 기준값은 임의로 정하지 않고,
# Calibration 측정값 사이의 가운데 값들로 만든다.
# 고르는 순서:
#   1) 오류(FP+FN)가 가장 적은 것
#   2) FN(불량을 OK로 통과)이 적은 것  → 검사에서는 불량 통과가 더 위험
#   3) 실제로 쓰는 조건이 적은 것      → Calibration이 적을수록
#                                         조건이 많으면 데이터에만 맞춰지기 쉽다
#   4) 기준값과 가장 가까운 Calibration 값까지의 여유(margin)가 큰 것
# --------------------------------------------------

RULE_FEATURES = ["sift_good_matches", "inlier_ratio"]


def candidate_thresholds(feature):
    values = sorted({r[feature] for r in calibration})
    mids = [(a + b) / 2 for a, b in zip(values, values[1:])]
    return [None] + mids          # None = 이 Feature는 조건으로 쓰지 않음


def evaluate(thresholds):
    thresholds = {k: v for k, v in thresholds.items() if v is not None}
    rule = {"require_homography": True, "thresholds": thresholds}
    fp = fn = 0
    for r in calibration:
        decision, _ = apply_rule(r, rule)
        # 5일차 기준: Positive = NG
        if r["actual"] == "OK" and decision == "NG":
            fp += 1          # 정상을 불량으로 잘못 판정
        if r["actual"] == "NG" and decision == "OK":
            fn += 1          # 불량을 정상으로 통과 (놓침)
    # margin: 각 Feature 범위로 나눈 "기준과 가장 가까운 값까지의 거리"의 최솟값
    margins = []
    for feature, t in thresholds.items():
        values = [r[feature] for r in calibration]
        spread = (max(values) - min(values)) or 1.0
        margins.append(min(abs(v - t) for v in values) / spread)
    return fp, fn, (min(margins) if margins else 0.0)


results = []
for combo in product(*(candidate_thresholds(f) for f in RULE_FEATURES)):
    thresholds = {k: v for k, v in zip(RULE_FEATURES, combo) if v is not None}
    if not thresholds:
        continue
    fp, fn, margin = evaluate(thresholds)
    results.append({"errors": fp + fn, "fp": fp, "fn": fn, "margin": margin,
                    "conditions": len(thresholds), "thresholds": thresholds})

results.sort(key=lambda x: (x["errors"], x["fn"], x["conditions"], -x["margin"]))
best = results[0]

print("=" * 60)
print("Threshold 후보 상위 5개 (Calibration 기준)")
print("=" * 60)
for c in results[:5]:
    text = "  AND  ".join(f"{k} >= {v:.4f}" for k, v in c["thresholds"].items())
    print(f"errors={c['errors']}  FP={c['fp']}  FN={c['fn']}  "
          f"조건 {c['conditions']}개  margin={c['margin']:.3f}  |  {text}")
print()


# --------------------------------------------------
# ⑤ Rule 고정 (Freeze)  (TODO 5)
#
# 여기서 저장한 JSON만 Test에서 읽는다.
# Test 결과를 본 뒤 Threshold를 고치지 않는다.
# --------------------------------------------------

rule = {
    "product": "mini_challenge",
    "require_homography": True,
    "thresholds": {k: round(v, 4) for k, v in best["thresholds"].items()},
    "calibration": {
        "ok_images": len(ok_rows),
        "ng_images": len(ng_rows),
        "errors": best["errors"],
        "fp": best["fp"],
        "fn": best["fn"],
        "margin": round(best["margin"], 4),
    },
    "feature_gap": {k: round(v, 4) for k, v in feature_gap.items()},
}

with open(RULE_PATH, "w", encoding="utf-8") as fp:
    json.dump(rule, fp, ensure_ascii=False, indent=2)

print("Rule 고정:", RULE_PATH)
print("  Homography 성공")
for feature, threshold in rule["thresholds"].items():
    print(f"  AND {feature} >= {threshold}")
print()

del rule   # 아래에서는 반드시 저장된 파일을 다시 읽어서 사용한다.


# --------------------------------------------------
# ⑥ Test에 같은 Rule 적용  (TODO 6)
# --------------------------------------------------

with open(RULE_PATH, encoding="utf-8") as fp:
    frozen_rule = json.load(fp)

test_rows = (
    measure_folder(TEST_DIR / "ok", "OK", reference_data)
    + measure_folder(TEST_DIR / "ng", "NG", reference_data)
)

if not test_rows:
    raise RuntimeError(f"Test 이미지가 없습니다: {TEST_DIR}/ok, {TEST_DIR}/ng")

print("=" * 60)
print("Test 결과 (고정된 Rule 적용)")
print("=" * 60)

for r in test_rows:
    decision, reason = apply_rule(r, frozen_rule)
    r["predicted"] = decision
    r["reason"] = reason
    # 5일차 기준: Positive = NG (불량을 잡아내는 것이 목적)
    if r["actual"] == "NG":
        r["result"] = "TP" if decision == "NG" else "FN"
    else:
        r["result"] = "TN" if decision == "OK" else "FP"
    save_preview(r, decision, reason, OUTPUT_DIR)
    print(f"{r['file']:16} actual={r['actual']}  predicted={decision}  {r['result']}  "
          f"sift={r['sift_good_matches']}  ratio={r['inlier_ratio']:.3f}  {reason}")

save_csv(test_rows, TEST_CSV, extra_columns=("predicted", "result", "reason"))
print()


# --------------------------------------------------
# ⑦ TP / TN / FP / FN  (TODO 7)
#
# 5일차 본 실습(38_day05_evaluate.py)과 같이 Positive Class = NG
#   TP : 실제 NG → NG (불량을 잡음)      FN : 실제 NG → OK (불량을 놓침)
#   TN : 실제 OK → OK                    FP : 실제 OK → NG (정상을 불량으로)
# --------------------------------------------------

counts = {key: sum(r["result"] == key for r in test_rows) for key in ("TP", "TN", "FP", "FN")}
total = len(test_rows)
accuracy = (counts["TP"] + counts["TN"]) / total

print("=" * 60)
print("Test 평가")
print("=" * 60)
print(f"Test 이미지 : {total}장")
print(f"TP={counts['TP']}  TN={counts['TN']}  FP={counts['FP']}  FN={counts['FN']}")
print(f"Accuracy    : {accuracy:.3f}")
ng_precision = counts["TP"] / (counts["TP"] + counts["FP"]) if counts["TP"] + counts["FP"] else 0.0
ng_recall = counts["TP"] / (counts["TP"] + counts["FN"]) if counts["TP"] + counts["FN"] else 0.0
print(f"NG Precision: {ng_precision:.3f}   (NG라고 한 것 중 실제 NG 비율)")
print(f"NG Recall   : {ng_recall:.3f}   (실제 NG 중 잡아낸 비율)")

wrong = [r for r in test_rows if r["result"] in ("FP", "FN")]
if wrong:
    print()
    print("틀린 이미지 (원인을 확인하세요):")
    for r in wrong:
        print(f"  {r['file']}: actual={r['actual']} predicted={r['predicted']} "
              f"{r['reason'] or ''}  sift={r['sift_good_matches']} ratio={r['inlier_ratio']:.3f}")
else:
    print("틀린 이미지: 없음")

print()
print("Calibration CSV :", CALIBRATION_CSV)
print("Test CSV        :", TEST_CSV)
print("Rule JSON       :", RULE_PATH)
print("결과 이미지     :", OUTPUT_DIR)
