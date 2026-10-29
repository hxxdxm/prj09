from pathlib import Path
from time import perf_counter
import csv

import cv2
import numpy as np


# ==================================================
# 32_day04_challenge.py  (Day04 Mini Challenge)
#
# 질문: 실제로 다시 찍은 사진(위치·회전·거리·기울기·반사)에서
#       광천김 로고를 찾으려면 어떤 방법이 좋은가?
#
# 선택한 방법: SIFT_HOMOGRAPHY
#   - 회전·크기 변화에 강한 SIFT 특징점으로 비교하고
#   - RANSAC + Homography로 실제 로고 위치(사각형)까지 찾는다.
#
# 비교를 위해 selected_method를 바꾸면
# TEMPLATE / ORB / SIFT로도 실행할 수 있다.
# ==================================================


# --------------------------------------------------
# 1. 경로
# --------------------------------------------------

reference_path = Path("samples/day04/reference/reference_label.jpg")
challenge_dir = Path("samples/day04/challenge")
output_dir = Path("outputs/day04/challenge")
report_path = Path("reports/day04_challenge_results.csv")

output_dir.mkdir(parents=True, exist_ok=True)
report_path.parent.mkdir(parents=True, exist_ok=True)

challenge_files = [
    ("shift", challenge_dir / "shift.jpg"),            # 위치 이동
    ("rotate", challenge_dir / "rotate.jpg"),          # 회전
    ("distance", challenge_dir / "distance.jpg"),      # 촬영 거리 (크기)
    ("tilt_glare", challenge_dir / "tilt_glare.jpg"),  # 기울기·반사
]


# --------------------------------------------------
# 2. 사용할 방법 선택
#
# TEMPLATE / ORB / SIFT / SIFT_HOMOGRAPHY
# --------------------------------------------------

selected_method = "SIFT_HOMOGRAPHY"

RATIO_TEST = 0.75          # Lowe's ratio test (27~30번과 같은 값)
RANSAC_THRESHOLD = 5.0     # RANSAC 허용 오차 (px)
MIN_GOOD_MATCHES = 10      # Homography를 시도할 최소 Good Match 수


# --------------------------------------------------
# 3. Reference 읽기
# --------------------------------------------------

reference = cv2.imread(str(reference_path))
if reference is None:
    raise FileNotFoundError(reference_path)

reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
ref_h, ref_w = reference_gray.shape[:2]

reference_corners = np.float32([
    [0, 0],
    [ref_w - 1, 0],
    [ref_w - 1, ref_h - 1],
    [0, ref_h - 1],
]).reshape(-1, 1, 2)


# --------------------------------------------------
# 4. 방법별 준비
# --------------------------------------------------

if selected_method in ("SIFT", "SIFT_HOMOGRAPHY"):
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("현재 OpenCV에서 SIFT_create를 사용할 수 없습니다.")
    detector = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
elif selected_method == "ORB":
    detector = cv2.ORB_create(nfeatures=1000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
elif selected_method == "TEMPLATE":
    detector = None
else:
    raise ValueError(f"알 수 없는 방법: {selected_method}")

if detector is not None:
    ref_kp, ref_des = detector.detectAndCompute(reference_gray, None)
    if ref_des is None:
        raise RuntimeError("Reference에서 Descriptor를 만들 수 없습니다.")
    print("Reference Keypoints:", len(ref_kp))


def good_matches_of(target_des):
    """Reference와 Target Descriptor를 비교해 Good Match만 남긴다."""
    good = []
    if target_des is None or len(target_des) < 2:
        return good
    for pair in matcher.knnMatch(ref_des, target_des, k=2):
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < RATIO_TEST * n.distance:
            good.append(m)
    return good


# --------------------------------------------------
# 5. Challenge 이미지를 하나씩 검사
# --------------------------------------------------

rows = []

print(f"method = {selected_method}")

for condition, target_path in challenge_files:

    target = cv2.imread(str(target_path))
    if target is None:
        print("읽기 실패:", target_path)
        continue

    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    t_h, t_w = target_gray.shape[:2]
    preview = target.copy()

    row = {
        "condition": condition,
        "file": target_path.name,
        "method": selected_method,
        "target_size": f"{t_w}x{t_h}",
        "score": "",
        "target_kp": "",
        "good_matches": "",
        "inliers": "",
        "inlier_ratio": "",
        "status": "NOT_FOUND",
        "note": "",
    }

    start = perf_counter()

    # ---------- TEMPLATE ----------
    if selected_method == "TEMPLATE":
        if t_w < ref_w or t_h < ref_h:
            row["status"] = "SKIP"
            row["note"] = "Target이 Template보다 작아 비교 불가"
        else:
            score_map = cv2.matchTemplate(target_gray, reference_gray, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(score_map)
            row["score"] = round(score, 4)
            row["status"] = "CANDIDATE"
            cv2.rectangle(preview, loc, (loc[0] + ref_w, loc[1] + ref_h), (0, 255, 0), 3)

    # ---------- ORB / SIFT / SIFT_HOMOGRAPHY ----------
    else:
        t_kp, t_des = detector.detectAndCompute(target_gray, None)
        good = good_matches_of(t_des)
        row["target_kp"] = len(t_kp)
        row["good_matches"] = len(good)
        row["status"] = "CANDIDATE" if good else "NOT_FOUND"

        if selected_method == "SIFT_HOMOGRAPHY":
            row["status"] = "NOT_FOUND"
            if len(good) >= MIN_GOOD_MATCHES:
                src = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([t_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESHOLD)
                if H is not None and mask is not None:
                    inliers = int(mask.ravel().sum())
                    row["inliers"] = inliers
                    row["inlier_ratio"] = round(inliers / len(good), 3)
                    corners = cv2.perspectiveTransform(reference_corners, H)
                    # 사각형이 볼록하면 정상적인 위치로 본다.
                    if cv2.isContourConvex(np.int32(corners)):
                        row["status"] = "FOUND"
                        cv2.polylines(preview, [np.int32(corners)], True, (0, 255, 0), 3)
                    else:
                        row["note"] = "사각형이 뒤틀림 (잘못된 Homography)"
            else:
                row["note"] = f"Good Match {len(good)}개로 부족"

        # Match 연결선 이미지 (특징점 방법만)
        match_view = cv2.drawMatches(
            reference, ref_kp, target, t_kp, good[:50], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(str(output_dir / f"{condition}_{selected_method.lower()}_matches.jpg"), match_view)

    elapsed = (perf_counter() - start) * 1000
    row["time_ms"] = round(elapsed, 1)

    cv2.putText(preview, f"{condition} {row['status']}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imwrite(str(output_dir / f"{condition}_{selected_method.lower()}.jpg"), preview)

    rows.append(row)
    print(f"{condition:10} size={row['target_size']:9} score={row['score']!s:6} "
          f"kp={row['target_kp']!s:5} good={row['good_matches']!s:4} "
          f"inlier={row['inliers']!s:4} ratio={row['inlier_ratio']!s:5} "
          f"status={row['status']:9} time={row['time_ms']} ms {row['note']}")


# --------------------------------------------------
# 6. CSV 저장
#
# 이번 결과로 OK / NG 기준값은 정하지 않는다. (Day05 Calibration)
# --------------------------------------------------

with open(report_path, "w", newline="", encoding="utf-8-sig") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("CSV  :", report_path)
print("결과 :", output_dir)