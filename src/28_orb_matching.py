from pathlib import Path
from time import perf_counter
import csv

import cv2


# --------------------------------------------------
# 1. 경로 설정
# --------------------------------------------------

reference_path = Path(
    "samples/day04/reference/reference_label.jpg"
)

target_dir = Path(
    "samples/day04/targets"
)

output_dir = Path(
    "outputs/day04/orb"
)

report_path = Path(
    "reports/day04_orb_results.csv"
)


output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

report_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 2. 비교할 Target을 순서대로 준비
# --------------------------------------------------

target_settings = [
    (
        "normal",
        target_dir / "target_normal.jpg",
    ),
    (
        "shift",
        target_dir / "target_shift.jpg",
    ),
    (
        "rot15",
        target_dir / "target_rot15.jpg",
    ),
    (
        "scale90",
        target_dir / "target_scale90.jpg",
    ),
]


# --------------------------------------------------
# 3. 광천김 Reference 읽기
# --------------------------------------------------

reference = cv2.imread(
    str(reference_path),
    cv2.IMREAD_GRAYSCALE,
)

if reference is None:
    raise FileNotFoundError(
        reference_path
    )


# --------------------------------------------------
# 4. ORB 준비
#
# 이미지에서 최대 1000개의
# 특징점을 찾도록 설정
# --------------------------------------------------

orb = cv2.ORB_create(
    nfeatures=1000
)


# --------------------------------------------------
# 5. Reference에서
# Keypoint와 Descriptor 만들기
# --------------------------------------------------

reference_keypoints, reference_descriptors = (
    orb.detectAndCompute(
        reference,
        None,
    )
)


if reference_descriptors is None:
    raise RuntimeError(
        "Reference에서 ORB Descriptor를 만들 수 없습니다."
    )


print(
    "Reference Keypoints:",
    len(reference_keypoints),
)


# --------------------------------------------------
# 6. Descriptor를 비교할 Matcher 준비
#
# ORB Descriptor는
# Hamming Distance로 비교
# --------------------------------------------------

matcher = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False,
)


rows = []


# --------------------------------------------------
# 7. 네 개의 Target에 같은 ORB 적용
# --------------------------------------------------

for condition, target_path in target_settings:

    target_color = cv2.imread(
        str(target_path)
    )

    if target_color is None:
        raise FileNotFoundError(
            target_path
        )


    target_gray = cv2.cvtColor(
        target_color,
        cv2.COLOR_BGR2GRAY,
    )


    start = perf_counter()


    # --------------------------------------------------
    # 8. Target에서도
    # Keypoint와 Descriptor 만들기
    # --------------------------------------------------

    target_keypoints, target_descriptors = (
        orb.detectAndCompute(
            target_gray,
            None,
        )
    )


    good_matches = []


    # --------------------------------------------------
    # 9. Reference와 Target의
    # Descriptor 비교
    # --------------------------------------------------

    if target_descriptors is not None:

        match_candidates = matcher.knnMatch(
            reference_descriptors,
            target_descriptors,
            k=2,
        )


        # --------------------------------------------------
        # 10. Ratio Test
        #
        # 여러 Match 후보 중에서
        # 비교적 확실한 연결만
        # Good Match로 남김
        # --------------------------------------------------

        for candidate in match_candidates:

            if len(candidate) < 2:
                continue


            m, n = candidate


            if m.distance < 0.75 * n.distance:

                good_matches.append(
                    m
                )


    elapsed_ms = (
        perf_counter() - start
    ) * 1000


    # --------------------------------------------------
    # 11. 더 가까운 Match부터 정렬
    # --------------------------------------------------

    good_matches = sorted(
        good_matches,
        key=lambda match: match.distance,
    )


    # 화면이 너무 복잡해지지 않도록
    # 상위 30개까지만 선으로 표시
    shown_matches = good_matches[:30]


    # --------------------------------------------------
    # 12. Reference와 Target 사이의
    # Good Match를 선으로 연결
    # --------------------------------------------------

    preview = cv2.drawMatches(
        reference,
        reference_keypoints,
        target_color,
        target_keypoints,
        shown_matches,
        None,
        flags=(
            cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        ),
    )


    # --------------------------------------------------
    # 13. 결과 이미지 저장
    # --------------------------------------------------

    output_path = (
        output_dir
        / f"{target_path.stem}_orb.jpg"
    )


    cv2.imwrite(
        str(output_path),
        preview,
    )


    # --------------------------------------------------
    # 14. 측정값 저장
    # --------------------------------------------------

    rows.append({
        "condition": condition,
        "filename": target_path.name,
        "reference_keypoints": len(
            reference_keypoints
        ),
        "target_keypoints": len(
            target_keypoints
        ),
        "good_matches": len(
            good_matches
        ),
        "time_ms": f"{elapsed_ms:.3f}",
    })


    print(
        f"{condition:7s}",
        f"reference_kp={len(reference_keypoints)}",
        f"target_kp={len(target_keypoints)}",
        f"good={len(good_matches)}",
        f"time={elapsed_ms:.2f} ms",
    )


# --------------------------------------------------
# 15. CSV 저장
# --------------------------------------------------

with report_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "condition",
            "filename",
            "reference_keypoints",
            "target_keypoints",
            "good_matches",
            "time_ms",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)


print()
print(
    "CSV 저장:",
    report_path,
)

print(
    "결과 이미지:",
    output_dir,
)