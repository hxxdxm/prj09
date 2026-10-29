from pathlib import Path
from time import perf_counter
import csv

import cv2
import numpy as np


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
    "outputs/day04/homography"
)

report_path = Path(
    "reports/day04_homography_results.csv"
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
# 3. SIFT 사용 가능 여부 확인
# --------------------------------------------------

if not hasattr(
    cv2,
    "SIFT_create",
):
    raise RuntimeError(
        "현재 OpenCV에서 SIFT_create를 사용할 수 없습니다."
    )


# --------------------------------------------------
# 4. 광천김 Reference 읽기
# --------------------------------------------------

reference = cv2.imread(
    str(reference_path)
)

if reference is None:
    raise FileNotFoundError(
        reference_path
    )


reference_gray = cv2.cvtColor(
    reference,
    cv2.COLOR_BGR2GRAY,
)


# --------------------------------------------------
# 5. SIFT 준비
# --------------------------------------------------

sift = cv2.SIFT_create()


# --------------------------------------------------
# 6. Reference에서
# Keypoint와 Descriptor 만들기
# --------------------------------------------------

reference_keypoints, reference_descriptors = (
    sift.detectAndCompute(
        reference_gray,
        None,
    )
)


if reference_descriptors is None:
    raise RuntimeError(
        "Reference에서 SIFT Descriptor를 만들 수 없습니다."
    )


print(
    "Reference Keypoints:",
    len(reference_keypoints),
)


# --------------------------------------------------
# 7. SIFT Descriptor 비교용 Matcher 준비
# --------------------------------------------------

matcher = cv2.BFMatcher(
    cv2.NORM_L2,
    crossCheck=False,
)


# --------------------------------------------------
# 8. Reference 이미지의 네 모서리 준비
#
# Homography를 계산한 뒤
# 이 네 점을 Target 위치로 옮길 예정
# --------------------------------------------------

reference_h, reference_w = (
    reference_gray.shape[:2]
)


reference_corners = np.float32([
    [0, 0],
    [reference_w - 1, 0],
    [reference_w - 1, reference_h - 1],
    [0, reference_h - 1],
]).reshape(
    -1,
    1,
    2,
)


rows = []


# --------------------------------------------------
# 9. 네 개의 Target에 같은 과정 적용
# --------------------------------------------------

for condition, target_path in target_settings:

    target = cv2.imread(
        str(target_path)
    )

    if target is None:
        raise FileNotFoundError(
            target_path
        )


    target_gray = cv2.cvtColor(
        target,
        cv2.COLOR_BGR2GRAY,
    )


    start = perf_counter()


    # --------------------------------------------------
    # 10. Target에서도
    # SIFT Keypoint와 Descriptor 만들기
    # --------------------------------------------------

    target_keypoints, target_descriptors = (
        sift.detectAndCompute(
            target_gray,
            None,
        )
    )


    good_matches = []

    inlier_count = 0
    inlier_ratio = 0.0

    status = "NOT_FOUND"

    preview = target.copy()


    # --------------------------------------------------
    # 11. Reference와 Target의
    # Descriptor 비교
    # --------------------------------------------------

    if target_descriptors is not None:

        match_candidates = matcher.knnMatch(
            reference_descriptors,
            target_descriptors,
            k=2,
        )


        # --------------------------------------------------
        # 12. Ratio Test
        #
        # 애매한 Match는 줄이고
        # 비교적 확실한 Good Match만 남김
        # --------------------------------------------------

        for candidate in match_candidates:

            if len(candidate) < 2:
                continue


            m, n = candidate


            if m.distance < 0.75 * n.distance:

                good_matches.append(
                    m
                )


    # --------------------------------------------------
    # 13. Homography는 최소 4개의
    # 대응점이 필요함
    # --------------------------------------------------

    if len(good_matches) >= 4:

        # Reference에서 Good Match의 좌표
        reference_points = np.float32([
            reference_keypoints[
                match.queryIdx
            ].pt
            for match in good_matches
        ]).reshape(
            -1,
            1,
            2,
        )


        # Target에서 Good Match의 좌표
        target_points = np.float32([
            target_keypoints[
                match.trainIdx
            ].pt
            for match in good_matches
        ]).reshape(
            -1,
            1,
            2,
        )


        # --------------------------------------------------
        # 14. RANSAC + Homography
        #
        # Good Match 중에서
        # 같은 위치 관계를 잘 따르는 점을
        # Inlier로 선택하면서
        # Homography를 계산
        # --------------------------------------------------

        homography, inlier_mask = (
            cv2.findHomography(
                reference_points,
                target_points,
                cv2.RANSAC,
                5.0,
            )
        )


        if (
            homography is not None
            and inlier_mask is not None
        ):

            # --------------------------------------------------
            # 15. Inlier 수와 비율 계산
            # --------------------------------------------------

            inlier_count = int(
                inlier_mask.ravel().sum()
            )


            inlier_ratio = (
                inlier_count
                / len(good_matches)
            )


            # --------------------------------------------------
            # 16. Reference의 네 모서리를
            # Target 위치로 이동
            # --------------------------------------------------

            projected_corners = (
                cv2.perspectiveTransform(
                    reference_corners,
                    homography,
                )
            )


            # --------------------------------------------------
            # 17. 계산된 라벨 위치를
            # 초록색 사각형으로 표시
            # --------------------------------------------------

            polygon = np.int32(
                projected_corners
            )


            cv2.polylines(
                preview,
                [polygon],
                True,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )


            status = "FOUND"


    elapsed_ms = (
        perf_counter() - start
    ) * 1000


    # --------------------------------------------------
    # 18. 측정 결과를 이미지에 표시
    # --------------------------------------------------

    cv2.putText(
        preview,
        (
            f"good={len(good_matches)} "
            f"inlier={inlier_count} "
            f"ratio={inlier_ratio:.2f}"
        ),
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


    # --------------------------------------------------
    # 19. 결과 이미지 저장
    # --------------------------------------------------

    output_path = (
        output_dir
        / f"{target_path.stem}_homography.jpg"
    )


    cv2.imwrite(
        str(output_path),
        preview,
    )


    # --------------------------------------------------
    # 20. CSV에 측정값 기록
    # --------------------------------------------------

    rows.append({
        "condition": condition,
        "filename": target_path.name,
        "good_matches": len(
            good_matches
        ),
        "inliers": inlier_count,
        "inlier_ratio": f"{inlier_ratio:.6f}",
        "status": status,
        "time_ms": f"{elapsed_ms:.3f}",
    })


    print(
        f"{condition:7s}",
        f"good={len(good_matches)}",
        f"inlier={inlier_count}",
        f"ratio={inlier_ratio:.3f}",
        f"status={status}",
        f"time={elapsed_ms:.2f} ms",
    )


# --------------------------------------------------
# 21. CSV 저장
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
            "good_matches",
            "inliers",
            "inlier_ratio",
            "status",
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