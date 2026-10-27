from pathlib import Path

import cv2


# --------------------------------------------------
# 1. Challenge 이미지 경로와 결과 폴더를 정한다.
# --------------------------------------------------

image_path = Path(
    "samples/day02/challenge.jpg"
)

output_dir = Path(
    "outputs/day02/contrast_challenge"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 2. 이미지를 읽고 정상적으로 읽혔는지 확인한다.
# --------------------------------------------------

image = cv2.imread(
    str(image_path)
)

if image is None:
    raise FileNotFoundError(
        f"이미지를 읽지 못했습니다: {image_path}"
    )


# --------------------------------------------------
# 3. Gray로 변환한다.
#
# Contrast와 CLAHE 모두 밝기 정보만 사용해 비교한다.
# --------------------------------------------------

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY,
)


# --------------------------------------------------
# 4. 비교할 방법을 준비한다.
#
# Contrast : alpha를 키우고 beta로 평균 밝기를 유지
#            (평균 124 근처 유지 → beta = 124 * (1 - alpha))
# CLAHE    : clipLimit이 클수록 대비를 강하게 키움
#            tileGridSize는 (8, 8) 고정
# --------------------------------------------------

results = {
    "original": gray,
}

for alpha in [1.5, 2.0]:

    beta = int(124 * (1 - alpha))

    results[f"contrast_a{alpha}"] = cv2.convertScaleAbs(
        gray,
        alpha=alpha,
        beta=beta,
    )

for clip in [1.0, 2.0, 4.0]:

    clahe = cv2.createCLAHE(
        clipLimit=clip,
        tileGridSize=(8, 8),
    )

    results[f"clahe_c{clip}"] = clahe.apply(gray)


# --------------------------------------------------
# 5. 각 결과에 Otsu를 적용하고
#    "결과 | Otsu"를 한 줄로 붙여 모은다.
# --------------------------------------------------

rows = []

print("방법           | 표준편차 | Otsu 임계값 | 흰색비율")

for name, result in results.items():

    # 밝은 병뚜껑을 흰색 객체로 보기 위해 THRESH_BINARY 사용
    otsu_value, otsu = cv2.threshold(
        result,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    white_ratio = (otsu > 0).mean() * 100

    print(
        f"{name:14} | {result.std():8.1f} | "
        f"{otsu_value:11.1f} | {white_ratio:6.1f}%"
    )

    row = cv2.hconcat([result, otsu])
    row = cv2.cvtColor(row, cv2.COLOR_GRAY2BGR)
    row = cv2.resize(row, None, fx=0.4, fy=0.4)

    cv2.putText(
        row, name, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
    )

    rows.append(row)


# --------------------------------------------------
# 6. 모든 결과를 세로로 쌓아 한 장으로 저장한다.
# --------------------------------------------------

compare_path = output_dir / "contrast_compare.jpg"

cv2.imwrite(
    str(compare_path),
    cv2.vconcat(rows),
)

print("저장:", compare_path)


# --------------------------------------------------
# 7. 결과를 비교하며 확인할 점
#
# - 표준편차가 커졌을 때 Otsu 결과도 실제로 달라졌는가?
# - CLAHE에서 병뚜껑 표면 질감·벽 무늬까지 강조되지 않았는가?
# - clipLimit을 키울수록 점 Noise가 늘어나지 않았는가?
# - 병뚜껑과 배경이 서로 다른 색으로 분리되었는가?
# --------------------------------------------------
