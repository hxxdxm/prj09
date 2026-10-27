from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# 1. Challenge 이미지 경로와 결과 폴더를 정한다.
# --------------------------------------------------

image_path = Path(
    "samples/day02/challenge.jpg"
)

output_dir = Path(
    "outputs/day02/sharpen_challenge"
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
# --------------------------------------------------

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY,
)


# --------------------------------------------------
# 4. Sharpening 방법을 준비한다.
#
# kernel  : 14_sharpen.py와 같은 3x3 커널 (강함, Noise도 강조)
# blur+kernel : Noise를 먼저 줄인 뒤 커널 적용
# unsharp : 흐린 이미지를 빼서 경계만 키움 (부드러움)
#           amount가 클수록 경계를 강하게 강조
# --------------------------------------------------

kernel = np.array(
    [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ],
    dtype=np.float32,
)


def unsharp(source, amount):
    soft = cv2.GaussianBlur(source, (0, 0), 3)
    return cv2.addWeighted(source, 1 + amount, soft, -amount, 0)


denoised = cv2.GaussianBlur(gray, (5, 5), 0)

results = {
    "original": gray,
    "kernel": cv2.filter2D(gray, -1, kernel),
    "blur+kernel": cv2.filter2D(denoised, -1, kernel),
    "blur+unsharp0.8": unsharp(denoised, 0.8),
    "blur+unsharp1.5": unsharp(denoised, 1.5),
}


# --------------------------------------------------
# 5. 각 결과에 Otsu를 적용하고
#    "결과 | Otsu"를 한 줄로 붙여 모은다.
#
# 조각 수: Otsu 결과의 흰색 덩어리 개수
#          (작을수록 Noise 점이 적고 깔끔함)
# 선명도 : Laplacian 분산 (클수록 경계가 강함)
# --------------------------------------------------

rows = []

print("방법             | 선명도 | Otsu 임계값 | 조각 수")

for name, result in results.items():

    otsu_value, otsu = cv2.threshold(
        result,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    count, _ = cv2.connectedComponents(otsu)
    sharpness = cv2.Laplacian(result, cv2.CV_64F).var()

    print(
        f"{name:16} | {sharpness:6.1f} | "
        f"{otsu_value:11.1f} | {count - 1:6d}"
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

compare_path = output_dir / "sharpen_compare.jpg"

cv2.imwrite(
    str(compare_path),
    cv2.vconcat(rows),
)

print("저장:", compare_path)


# --------------------------------------------------
# 7. 결과를 비교하며 확인할 점
#
# - 선명도가 커질 때 조각 수(Noise 점)도 같이 늘었는가?
# - Blur를 먼저 한 경우 조각 수가 줄었는가?
# - 병뚜껑 외곽선이 Otsu 결과에서 더 뚜렷해졌는가?
# - 세로 홈 같은 작은 형태가 유지되는가?
# --------------------------------------------------
