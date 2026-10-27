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
    "outputs/day02/outline_challenge"
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
# 3. Parameter를 한곳에 모아 둔다.
# --------------------------------------------------

BLUR_SIZE = (21, 21)     # S 채널의 JPEG 블록 Noise 제거
THRESHOLD = 11           # S 값이 이보다 크면 병뚜껑으로 판단 (9~13 비교)
KERNEL_SIZE = (15, 15)   # Morphology 구멍 메우기 크기


# --------------------------------------------------
# 4. [핵심] Gray 대신 HSV의 S(채도) 채널을 사용한다.
#
# Gray에서는 병뚜껑(약 129)이 배경(약 134~143)과 밝기가 비슷하다.
# 하지만 병뚜껑은 살짝 푸른빛, 배경은 회색이라
# S(채도) 값은 병뚜껑 약 17, 배경 약 5~9로 차이가 난다.
# --------------------------------------------------

hsv = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2HSV,
)

saturation = hsv[:, :, 1]


# --------------------------------------------------
# 5. S 채널을 Blur로 부드럽게 만든다.
#
# S 채널은 JPEG 압축 때문에 네모난 블록 Noise가 많다.
# --------------------------------------------------

blurred = cv2.GaussianBlur(
    saturation,
    BLUR_SIZE,
    0,
)


# --------------------------------------------------
# 6. 고정 Threshold로 병뚜껑(채도 높음)을 흰색 객체로 분리한다.
#
# Otsu는 이 이미지에서 임계값을 너무 높게 잡아
# 병뚜껑 일부만 남기므로 고정값을 직접 비교해 정한다.
# --------------------------------------------------

_, mask = cv2.threshold(
    blurred,
    THRESHOLD,
    255,
    cv2.THRESH_BINARY,
)


# --------------------------------------------------
# 7. Morphology로 작은 구멍을 메우고 작은 점을 지운다.
#
# CLOSE : 객체 안쪽의 구멍 메우기
# OPEN  : 객체 바깥의 작은 점 지우기
# (3일차에서 자세히 배우는 내용)
# --------------------------------------------------

kernel = np.ones(
    KERNEL_SIZE,
    np.uint8,
)

mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


# --------------------------------------------------
# 8. 윤곽선을 찾고 가장 큰 윤곽선(병뚜껑)을 원본에 그린다.
# --------------------------------------------------

contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

outline = image.copy()

if contours:

    largest = max(contours, key=cv2.contourArea)

    cv2.drawContours(
        outline,
        [largest],
        -1,
        (0, 0, 255),
        4,
    )

    print("윤곽선 개수  :", len(contours))
    print("가장 큰 면적 :", int(cv2.contourArea(largest)))

else:
    print("윤곽선을 찾지 못했습니다.")

print("Threshold    :", THRESHOLD)


# --------------------------------------------------
# 9. 단계별 결과를 한 장으로 붙여 저장한다.
#
# 원본 | S 채널 | Mask | 윤곽선
# --------------------------------------------------

def to_bgr(source):
    if len(source.shape) == 2:
        return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    return source


# S 값은 0~44 정도로 작아서 보기 쉽게 늘려서 표시만 한다.
saturation_view = cv2.normalize(
    saturation, None, 0, 255, cv2.NORM_MINMAX,
)

views = [
    ("original", image),
    ("S channel", saturation_view),
    ("mask", mask),
    ("outline", outline),
]

tiles = []

for name, view in views:

    tile = cv2.resize(to_bgr(view), None, fx=0.4, fy=0.4)

    cv2.putText(
        tile, name, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
    )

    tiles.append(tile)

compare_path = output_dir / "outline_compare.jpg"

cv2.imwrite(str(compare_path), cv2.hconcat(tiles))
cv2.imwrite(str(output_dir / "outline_result.jpg"), outline)

print("저장:", compare_path)