from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# 1. 이미지 경로와 결과 폴더를 정한다.
# --------------------------------------------------

image_path = Path(
    "samples/day02/challenge.jpg"
)

output_dir = Path(
    "outputs/day02/s_contrast"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

image = cv2.imread(
    str(image_path)
)

if image is None:
    raise FileNotFoundError(
        f"이미지를 읽지 못했습니다: {image_path}"
    )


# --------------------------------------------------
# 2. 3번 사진과 같은 S 채널 Blur 이미지를 만든다.
# --------------------------------------------------

s = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]

blurred = cv2.GaussianBlur(
    s,
    (21, 21),
    0,
)


# --------------------------------------------------
# 3. 대비를 넣는 세 가지 방법
#
# (A) 단순 늘리기 : 최솟값~최댓값을 0~255로 늘림
# (B) 기준 늘리기 : 배경(S=5)은 0, 병뚜껑(S=17)은 255가 되도록
#                   LOW~HIGH 구간만 0~255로 늘림 (가장 강함)
# (C) CLAHE      : 영역별로 대비를 키움
# --------------------------------------------------

LOW = 6     # 이 값 이하는 검정 (배경 S 약 5)
HIGH = 16   # 이 값 이상은 흰색 (병뚜껑 S 약 17)

stretch_a = cv2.normalize(
    blurred,
    None,
    0,
    255,
    cv2.NORM_MINMAX,
)

alpha = 255 / (HIGH - LOW)
beta = -LOW * alpha

stretch_b = cv2.convertScaleAbs(
    blurred,
    alpha=alpha,
    beta=beta,
)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8),
)

stretch_c = clahe.apply(stretch_a)


# --------------------------------------------------
# 4. 결과마다 병뚜껑·배경·유리병 값을 비교한다.
# --------------------------------------------------

height, width = blurred.shape


def region_mean(channel, box):
    x1, y1, x2, y2 = box
    return channel[
        int(y1 * height):int(y2 * height),
        int(x1 * width):int(x2 * width),
    ].mean()


cap_box = (0.25, 0.22, 0.70, 0.38)
bg_box = (0.00, 0.05, 0.06, 0.22)
glass_box = (0.25, 0.85, 0.70, 0.98)

results = {
    "3 blur (S)": blurred,
    "A normalize": stretch_a,
    f"B stretch {LOW}-{HIGH}": stretch_b,
    "C CLAHE": stretch_c,
}

print("방법             | 병뚜껑 | 배경  | 유리병 | 병뚜껑-배경")

tiles = []

for name, result in results.items():

    cap = region_mean(result, cap_box)
    bg = region_mean(result, bg_box)
    glass = region_mean(result, glass_box)

    print(
        f"{name:16} | {cap:6.1f} | {bg:5.1f} | {glass:6.1f} | {cap - bg:+8.1f}"
    )

    tile = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    tile = cv2.resize(tile, None, fx=0.4, fy=0.4)

    cv2.putText(
        tile, name, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
    )

    tiles.append(tile)


# --------------------------------------------------
# 5. 한 장으로 저장한다.
# --------------------------------------------------

compare_path = output_dir / "s_contrast_compare.jpg"

cv2.imwrite(str(compare_path), cv2.hconcat(tiles))
cv2.imwrite(str(output_dir / "s_stretch.jpg"), stretch_b)

print("저장:", compare_path)
