from pathlib import Path
from time import perf_counter

import cv2


image_path = Path(
    "samples/day04/targets/target_normal.jpg"
)

output_dir = Path(
    "outputs/day04/meanshift"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 1. 이미지 읽기
# --------------------------------------------------

image = cv2.imread(
    str(image_path)
)

if image is None:
    raise FileNotFoundError(
        image_path
    )


original_h, original_w = image.shape[:2]

print(
    "Original size:",
    original_w,
    "x",
    original_h,
)


# --------------------------------------------------
# 2. MeanShift 처리용 이미지 축소
# --------------------------------------------------

process_width = 900

if original_w > process_width:

    scale = (
        process_width
        / original_w
    )

    process_height = int(
        original_h * scale
    )

    image = cv2.resize(
        image,
        (
            process_width,
            process_height,
        ),
        interpolation=cv2.INTER_AREA,
    )


height, width = image.shape[:2]

print(
    "Process size :",
    width,
    "x",
    height,
)


# 비교할 기준 이미지를 저장
cv2.imwrite(
    str(
        output_dir /
        "resized_input.jpg"
    ),
    image,
)


# --------------------------------------------------
# 3. MeanShift Parameter 준비
#
# sp
# → 주변 위치를 얼마나 넓게 볼 것인가?
#
# sr
# → 어느 정도 색상 차이까지
#   비슷한 색으로 묶을 것인가?
# --------------------------------------------------

settings = [
    (10, 20),
    (20, 30),
    (30, 50),
    (50, 80),
]


# --------------------------------------------------
# 4. 각 Parameter를 같은 이미지에 적용
# --------------------------------------------------

for sp, sr in settings:

    start = perf_counter()


    shifted = cv2.pyrMeanShiftFiltering(
        image,
        sp=sp,
        sr=sr,
    )


    elapsed_ms = (
        perf_counter()
        - start
    ) * 1000


    output_path = (
        output_dir /
        f"meanshift_sp{sp}_sr{sr}.jpg"
    )


    cv2.imwrite(
        str(output_path),
        shifted,
    )


    print(
        f"sp={sp:02d} "
        f"sr={sr:02d} "
        f"time={elapsed_ms:.2f} ms"
    )


print()
print(
    "MeanShift 결과 저장:",
    output_dir
)