from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


image_path = Path(
    "samples/day04/source/grabcut_scene.jpg"
)

output_dir = Path(
    "outputs/day04/grabcut"
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
# 2. GrabCut 처리용으로 이미지 축소
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


cv2.imwrite(
    str(
        output_dir /
        "resized_input.jpg"
    ),
    image,
)


# --------------------------------------------------
# 3. Rectangle 크기를 다르게 준비
#
# wide
# → 제품보다 넉넉하게
#
# medium
# → 제품에 조금 더 가깝게
#
# tight
# → 일부가 잘릴 가능성이 있는 비교용
# --------------------------------------------------

margin_ratios = [
    ("wide", 0.03),
    ("medium", 0.07),
    ("tight", 0.12),
]


for name, margin_ratio in margin_ratios:

    margin_x = int(
        width * margin_ratio
    )

    margin_y = int(
        height * margin_ratio
    )


    rect_x = margin_x
    rect_y = margin_y

    rect_w = (
        width
        - 2 * margin_x
    )

    rect_h = (
        height
        - 2 * margin_y
    )


    if rect_w <= 0 or rect_h <= 0:
        continue


    rectangle = (
        rect_x,
        rect_y,
        rect_w,
        rect_h,
    )


    # --------------------------------------------------
    # 4. GrabCut용 Mask와 내부 모델 준비
    # --------------------------------------------------

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    bg_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    fg_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )


    # --------------------------------------------------
    # 5. GrabCut 실행
    # --------------------------------------------------

    start = perf_counter()


    cv2.grabCut(
        image,
        mask,
        rectangle,
        bg_model,
        fg_model,
        3,
        cv2.GC_INIT_WITH_RECT,
    )


    elapsed_ms = (
        perf_counter()
        - start
    ) * 1000


    # --------------------------------------------------
    # 6. GrabCut 결과를 흑백 Mask로 변환
    #
    # 확실한 전경
    # +
    # 전경일 가능성이 높은 영역
    # → 흰색
    #
    # 나머지
    # → 검은색
    # --------------------------------------------------

    raw_mask = np.where(
        (mask == cv2.GC_FGD)
        |
        (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(
        np.uint8
    )


    # --------------------------------------------------
    # 7. 가장 큰 외곽 Contour를 찾음
    #
    # 광천김 포장지 안쪽의
    # 검은 구멍과 작은 Noise를 정리하기 위함
    # --------------------------------------------------

    contours, _ = cv2.findContours(
        raw_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )


    clean_mask = np.zeros_like(
        raw_mask
    )


    if contours:

        largest_contour = max(
            contours,
            key=cv2.contourArea,
        )

        cv2.drawContours(
            clean_mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED,
        )


    # --------------------------------------------------
    # 8. Clean Mask를 원본에 적용
    # --------------------------------------------------

    foreground = cv2.bitwise_and(
        image,
        image,
        mask=clean_mask,
    )


    # --------------------------------------------------
    # 9. 초기 Rectangle 확인용 이미지
    # --------------------------------------------------

    preview = image.copy()

    cv2.rectangle(
        preview,
        (rect_x, rect_y),
        (
            rect_x + rect_w,
            rect_y + rect_h,
        ),
        (0, 255, 255),
        3,
    )


    # --------------------------------------------------
    # 10. 결과 저장
    # --------------------------------------------------

    cv2.imwrite(
        str(
            output_dir /
            f"{name}_rect.jpg"
        ),
        preview,
    )


    cv2.imwrite(
        str(
            output_dir /
            f"{name}_raw_mask.png"
        ),
        raw_mask,
    )


    cv2.imwrite(
        str(
            output_dir /
            f"{name}_clean_mask.png"
        ),
        clean_mask,
    )


    cv2.imwrite(
        str(
            output_dir /
            f"{name}_foreground.jpg"
        ),
        foreground,
    )


    foreground_ratio = (
        cv2.countNonZero(
            clean_mask
        )
        / clean_mask.size
    )


    print(
        f"{name:6s}",
        f"foreground_ratio={foreground_ratio:.4f}",
        f"time={elapsed_ms:.2f} ms",
    )