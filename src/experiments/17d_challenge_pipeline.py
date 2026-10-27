from pathlib import Path

import cv2


# --------------------------------------------------
# 1. Challenge 이미지 경로와 결과 폴더를 정한다.
# --------------------------------------------------

image_path = Path(
    "samples/day02/challenge.jpg"
)

output_dir = Path(
    "outputs/day02/pipeline_challenge"
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
# 3. 사용할 Parameter를 한곳에 모아 둔다.
#
# 값을 바꿔 가며 다시 실행하고 Parameter Table에 기록한다.
# --------------------------------------------------

BETA = 50            # 밝기: 모든 픽셀에 더할 값
CLIP_LIMIT = 2.0     # CLAHE 대비 강도
TILE_SIZE = (8, 8)   # CLAHE 영역 크기
BLUR_SIZE = (5, 5)   # Sharpening 전 Noise 제거
AMOUNT = 0.8         # Unsharp 경계 강조 강도


# --------------------------------------------------
# 4. Gray로 변환한다.
# --------------------------------------------------

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY,
)


# --------------------------------------------------
# 5. [1단계] 밝기를 올린다.
# --------------------------------------------------

step1 = cv2.convertScaleAbs(
    gray,
    alpha=1.0,
    beta=BETA,
)


# --------------------------------------------------
# 6. [2단계] 밝기를 올린 결과에 CLAHE를 적용한다.
# --------------------------------------------------

clahe = cv2.createCLAHE(
    clipLimit=CLIP_LIMIT,
    tileGridSize=TILE_SIZE,
)

step2 = clahe.apply(step1)


# --------------------------------------------------
# 7. [3단계] CLAHE 결과에 Sharpening을 적용한다.
#
# CLAHE가 키운 Noise가 강조되지 않도록
# Blur로 먼저 정리한 뒤 Unsharp Mask를 적용한다.
# --------------------------------------------------

denoised = cv2.GaussianBlur(
    step2,
    BLUR_SIZE,
    0,
)

soft = cv2.GaussianBlur(
    denoised,
    (0, 0),
    3,
)

step3 = cv2.addWeighted(
    denoised,
    1 + AMOUNT,
    soft,
    -AMOUNT,
    0,
)


# --------------------------------------------------
# 8. 단계별 결과에 Otsu를 적용하고
#    "결과 | Otsu"를 한 줄로 붙여 모은다.
# --------------------------------------------------

stages = {
    "0 original": gray,
    f"1 brightness b{BETA}": step1,
    f"2 + CLAHE c{CLIP_LIMIT}": step2,
    f"3 + sharpen a{AMOUNT}": step3,
}

rows = []

print("단계                 | 평균밝기 | 표준편차 | Otsu 임계값 | 조각 수")

for name, result in stages.items():

    otsu_value, otsu = cv2.threshold(
        result,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    count, _ = cv2.connectedComponents(otsu)

    print(
        f"{name:20} | {result.mean():8.1f} | {result.std():8.1f} | "
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
# 9. 단계별 비교 이미지와 최종 결과를 저장한다.
# --------------------------------------------------

compare_path = output_dir / "pipeline_compare.jpg"
result_path = output_dir / "pipeline_result.jpg"

cv2.imwrite(str(compare_path), cv2.vconcat(rows))
cv2.imwrite(str(result_path), step3)

print("비교 이미지:", compare_path)
print("최종 결과  :", result_path)


# --------------------------------------------------
# 10. 결과를 비교하며 확인할 점
#
# - 단계를 하나 추가할 때마다 Otsu 결과가 좋아졌는가?
# - 오히려 나빠진 단계가 있다면 그 단계는 빼는 것이 맞다.
# - BETA를 0으로 바꿔도 결과가 같다면
#   밝기 조절은 이 파이프라인에 필요 없는 단계다.
# --------------------------------------------------
