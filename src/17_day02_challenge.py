from pathlib import Path

import cv2


image_path = Path(
    "samples/day02/challenge.jpg"
)

output_dir = Path(
    "outputs/day02/challenge"
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
# TODO
#
# 1. 이미지의 문제를 판단합니다.
# 2. 필요한 전처리를 선택합니다.
# 3. Parameter를 직접 정합니다.
# 4. result에 최종 결과를 저장합니다.
# --------------------------------------------------


result = image.copy()


# --------------------------------------------------
# 전처리 결과 저장
# --------------------------------------------------

result_path = (
    output_dir
    / "challenge_result.jpg"
)

cv2.imwrite(
    str(result_path),
    result,
)


# --------------------------------------------------
# Gray 변환
# --------------------------------------------------

def to_gray(source):

    if len(source.shape) == 2:
        return source

    return cv2.cvtColor(
        source,
        cv2.COLOR_BGR2GRAY,
    )


original_gray = to_gray(
    image
)

result_gray = to_gray(
    result
)


# --------------------------------------------------
# 전처리 전 Otsu
#
# 이번 Challenge에서는
# 밝은 병뚜껑을 흰색 객체로 확인합니다.
# --------------------------------------------------

_, original_otsu = cv2.threshold(
    original_gray,
    0,
    255,
    cv2.THRESH_BINARY
    + cv2.THRESH_OTSU,
)


# --------------------------------------------------
# 전처리 후 Otsu
# --------------------------------------------------

_, result_otsu = cv2.threshold(
    result_gray,
    0,
    255,
    cv2.THRESH_BINARY
    + cv2.THRESH_OTSU,
)


cv2.imwrite(
    str(
        output_dir
        / "challenge_original_otsu.jpg"
    ),
    original_otsu,
)

cv2.imwrite(
    str(
        output_dir
        / "challenge_result_otsu.jpg"
    ),
    result_otsu,
)


print("입력 :", image_path)
print("결과 :", result_path)
print("Challenge 완료")