from pathlib import Path

import cv2
import numpy as np


source_path = Path(
    "samples/day04/source/full_front.jpg"
)

reference_dir = Path(
    "samples/day04/reference"
)

target_dir = Path(
    "samples/day04/targets"
)

reference_dir.mkdir(parents=True, exist_ok=True)
target_dir.mkdir(parents=True, exist_ok=True)


image = cv2.imread(str(source_path))

if image is None:
    raise FileNotFoundError(source_path)

height, width = image.shape[:2]


# 자신의 이미지에서
# KC 로고 + (주)광천김 + 영문 영역에 맞게 수정합니다.
x1 = 180
y1 = 120
x2 = 420
y2 = 280


x1 = max(0, min(x1, width))
x2 = max(0, min(x2, width))
y1 = max(0, min(y1, height))
y2 = max(0, min(y2, height))

if x1 >= x2 or y1 >= y2:
    raise ValueError(
        f"잘못된 Template 좌표: {(x1, y1, x2, y2)}"
    )


# --------------------------------------------------
# 1. Reference 이미지 만들기
# --------------------------------------------------

reference = image[y1:y2, x1:x2]

reference_path = reference_dir / "reference_label.jpg"
cv2.imwrite(str(reference_path), reference)


# --------------------------------------------------
# 2. Preview 이미지 만들기
# 원본에서 Template 위치를 확인하기 위한 이미지
# --------------------------------------------------

preview = image.copy()

cv2.rectangle(
    preview,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    3,
)

cv2.imwrite(
    str(reference_dir / "reference_preview.jpg"),
    preview,
)


# --------------------------------------------------
# 3. 정상 Target
# --------------------------------------------------

cv2.imwrite(
    str(target_dir / "target_normal.jpg"),
    image,
)


# --------------------------------------------------
# 4. 위치 이동 Target
# 오른쪽 35px, 아래쪽 20px 이동
# --------------------------------------------------

shift_x = 35
shift_y = 20

move_matrix = np.float32([
    [1, 0, shift_x],
    [0, 1, shift_y],
])

shifted = cv2.warpAffine(
    image,
    move_matrix,
    (width, height),
    borderMode=cv2.BORDER_REPLICATE,
)

cv2.imwrite(
    str(target_dir / "target_shift.jpg"),
    shifted,
)


# --------------------------------------------------
# 5. 회전 Target
# 이미지 중심 기준 15도 회전
# --------------------------------------------------

center = (width / 2, height / 2)

rotation_matrix = cv2.getRotationMatrix2D(
    center,
    15,
    1.0,
)

rotated = cv2.warpAffine(
    image,
    rotation_matrix,
    (width, height),
    borderMode=cv2.BORDER_REPLICATE,
)

cv2.imwrite(
    str(target_dir / "target_rot15.jpg"),
    rotated,
)


# --------------------------------------------------
# 6. 크기 변화 Target
# 이미지 중심 기준 90% 크기로 축소
# --------------------------------------------------

scale_matrix = cv2.getRotationMatrix2D(
    center,
    0,
    0.90,
)

scaled = cv2.warpAffine(
    image,
    scale_matrix,
    (width, height),
    borderMode=cv2.BORDER_REPLICATE,
)

cv2.imwrite(
    str(target_dir / "target_scale90.jpg"),
    scaled,
)


print("Source :", source_path)
print("Template:", reference_path)
print("Template 좌표:", (x1, y1, x2, y2))
print("Template 크기:", reference.shape)
print("Day04 Target 생성 완료")