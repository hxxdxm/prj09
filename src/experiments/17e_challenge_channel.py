from pathlib import Path

import cv2


# --------------------------------------------------
# 1. Challenge 이미지 경로와 결과 폴더를 정한다.
# --------------------------------------------------

image_path = Path(
    "samples/day02/challenge.jpg"
)

output_dir = Path(
    "outputs/day02/step2_channel"
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

height, width = image.shape[:2]

print("이미지 크기:", width, "x", height)


# --------------------------------------------------
# 3. Gray와 HSV로 변환한다.
#
# H (Hue)        : 색의 종류 (빨강, 파랑 ...)
# S (Saturation) : 색이 얼마나 진한가 (회색일수록 0)
# V (Value)      : 밝기
# --------------------------------------------------

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY,
)

hsv = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2HSV,
)

h, s, v = cv2.split(hsv)


# --------------------------------------------------
# 4. 병뚜껑과 배경에서 작은 영역을 하나씩 골라
#    채널별 평균값을 비교한다.
#
# 영역은 비율로 정해서 이미지 크기가 달라도 동작하게 한다.
# 결과 이미지에 표시되는 사각형을 보고
# 병뚜껑 / 배경 위에 제대로 올라가 있는지 확인한다.
# 위치가 맞지 않으면 아래 비율을 조정한다.
#
# (x 시작, y 시작, x 끝, y 끝) → 이미지 너비·높이에 대한 비율
# --------------------------------------------------

cap_box = (0.25, 0.22, 0.70, 0.38)   # 병뚜껑 윗면
bg_box = (0.00, 0.05, 0.06, 0.22)    # 왼쪽 위 배경 벽


def to_pixels(box):
    x1, y1, x2, y2 = box
    return (
        int(x1 * width),
        int(y1 * height),
        int(x2 * width),
        int(y2 * height),
    )


def region_mean(channel, box):
    x1, y1, x2, y2 = to_pixels(box)
    return channel[y1:y2, x1:x2].mean()


channels = {
    "Gray": gray,
    "H": h,
    "S": s,
    "V": v,
}

print()
print("채널 | 병뚜껑 | 배경  | 차이")

for name, channel in channels.items():

    cap_mean = region_mean(channel, cap_box)
    bg_mean = region_mean(channel, bg_box)

    print(
        f"{name:4} | {cap_mean:6.1f} | {bg_mean:5.1f} | "
        f"{cap_mean - bg_mean:+6.1f}"
    )


# --------------------------------------------------
# 5. 채널의 값 범위를 확인한다.
#
# S 값의 범위가 좁으면 그대로 보면 거의 검게 보이므로
# 화면 표시용으로만 0~255 범위로 늘린다.
# --------------------------------------------------

print()
print("S 채널 최소 / 최대:", s.min(), "/", s.max())

s_view = cv2.normalize(
    s,
    None,
    0,
    255,
    cv2.NORM_MINMAX,
)


# --------------------------------------------------
# 6. 원본에 비교 영역을 사각형으로 표시한다.
# --------------------------------------------------

marked = image.copy()

for box, color in [(cap_box, (0, 0, 255)), (bg_box, (255, 0, 0))]:

    x1, y1, x2, y2 = to_pixels(box)

    cv2.rectangle(
        marked,
        (x1, y1),
        (x2, y2),
        color,
        4,
    )


# --------------------------------------------------
# 7. 원본 | Gray | S | V 를 한 장으로 붙여 저장한다.
#
# 빨간 사각형 = 병뚜껑 영역, 파란 사각형 = 배경 영역
# --------------------------------------------------

views = [
    ("original", marked),
    ("Gray", gray),
    ("S (normalized)", s_view),
    ("V", v),
]

tiles = []

for name, view in views:

    if len(view.shape) == 2:
        view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)

    tile = cv2.resize(view, None, fx=0.4, fy=0.4)

    cv2.putText(
        tile, name, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
    )

    tiles.append(tile)

compare_path = output_dir / "channel_compare.jpg"

cv2.imwrite(str(compare_path), cv2.hconcat(tiles))
cv2.imwrite(str(output_dir / "s_channel.jpg"), s)

print()
print("저장:", compare_path)


# --------------------------------------------------
# 8. 결과를 보며 확인할 점
#
# - Gray에서 병뚜껑과 배경의 차이는 얼마인가? (부호도 확인)
# - S에서는 병뚜껑이 배경보다 확실히 밝게 보이는가?
# - S 이미지에 네모난 얼룩(JPEG 블록 Noise)이 보이는가?
#   → 다음 3단계(Blur)가 필요한 이유
# --------------------------------------------------
