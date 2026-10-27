from pathlib import Path

import cv2


# --------------------------------------------------
# 1. Challenge 이미지의 경로를 정한다.
#
# challenge.jpg는 이미 문제가 있는 이미지이므로
# 어둡게 만드는 단계 없이 원본에 바로 적용한다.
# --------------------------------------------------

image_path = Path(
    "samples/day02/challenge.jpg"
)


# --------------------------------------------------
# 2. Brightness 비교 결과를 저장할 폴더를 준비한다.
# --------------------------------------------------

output_dir = Path(
    "outputs/day02/brightness_challenge"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 3. Challenge 이미지를 읽는다.
# --------------------------------------------------

image = cv2.imread(
    str(image_path)
)


# --------------------------------------------------
# 4. 이미지가 정상적으로 읽히지 않았다면 오류를 발생시킨다.
# --------------------------------------------------

if image is None:
    raise FileNotFoundError(
        f"이미지를 읽지 못했습니다: {image_path}"
    )


# --------------------------------------------------
# 5. 비교할 alpha, beta 값을 준비한다.
#
# alpha = 1.0 고정, beta만 변경 → Brightness 비교
# alpha > 1.0, beta < 0       → Contrast 비교
# --------------------------------------------------

settings = [
    (1.0, 0),      # 원본
    (1.0, 50),     # 밝기만 증가
    (1.5, -60),    # 대비 증가
]

# --------------------------------------------------
# 6. 각 설정값을 원본에 적용하고
#    결과 이미지와 Otsu 결과를 함께 저장한다.
# --------------------------------------------------

print("alpha | beta | 평균밝기 | 표준편차 | Otsu 임계값 | 흰색비율")

for alpha, beta in settings:

    # 6-1. alpha, beta 적용
    result = cv2.convertScaleAbs(
        image,
        alpha=alpha,
        beta=beta,
    )

    # 6-2. Gray 변환
    gray = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2GRAY,
    )

    # 6-3. Otsu 이진화
    # 밝은 병뚜껑을 흰색 객체로 보기 위해 THRESH_BINARY 사용
    otsu_value, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # 6-4. 결과 저장
    name = f"alpha_{alpha}_beta_{beta}"

    cv2.imwrite(
        str(output_dir / f"{name}.jpg"),
        result,
    )

    cv2.imwrite(
        str(output_dir / f"{name}_otsu.jpg"),
        otsu,
    )

    # 6-5. 수치 출력 → Parameter Table에 기록
    white_ratio = (otsu > 0).mean() * 100

    print(
        f"{alpha:5} | {beta:4} | "
        f"{gray.mean():8.1f} | {gray.std():8.1f} | "
        f"{otsu_value:11.1f} | {white_ratio:6.1f}%"
    )


print("저장 위치:", output_dir)
print("Brightness / Contrast 비교 완료")


# --------------------------------------------------
# 7. 결과를 비교하며 확인할 점
#
# - beta만 바꾼 4장의 Otsu 결과가 서로 같은가?
#   (Otsu 임계값도 beta만큼 같이 올라가는지 확인)
# - alpha를 키운 결과에서 표준편차가 커졌는가?
# - 그래서 병뚜껑과 배경의 분리가 실제로 좋아졌는가?
# - alpha=2.0에서 밝은 영역이 하얗게 날아가지 않았는가?
# --------------------------------------------------