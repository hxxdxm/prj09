from pathlib import Path
from time import perf_counter
import csv

import cv2


# --------------------------------------------------
# 1. 경로 설정
# --------------------------------------------------

template_path = Path(
    "samples/day04/reference/reference_label.jpg"
)

target_dir = Path(
    "samples/day04/targets"
)

output_dir = Path(
    "outputs/day04/template"
)

report_path = Path(
    "reports/day04_template_results.csv"
)


output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

report_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 2. 비교할 Target을 순서대로 지정
# --------------------------------------------------

target_settings = [
    (
        "normal",
        target_dir / "target_normal.jpg",
    ),
    (
        "shift",
        target_dir / "target_shift.jpg",
    ),
    (
        "rot15",
        target_dir / "target_rot15.jpg",
    ),
    (
        "scale90",
        target_dir / "target_scale90.jpg",
    ),
]


# --------------------------------------------------
# 3. 작은 기준 Template 읽기
# --------------------------------------------------

template = cv2.imread(
    str(template_path),
    cv2.IMREAD_GRAYSCALE,
)

if template is None:
    raise FileNotFoundError(
        template_path
    )


template_h, template_w = (
    template.shape[:2]
)

print(
    "Template size:",
    template_w,
    "x",
    template_h,
)


# --------------------------------------------------
# 4. 결과 저장용 리스트
# --------------------------------------------------

rows = []


# --------------------------------------------------
# 5. 네 개의 Target에 같은 방법 적용
# --------------------------------------------------

for condition, target_path in target_settings:

    target = cv2.imread(
        str(target_path)
    )

    if target is None:
        raise FileNotFoundError(
            target_path
        )


    gray = cv2.cvtColor(
        target,
        cv2.COLOR_BGR2GRAY,
    )


    target_h, target_w = (
        gray.shape[:2]
    )


    # Template이 Target보다 크면
    # Template Matching을 할 수 없음
    if (
        template_w > target_w
        or template_h > target_h
    ):
        raise ValueError(
            f"Template이 Target보다 큽니다: "
            f"{target_path.name}"
        )


    # --------------------------------------------------
    # 6. Template Matching 실행
    # --------------------------------------------------

    start = perf_counter()


    score_map = cv2.matchTemplate(
        gray,
        template,
        cv2.TM_CCOEFF_NORMED,
    )


    _, max_score, _, max_loc = (
        cv2.minMaxLoc(
            score_map
        )
    )


    elapsed_ms = (
        perf_counter() - start
    ) * 1000


    # --------------------------------------------------
    # 7. 가장 비슷한 위치 계산
    # --------------------------------------------------

    top_left = max_loc

    bottom_right = (
        top_left[0] + template_w,
        top_left[1] + template_h,
    )


    center_x = (
        top_left[0]
        + template_w / 2
    )

    center_y = (
        top_left[1]
        + template_h / 2
    )


    # --------------------------------------------------
    # 8. 찾은 위치를 이미지에 표시
    # --------------------------------------------------

    preview = target.copy()


    cv2.rectangle(
        preview,
        top_left,
        bottom_right,
        (0, 255, 0),
        3,
    )


    cv2.putText(
        preview,
        f"score={max_score:.3f}",
        (
            top_left[0],
            max(
                30,
                top_left[1] - 10,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


    # --------------------------------------------------
    # 9. 결과 이미지 저장
    # --------------------------------------------------

    output_path = (
        output_dir
        / f"{condition}_template.jpg"
    )


    cv2.imwrite(
        str(output_path),
        preview,
    )


    # --------------------------------------------------
    # 10. CSV에 기록할 값 저장
    # --------------------------------------------------

    rows.append({
        "condition": condition,
        "filename": target_path.name,
        "template_score": f"{max_score:.6f}",
        "x": top_left[0],
        "y": top_left[1],
        "center_x": f"{center_x:.2f}",
        "center_y": f"{center_y:.2f}",
        "time_ms": f"{elapsed_ms:.3f}",
    })


    print(
        f"{condition:7s}",
        f"score={max_score:.4f}",
        f"location={top_left}",
        f"time={elapsed_ms:.2f} ms",
    )


# --------------------------------------------------
# 11. CSV 저장
# --------------------------------------------------

with report_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "condition",
            "filename",
            "template_score",
            "x",
            "y",
            "center_x",
            "center_y",
            "time_ms",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)


print()
print(
    "CSV 저장:",
    report_path,
)

print(
    "결과 이미지:",
    output_dir,
)