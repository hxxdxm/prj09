from pathlib import Path
import csv


report_dir = Path("reports")

paths = {
    "template": report_dir / "day04_template_results.csv",
    "orb": report_dir / "day04_orb_results.csv",
    "sift": report_dir / "day04_sift_results.csv",
    "homography": report_dir / "day04_homography_results.csv",
}

output_path = (
    report_dir /
    "day04_rotation_compare.csv"
)


# --------------------------------------------------
# CSV를 condition 기준으로 읽는 함수
# --------------------------------------------------

def read_results(path):

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        rows = csv.DictReader(file)

        return {
            row["condition"]: row
            for row in rows
        }


# --------------------------------------------------
# 앞에서 만든 결과 읽기
# --------------------------------------------------

template = read_results(
    paths["template"]
)

orb = read_results(
    paths["orb"]
)

sift = read_results(
    paths["sift"]
)

homography = read_results(
    paths["homography"]
)


# --------------------------------------------------
# 같은 네 조건을 하나의 표로 정리
# --------------------------------------------------

conditions = [
    "normal",
    "shift",
    "rot15",
    "scale90",
]


rows = []


for condition in conditions:

    row = {
        "condition": condition,

        "template_score":
            template[condition][
                "template_score"
            ],

        "orb_good_matches":
            orb[condition][
                "good_matches"
            ],

        "sift_good_matches":
            sift[condition][
                "good_matches"
            ],

        "homography_inliers":
            homography[condition][
                "inliers"
            ],

        "inlier_ratio":
            homography[condition][
                "inlier_ratio"
            ],

        "homography_status":
            homography[condition][
                "status"
            ],
    }

    rows.append(row)


# --------------------------------------------------
# 비교 CSV 저장
# --------------------------------------------------

fieldnames = [
    "condition",
    "template_score",
    "orb_good_matches",
    "sift_good_matches",
    "homography_inliers",
    "inlier_ratio",
    "homography_status",
]


with output_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(rows)


print("저장:", output_path)
print()


for row in rows:

    print(
        row["condition"],
        "| Template:",
        row["template_score"],
        "| ORB:",
        row["orb_good_matches"],
        "| SIFT:",
        row["sift_good_matches"],
        "| Inlier:",
        row["homography_inliers"],
        "| Status:",
        row["homography_status"],
    )