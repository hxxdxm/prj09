# --------------------------------------------------
# 1. 환경 변화 OCR 검사에 필요한 라이브러리 불러오기
#
# 무엇을 하나요?
# - CSV 결과 저장, 이미지 처리, 파일 경로 기능을 준비합니다.
# - 기존 OCR 공통 함수와 문자열 검사 함수를 불러옵니다.
#
# csv:
# 환경별 OCR 결과를 CSV 파일로 저장
#
# Path:
# 이미지 파일명에서 실제 SN 정답 추출
#
# cv2:
# 이미지 읽기, 4배 확대, 결과 이미지 저장
#
# load_rules():
# 확정된 OCR Rule JSON 읽기
#
# preprocess_for_ocr():
# JSON에서 선택한 전처리 적용
#
# rotate_image():
# 이미지를 지정한 각도로 회전
# Tilt 이미지의 기울기 보정에 사용
#
# run_ocr():
# Tesseract OCR 실행
#
# inspect_text():
# 문자열 Cleaning, 형식 검사, Expected SN 비교,
# 최종 OK/NG 판정
#
# 왜 필요한가요?
# 앞에서 만든 공통 함수를 재사용하여
# 촬영환경 변화에 따른 OCR 결과를 비교하기 위해서입니다.
#
# 주의:
# rotate_image() 함수는 day06_ocr_utils.py에
# 미리 정의되어 있어야 합니다.
# --------------------------------------------------

import csv
from pathlib import Path

import cv2

from day06_ocr_utils import (
    PROJECT_ROOT,
    load_rules,
    preprocess_for_ocr,
    rotate_image,
    run_ocr,
)

from day06_ocr_validation import (
    inspect_text,
)


# --------------------------------------------------
# 2. 환경 변화 이미지와 결과 저장 경로 설정
#
# 무엇을 하나요?
# - 47번에서 만든 환경 변화 이미지 폴더를 지정합니다.
# - OCR 입력 결과 이미지를 저장할 폴더를 지정합니다.
# - OCR 비교 결과 CSV 경로를 지정합니다.
#
# ROOT:
# Low Contrast, Blur, Tilt, Glare 이미지가 있는 폴더
#
# OUTPUT_DIR:
# 실제 OCR에 전달한 확대 이미지를 저장할 폴더
#
# REPORT_PATH:
# 환경 변화별 OCR 검사 결과를 저장할 CSV
#
# 왜 필요한가요?
# 네 가지 조건의 테스트 이미지를 읽고
# 검사 결과를 구분하여 저장하기 위해서입니다.
# --------------------------------------------------

ROOT = (
    PROJECT_ROOT
    / "samples"
    / "day06"
    / "challenge"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "day06"
    / "challenge"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "day06_challenge_results.csv"
)


# --------------------------------------------------
# 3. 확정된 OCR Rule 읽기 및 버전 확인
#
# 무엇을 하나요?
# - load_rules() 함수를 호출합니다.
# - 44번에서 확정한 OCR 설정을 읽습니다.
# - Rule 버전이 day06-ocr-v1.0인지 확인합니다.
#
# 이번 실습의 OCR 조건:
#
# Baseline Version = day06-ocr-v1.0
# Preprocess = original
# PSM = 7
# Whitelist = 0123456789
# Regex = ^[0-9]{13}$
#
# 왜 필요한가요?
# 환경 변화에 따른 OCR 성능을 비교할 때
# OCR Rule 자체가 변경되면 비교가 어려워지기 때문입니다.
#
# 주의:
# 이 코드는 버전 문자열만 확인합니다.
# JSON의 모든 설정이 실제로 동일한지까지
# 검증하는 것은 아닙니다.
# --------------------------------------------------

rules = load_rules()


if rules[
    "baseline_version"
] != "day06-ocr-v1.0":

    raise RuntimeError(
        "OCR Rule이 Freeze되지 않았습니다."
    )


# --------------------------------------------------
# 4. 대표 SN 정답과 결과 저장 폴더 준비
#
# 무엇을 하나요?
# - JSON에서 대표 이미지 파일명을 가져옵니다.
# - 파일명에서 실제 SN 숫자 부분만 추출합니다.
# - 결과 이미지 폴더를 생성합니다.
# - 검사 결과를 저장할 빈 리스트를 준비합니다.
#
# reference_file:
# 대표 제품 이미지 파일명
#
# 예:
# 0637145699424_side.png
#
# Path(reference_file).stem:
# 확장자 제거
#
# .split("_side")[0]:
# "_side" 앞의 SN 숫자 추출
#
# EXPECTED_SN:
# 환경 변화 이미지에서 읽어야 하는 원본 SN
#
# 왜 필요한가요?
# 네 가지 환경 변화 이미지는 모두
# 동일한 대표 SN을 변형해서 만들었기 때문입니다.
#
# 참고:
# 이번 실습에서 EXPECTED_SN은
# OCR이 원본 SN을 정확하게 읽었는지 확인하는
# 정답 문자열로 사용합니다.
# --------------------------------------------------

reference_file = rules[
    "reference_file"
]

EXPECTED_SN = (
    Path(reference_file)
    .stem
    .split("_side")[0]
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# 환경 변화별 OCR 결과를 저장할 리스트
rows = []


# --------------------------------------------------
# 5. 네 가지 환경 변화 폴더에서 이미지 읽기
#
# 무엇을 하나요?
# - 네 가지 환경 변화 조건을 순서대로 확인합니다.
# - 각 조건의 폴더에서 JPG 이미지를 찾습니다.
# - 이미지를 OpenCV로 읽습니다.
#
# 환경 변화 조건:
#
# low_contrast:
# 문자와 배경의 밝기 차이가 낮아진 상태
#
# blur:
# 숫자의 획이 흐려진 상태
#
# tilt:
# 숫자 줄이 +8도 기울어진 상태
#
# glare:
# 빛 반사로 숫자 일부가 밝아진 상태
#
# input_dir.glob("*.jpg"):
# 해당 폴더에서 JPG 파일을 찾습니다.
#
# sorted():
# 파일명 순서로 정렬합니다.
#
# cv2.imread():
# 이미지 파일을 읽는 함수입니다.
#
# if roi is None:
# 이미지를 읽지 못한 경우 다음 이미지로 이동합니다.
#
# 왜 필요한가요?
# 각각의 촬영환경에서 같은 OCR Rule의
# 인식 결과를 비교하기 위해서입니다.
#
# 주의:
# JPG 파일이 없는 조건은 검사하지 않고 넘어갑니다.
# --------------------------------------------------

for condition in [
    "low_contrast",
    "blur",
    "tilt",
    "glare",
]:

    # 현재 환경 변화 조건의 이미지 폴더
    input_dir = (
        ROOT
        / condition
    )

    # 해당 조건의 JPG 이미지 반복
    for image_path in sorted(
        input_dir.glob("*.jpg")
    ):

        # 환경 변화 SN ROI 읽기
        roi = cv2.imread(
            str(image_path)
        )

        # 이미지 읽기에 실패하면 다음 이미지로 이동
        if roi is None:
            continue


        # --------------------------------------------------
        # 6. Freeze된 전처리 방법 적용
        #
        # 무엇을 하나요?
        # - preprocess_for_ocr() 공통 함수를 호출합니다.
        # - JSON에서 확정한 전처리 방법을 적용합니다.
        #
        # 현재 설정:
        # preprocess_mode = original
        #
        # original:
        # 추가적인 Gray, CLAHE, Otsu 등을 적용하지 않고
        # 입력 ROI를 그대로 사용하는 방법입니다.
        #
        # processed:
        # OCR에 사용할 전처리 결과 이미지
        #
        # 왜 필요한가요?
        # 정상 이미지와 환경 변화 이미지 모두
        # 동일한 전처리 조건에서 비교하기 위해서입니다.
        #
        # 주의:
        # 환경 변화에 따라 OCR 설정을 변경하지 않습니다.
        # --------------------------------------------------

        processed = (
            preprocess_for_ocr(
                roi,
                rules,
            )
        )


        # --------------------------------------------------
        # 7. OCR 입력 이미지를 가로·세로 4배 확대
        #
        # 무엇을 하나요?
        # - 전처리한 SN 이미지를 4배 확대합니다.
        # - 43번과 45번에서 사용한 확대 조건을 유지합니다.
        #
        # cv2.resize():
        # 이미지 크기를 변경하는 함수
        #
        # fx=4.0:
        # 가로 4배 확대
        #
        # fy=4.0:
        # 세로 4배 확대
        #
        # INTER_CUBIC:
        # 주변 픽셀을 이용하는 보간 방법
        #
        # 왜 필요한가요?
        # 이미지 확대율을 동일하게 유지해야
        # 환경 변화에 따른 OCR 결과를 비교할 수 있기 때문입니다.
        # --------------------------------------------------

        ocr_input = cv2.resize(
            processed,
            None,
            fx=4.0,
            fy=4.0,
            interpolation=cv2.INTER_CUBIC,
        )


        # --------------------------------------------------
        # 8. 환경 변화 이미지에 Tesseract OCR 실행
        #
        # 무엇을 하나요?
        # - run_ocr() 공통 함수를 호출합니다.
        # - 확대된 이미지와 OCR Rule을 전달합니다.
        # - Tesseract가 인식한 Raw 문자열을 반환받습니다.
        #
        # OCR 설정:
        #
        # OEM = 3
        # PSM = 7
        # Whitelist = 0123456789
        #
        # raw_text:
        # Tesseract가 읽은 원본 문자열
        #
        # 예:
        # '0637145699424\n'
        #
        # 왜 필요한가요?
        # 흐림, 기울기, 대비 저하, 빛 반사 상태에서
        # 같은 OCR 설정이 얼마나 정확하게 동작하는지
        # 확인하기 위해서입니다.
        # --------------------------------------------------

        raw_text = run_ocr(
            ocr_input,
            rules,
        )


        # --------------------------------------------------
        # 9. OCR 문자열 검사 및 원본 SN과 정확도 비교
        #
        # 무엇을 하나요?
        # - inspect_text() 함수를 호출합니다.
        # - OCR 문자열의 공백과 줄바꿈을 제거합니다.
        # - 숫자 13자리 형식인지 확인합니다.
        # - 대표 SN 정답과 비교하여 OK/NG를 판정합니다.
        # - OCR 문자열이 정답과 정확히 같은지 확인합니다.
        #
        # result["final_text"]:
        # 정리된 OCR 문자열
        #
        # EXPECTED_SN:
        # 원본 대표 이미지의 실제 SN
        #
        # ocr_exact:
        # 두 문자열이 정확히 일치하는지 나타냅니다.
        #
        # True:
        # OCR이 원본 SN을 정확하게 읽은 경우
        #
        # False:
        # 하나라도 잘못 읽거나 인식하지 못한 경우
        #
        # 왜 필요한가요?
        # 촬영환경 변화로 OCR 인식 결과가
        # 달라졌는지 판단하기 위해서입니다.
        #
        # 참고:
        # 이번 테스트에서는 모든 변형 이미지의
        # 정답 SN이 동일합니다.
        # --------------------------------------------------

        result = inspect_text(
            raw_text=raw_text,
            expected_text=EXPECTED_SN,
            pattern=rules[
                "expected_pattern"
            ],
        )


        ocr_exact = (
            result["final_text"]
            == EXPECTED_SN
        )


        # --------------------------------------------------
        # 10. 실제 Tesseract OCR 입력 이미지 저장
        #
        # 무엇을 하나요?
        # - 현재 환경 변화 조건으로 파일명을 만듭니다.
        # - 4배 확대된 OCR 입력 이미지를 저장합니다.
        #
        # 저장 파일 예:
        #
        # low_contrast_ocr_input.jpg
        # blur_ocr_input.jpg
        # tilt_ocr_input.jpg
        # glare_ocr_input.jpg
        #
        # 왜 필요한가요?
        # OCR에 실제로 전달한 이미지를 확인하여
        # 실패 원인을 분석하기 위해서입니다.
        #
        # 주의:
        # 같은 조건에 이미지가 여러 장 있으면
        # 동일한 파일명으로 저장되어 덮어써집니다.
        # 이번 47번 실습에서는 조건별 1장을 생성합니다.
        # --------------------------------------------------

        output_path = (
            OUTPUT_DIR
            / f"{condition}_ocr_input.jpg"
        )

        cv2.imwrite(
            str(output_path),
            ocr_input,
        )


        # --------------------------------------------------
        # 11. Deskew를 적용하지 않은 OCR 결과 기록
        #
        # 무엇을 하나요?
        # - 현재 환경 변화 조건의 OCR 결과를 정리합니다.
        # - Deskew 여부를 "no"로 기록합니다.
        # - 결과를 rows 리스트에 추가합니다.
        #
        # condition:
        # 현재 환경 변화 조건 이름
        #
        # deskew:
        # 기울기 보정 사용 여부
        #
        # expected_sn:
        # 실제로 읽어야 하는 대표 SN
        #
        # raw_text:
        # Tesseract 원본 문자열
        #
        # final_text:
        # 공백과 줄바꿈을 제거한 OCR 문자열
        #
        # format_ok:
        # 숫자 13자리 형식 검사 결과
        #
        # expected_match:
        # Expected SN과 일치 여부
        #
        # ocr_exact:
        # OCR 결과가 원본 SN과 정확히 같은지 확인
        #
        # status:
        # 최종 OK / NG 판정
        #
        # reasons:
        # NG 판정 이유
        #
        # 왜 필요한가요?
        # 기울기 보정 이전의 OCR 결과를 기록하여
        # 이후 Deskew 결과와 비교하기 위해서입니다.
        # --------------------------------------------------

        rows.append(
            {
                "condition":
                    condition,

                "deskew":
                    "no",

                "expected_sn":
                    EXPECTED_SN,

                "raw_text":
                    repr(
                        result["raw_text"]
                    ),

                "final_text":
                    result["final_text"],

                "format_ok":
                    result["format_ok"],

                "expected_match":
                    result["expected_match"],

                "ocr_exact":
                    ocr_exact,

                "status":
                    result[
                        "predicted_status"
                    ],

                "reasons":
                    "|".join(
                        result["reasons"]
                    ),
            }
        )


        # --------------------------------------------------
        # 12. Deskew를 적용하지 않은 OCR 결과 출력
        #
        # 무엇을 하나요?
        # - 현재 환경 변화 조건을 출력합니다.
        # - Deskew를 적용하지 않았음을 표시합니다.
        # - OCR 문자열과 정답 일치 여부를 출력합니다.
        #
        # Condition:
        # 현재 테스트 환경
        #
        # Deskew: no
        # 기울기 보정을 적용하지 않음
        #
        # OCR:
        # Tesseract가 인식한 문자열
        #
        # Exact:
        # 원본 SN과 완전히 일치하는지 확인한 결과
        #
        # 왜 필요한가요?
        # 환경 변화별 OCR 성공과 실패를
        # 터미널에서 바로 확인하기 위해서입니다.
        # --------------------------------------------------

        print(
            "=" * 60
        )

        print(
            "Condition:",
            condition
        )

        print(
            "Deskew: no"
        )

        print(
            "OCR:",
            result["final_text"]
        )

        print(
            "Exact:",
            ocr_exact
        )


        # --------------------------------------------------
        # 13. Tilt 이미지에만 -8도 Deskew 적용
        #
        # 무엇을 하나요?
        # - 현재 환경 변화 조건이 tilt인지 확인합니다.
        # - Tilt 조건일 때만 추가 OCR 실험을 수행합니다.
        # - rotate_image()로 이미지를 -8도 회전시킵니다.
        # - 확정된 전처리를 다시 적용합니다.
        #
        # 47번 실습:
        # 정상 이미지 → +8도 Tilt
        #
        # 이번 48번 실습:
        # +8도 Tilt 이미지 → -8도 Deskew
        #
        # Deskew란?
        # 기울어진 문자 줄을 다시 수평에 가깝게
        # 보정하는 영상처리 방법입니다.
        #
        # 왜 필요한가요?
        # OCR 실패 원인이 문자 기울기라면
        # 기울기 보정 후 결과가 개선되는지 확인하기 위해서입니다.
        #
        # 주의:
        # -8도는 이번 실습에서 이미 알고 있는
        # +8도의 반대 각도를 사용하는 것입니다.
        #
        # 실제 산업현장에서는 기울어진 각도를
        # 별도로 추정해야 할 수 있습니다.
        # --------------------------------------------------

        if condition == "tilt":

            # 기울어진 이미지를 반대 방향으로 8도 회전
            deskewed = rotate_image(
                roi,
                -8.0,
            )

            # Deskew 결과에 확정된 전처리 적용
            deskew_processed = (
                preprocess_for_ocr(
                    deskewed,
                    rules,
                )
            )


            # --------------------------------------------------
            # 14. Deskew 이미지를 4배 확대하고 OCR 실행
            #
            # 무엇을 하나요?
            # - Deskew 결과 이미지를 4배 확대합니다.
            # - Tesseract OCR을 다시 실행합니다.
            # - 인식한 문자열을 검사합니다.
            # - 원본 SN과 일치하는지 확인합니다.
            #
            # deskew_input:
            # 기울기 보정 후 4배 확대된 이미지
            #
            # deskew_raw:
            # Tesseract가 반환한 원본 문자열
            #
            # deskew_result:
            # 문자열 형식 검사 및 OK/NG 판정 결과
            #
            # deskew_exact:
            # Deskew 후 OCR이 정답 SN을 정확하게 읽었는지
            # 확인하는 값
            #
            # 왜 필요한가요?
            # 동일한 Tilt 이미지에서
            # Deskew 적용 전후 OCR 결과를 비교하기 위해서입니다.
            #
            # 중요:
            # Deskew를 적용했다고 해서 OCR이 반드시
            # 성공하는 것은 아닙니다.
            # 회전 과정에서 이미 손실된 문자 정보는
            # 완전히 복원되지 않을 수 있습니다.
            # --------------------------------------------------

            deskew_input = cv2.resize(
                deskew_processed,
                None,
                fx=4.0,
                fy=4.0,
                interpolation=cv2.INTER_CUBIC,
            )

            deskew_raw = run_ocr(
                deskew_input,
                rules,
            )

            deskew_result = (
                inspect_text(
                    raw_text=deskew_raw,
                    expected_text=EXPECTED_SN,
                    pattern=rules[
                        "expected_pattern"
                    ],
                )
            )

            deskew_exact = (
                deskew_result[
                    "final_text"
                ]
                == EXPECTED_SN
            )


            # --------------------------------------------------
            # 15. Deskew 후 OCR 입력 이미지 저장
            #
            # 무엇을 하나요?
            # - 기울기 보정 후 실제 OCR에 입력한
            #   확대 이미지를 저장합니다.
            #
            # 저장 파일:
            # tilt_deskew_ocr_input.jpg
            #
            # 왜 필요한가요?
            # 기존 Tilt 이미지와 Deskew 결과를 비교하여
            # 숫자 줄이 얼마나 수평에 가까워졌는지
            # 확인하기 위해서입니다.
            # --------------------------------------------------

            cv2.imwrite(
                str(
                    OUTPUT_DIR
                    / "tilt_deskew_ocr_input.jpg"
                ),
                deskew_input,
            )


            # --------------------------------------------------
            # 16. Deskew 적용 후 OCR 결과 기록 및 출력
            #
            # 무엇을 하나요?
            # - Deskew 적용 결과를 CSV 저장 목록에 추가합니다.
            # - deskew 값을 "yes"로 기록합니다.
            # - OCR 문자열과 Exact Match 결과를 출력합니다.
            #
            # 기존 Tilt 결과:
            # condition = tilt
            # deskew = no
            #
            # Deskew 적용 결과:
            # condition = tilt
            # deskew = yes
            #
            # 왜 필요한가요?
            # 같은 이미지에서 기울기 보정 적용 전후
            # 결과를 서로 구분하여 비교하기 위해서입니다.
            #
            # 이번 실습에서 생성하는 결과 행:
            #
            # Low Contrast → 1행
            # Blur         → 1행
            # Tilt         → 2행 (Deskew 전/후)
            # Glare        → 1행
            #
            # 총 5행
            #
            # 단, 조건별 입력 이미지가 1장씩
            # 정상적으로 읽혔을 때의 결과입니다.
            # --------------------------------------------------

            rows.append(
                {
                    "condition":
                        "tilt",

                    "deskew":
                        "yes",

                    "expected_sn":
                        EXPECTED_SN,

                    "raw_text":
                        repr(
                            deskew_result[
                                "raw_text"
                            ]
                        ),

                    "final_text":
                        deskew_result[
                            "final_text"
                        ],

                    "format_ok":
                        deskew_result[
                            "format_ok"
                        ],

                    "expected_match":
                        deskew_result[
                            "expected_match"
                        ],

                    "ocr_exact":
                        deskew_exact,

                    "status":
                        deskew_result[
                            "predicted_status"
                        ],

                    "reasons":
                        "|".join(
                            deskew_result[
                                "reasons"
                            ]
                        ),
                }
            )


            print(
                "Deskew: yes"
            )

            print(
                "OCR:",
                deskew_result[
                    "final_text"
                ]
            )

            print(
                "Exact:",
                deskew_exact
            )


# --------------------------------------------------
# 17. 검사 결과 존재 여부 확인 및 CSV 저장
#
# 무엇을 하나요?
# - 테스트 결과가 한 건 이상 있는지 확인합니다.
# - 결과가 없다면 오류를 발생시킵니다.
# - CSV 저장 폴더를 준비합니다.
# - 환경 변화별 OCR 결과를 CSV로 저장합니다.
#
# if not rows:
# 검사 결과가 비어 있는지 확인합니다.
#
# csv.DictWriter():
# 딕셔너리 목록을 CSV로 저장하는 기능
#
# fieldnames=list(rows[0].keys()):
# 첫 번째 결과 딕셔너리의 Key를 CSV 컬럼명으로 사용
#
# writer.writeheader():
# 컬럼명 작성
#
# writer.writerows(rows):
# 전체 검사 결과 저장
#
# encoding="utf-8-sig":
# Excel에서 한글이 깨질 가능성을 줄입니다.
#
# 저장 파일:
# reports/day06_challenge_results.csv
#
# 왜 필요한가요?
# 환경 변화 조건별 OCR 성공과 실패를
# 표로 확인하고 비교하기 위해서입니다.
#
# 주의:
# 이번 코드에서는 결과 CSV가 이미 존재하면
# 기존 파일 내용을 덮어씁니다.
# --------------------------------------------------

if not rows:
    raise RuntimeError(
        "환경 변화 테스트 이미지가 없습니다."
    )


REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with REPORT_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


# --------------------------------------------------
# 18. 환경 변화 OCR 테스트 완료 메시지 출력
#
# 무엇을 하나요?
# - 결과 CSV 저장 경로를 출력합니다.
# - 환경 변화 OCR 검증 완료 메시지를 출력합니다.
#
# 왜 필요한가요?
# 네 가지 환경 변화 및 Tilt Deskew 실험 결과가
# 저장되었음을 확인하기 위해서입니다.
#
# 저장 결과:
# reports/day06_challenge_results.csv
#
# 다음 단계에서는 다음 내용을 확인합니다.
#
# - 어떤 환경 변화에서 OCR이 실패했는가?
# - Tilt 이미지의 Deskew 전후 결과가 달라졌는가?
# - 원본 Baseline Rule의 한계는 무엇인가?
#
# 주의:
# 이번 코드는 OCR 결과를 비교하는 실험입니다.
# 모든 환경에서 정확하게 인식하는
# 새로운 OCR Rule을 자동으로 만들어 주지는 않습니다.
# --------------------------------------------------

print()

print(
    "Saved:",
    REPORT_PATH
)

print(
    "Environment Stress Test Complete"
)