# --------------------------------------------------
# 1. CSV 처리 라이브러리와 프로젝트 경로 불러오기
#
# 무엇을 하나요?
# - CSV 파일을 읽는 기능을 불러옵니다.
# - 앞에서 만든 OCR 공통 모듈에서
#   프로젝트 최상위 폴더 경로를 가져옵니다.
#
# csv:
# 45번 실습에서 저장한 OCR 결과 CSV를 읽습니다.
#
# PROJECT_ROOT:
# 프로젝트 최상위 폴더 경로입니다.
#
# 왜 필요한가요?
# 앞에서 생성한 Batch 검사 결과를 읽어
# 전체 OCR 성능을 평가하기 위해서입니다.
#
# 주의:
# 이번 코드는 실제 OCR을 다시 실행하지 않습니다.
# 이미 저장된 검사 결과 CSV만 분석합니다.
# --------------------------------------------------

import csv

from day06_ocr_utils import (
    PROJECT_ROOT,
)


# --------------------------------------------------
# 2. Batch 결과 CSV와 평가 보고서 저장 경로 설정
#
# 무엇을 하나요?
# - 앞에서 저장한 OCR 검사 결과 CSV 경로를 지정합니다.
# - 전체 평가 결과를 저장할 Markdown 파일 경로를 지정합니다.
#
# RESULT_PATH:
# 45_day06_batch_ocr.py에서 생성한
# 이미지별 OCR 검사 결과 CSV입니다.
#
# METRIC_PATH:
# 이번 코드에서 생성할 전체 성능 평가 보고서입니다.
#
# 입력:
# reports/day06_ocr_results.csv
#
# 출력:
# reports/day06_ocr_metrics.md
#
# 왜 필요한가요?
# 이미지별 검사 결과를 읽고
# 전체 정확도를 하나의 보고서로 정리하기 위해서입니다.
# --------------------------------------------------

RESULT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "day06_ocr_results.csv"
)

METRIC_PATH = (
    PROJECT_ROOT
    / "reports"
    / "day06_ocr_metrics.md"
)


# --------------------------------------------------
# 3. Batch 검사 결과 CSV가 존재하는지 확인
#
# 무엇을 하나요?
# - RESULT_PATH에 CSV 파일이 존재하는지 확인합니다.
# - 파일이 없다면 오류를 발생시킵니다.
#
# RESULT_PATH.exists():
# 지정한 파일이나 경로가 존재하는지 확인합니다.
#
# FileNotFoundError:
# 필요한 파일을 찾지 못하면 발생시키는 오류입니다.
#
# 왜 필요한가요?
# 45번 Batch 검사를 완료해야
# 실제 OCR 결과와 제품 판정 결과를 평가할 수 있기 때문입니다.
#
# CSV가 없다면:
# 먼저 45_day06_batch_ocr.py를 실행해야 합니다.
# --------------------------------------------------

if not RESULT_PATH.exists():

    raise FileNotFoundError(
        "먼저 45_day06_batch_ocr.py를 "
        "실행해야 합니다.\n"
        f"결과 파일이 없습니다: "
        f"{RESULT_PATH}"
    )


# --------------------------------------------------
# 4. Batch 결과 CSV 읽기 및 데이터 존재 여부 확인
#
# 무엇을 하나요?
# - CSV 파일을 엽니다.
# - CSV의 각 행을 딕셔너리 형태로 읽습니다.
# - 검사 결과가 하나 이상 있는지 확인합니다.
#
# csv.DictReader():
# CSV의 컬럼 이름을 Key로 사용하여
# 각 행을 딕셔너리로 읽는 함수입니다.
#
# rows:
# 전체 이미지의 검사 결과가 저장된 리스트입니다.
#
# 예:
# rows[0]["ocr_exact"] = "True"
#
# rows[0]["inspection_correct"] = "True"
#
# encoding="utf-8-sig":
# UTF-8 BOM이 포함된 CSV도 읽도록 설정합니다.
#
# if not rows:
# 검사 결과가 하나도 없는지 확인합니다.
#
# 왜 필요한가요?
# 저장된 이미지별 검사 결과를 분석하여
# 전체 정확도를 계산하기 위해서입니다.
#
# 중요:
# rows가 비어 있으면 프로그램을 중단합니다.
# 따라서 뒤에서 전체 이미지 수로 나눌 때
# 분모가 0이 되는 문제를 방지할 수 있습니다.
# --------------------------------------------------

with RESULT_PATH.open(
    "r",
    newline="",
    encoding="utf-8-sig",
) as file:

    rows = list(
        csv.DictReader(file)
    )


if not rows:

    raise RuntimeError(
        "day06_ocr_results.csv에 "
        "결과가 없습니다."
    )


# --------------------------------------------------
# 5. CSV의 문자열 "True" / "False"를 Boolean으로 변환
#
# 무엇을 하나요?
# - as_bool() 함수를 정의합니다.
# - CSV에서 읽은 문자열을 True 또는 False로 변환합니다.
#
# 왜 필요한가요?
# Python 코드에서는 True와 False를 Boolean으로 사용하지만
# CSV 파일에서 읽으면 문자열이기 때문입니다.
#
# 예:
#
# CSV에 저장된 값:
# "True"
#
# Python에서 읽은 값:
# 문자열 "True"
#
# 우리가 계산에 사용할 값:
# Boolean True
#
# str(value):
# 전달받은 값을 문자열로 변환합니다.
#
# strip():
# 앞뒤 공백을 제거합니다.
#
# lower():
# 영문 대문자를 소문자로 변환합니다.
#
# == "true":
# 최종 문자열이 true와 같은지 확인합니다.
#
# 변환 결과:
#
# "True"  → True
# "true"  → True
# "FALSE" → False
# "False" → False
#
# 주의:
# 현재 함수에서는 "true"와 일치하는 값만 True입니다.
# 다른 값은 모두 False로 처리합니다.
# --------------------------------------------------

def as_bool(
    value,
):
    return (
        str(value)
        .strip()
        .lower()
        == "true"
    )


# --------------------------------------------------
# 6. 전체 검사 이미지 개수 계산
#
# 무엇을 하나요?
# - CSV에 저장된 검사 결과 행의 개수를 계산합니다.
#
# len(rows):
# rows 리스트에 들어 있는 항목 수를 반환합니다.
#
# count:
# 전체 검사 이미지 수
#
# 예:
# 이미지 6장을 검사했다면
# count = 6
#
# 왜 필요한가요?
# OCR 정확도와 제품 검사 정확도를 계산할 때
# 전체 이미지 수가 분모로 사용되기 때문입니다.
# --------------------------------------------------

count = len(
    rows
)


# --------------------------------------------------
# 7. OCR이 실제 인쇄된 SN을 정확히 읽은 개수 계산
#
# 무엇을 하나요?
# - 각 이미지의 ocr_exact 값을 확인합니다.
# - True인 이미지 개수를 계산합니다.
#
# ocr_exact:
# OCR 결과가 실제 인쇄된 SN과 일치하는지 나타냅니다.
#
# True:
# 실제 인쇄된 SN을 정확하게 읽은 경우
#
# False:
# 실제 인쇄된 SN과 OCR 결과가 다른 경우
#
# sum():
# True 값을 1로 계산하여 모두 더합니다.
#
# as_bool():
# CSV에 저장된 "True" 문자열을
# Boolean True로 변환합니다.
#
# 예:
#
# 이미지 6장
#
# True
# True
# True
# True
# True
# True
#
# 결과:
# ocr_correct = 6
#
# 왜 필요한가요?
# OCR 자체의 문자 인식 정확도를
# 계산하기 위해서입니다.
#
# 중요:
# 제품이 OK인지 NG인지는 관계없습니다.
# 실제 인쇄된 문자열을 정확히 읽었는지만 확인합니다.
# --------------------------------------------------

ocr_correct = sum(
    as_bool(
        row["ocr_exact"]
    )
    for row in rows
)


# --------------------------------------------------
# 8. 제품의 최종 OK / NG 판정이 정확한 개수 계산
#
# 무엇을 하나요?
# - 각 이미지의 inspection_correct 값을 확인합니다.
# - True인 이미지 개수를 계산합니다.
#
# inspection_correct:
# 프로그램이 예측한 OK/NG가
# 실제 제품 정답과 일치하는지 나타냅니다.
#
# 예:
#
# 실제 상태 = NG
# 프로그램 판정 = NG
#
# inspection_correct = True
#
# 실제 상태 = OK
# 프로그램 판정 = NG
#
# inspection_correct = False
#
# 왜 필요한가요?
# OCR이 문자를 잘 읽었는지와
# 제품의 정상/불량을 올바르게 판정했는지는
# 서로 다른 평가이기 때문입니다.
#
# 예:
# 6장 모두 정확하게 판정했다면
# inspection_correct_count = 6
# --------------------------------------------------

inspection_correct_count = sum(
    as_bool(
        row["inspection_correct"]
    )
    for row in rows
)


# --------------------------------------------------
# 9. SN 형식 검사와 Expected SN 일치 개수 계산
#
# 무엇을 하나요?
# - 숫자 13자리 형식을 만족한 이미지 수를 계산합니다.
# - OCR 결과가 Expected SN과 일치한 이미지 수를 계산합니다.
#
# format_ok:
# OCR 문자열이 숫자 13자리인지 검사한 결과
#
# 정규식:
# ^[0-9]{13}$
#
# expected_match:
# OCR 결과가 Expected SN과 같은지 확인한 결과
#
# 예:
#
# 제품 6장 중
# 6장이 모두 숫자 13자리이면
#
# format_ok_count = 6
#
# 4장은 Expected SN과 일치하고
# 2장은 다르다면
#
# expected_match_count = 4
#
# 왜 필요한가요?
# OCR 문자열의 형식 검사와
# Expected SN 비교 결과를
# 전체 이미지 기준으로 확인하기 위해서입니다.
#
# 중요:
# Expected Match는 정확도와 다릅니다.
#
# NG 제품에서 실제 인쇄된 SN이 Expected SN과
# 다르다면 Expected Match = False가 정상일 수 있습니다.
# --------------------------------------------------

format_ok_count = sum(
    as_bool(
        row["format_ok"]
    )
    for row in rows
)


expected_match_count = sum(
    as_bool(
        row["expected_match"]
    )
    for row in rows
)


# --------------------------------------------------
# 10. 전체 OCR 정확도 계산
#
# 무엇을 하나요?
# - 실제 SN을 정확하게 읽은 비율을 계산합니다.
#
# 계산식:
#
# OCR Exact Accuracy
# = OCR을 정확하게 읽은 이미지 수 / 전체 이미지 수
#
# 예:
#
# OCR Correct = 6
# Total Images = 6
#
# OCR Accuracy = 6 / 6
#              = 1.0
#              = 100%
#
# 왜 필요한가요?
# Tesseract가 이미지에 실제로 인쇄된 SN을
# 얼마나 정확하게 읽었는지 확인하기 위해서입니다.
#
# 주의:
# 여기서의 정확도는 SN 전체 문자열의
# 완전 일치 비율입니다.
#
# 개별 숫자 하나하나의 문자 정확도나
# Character Error Rate를 계산하는 것은 아닙니다.
# --------------------------------------------------

ocr_accuracy = (
    ocr_correct
    / count
)


# --------------------------------------------------
# 11. 전체 제품 OK / NG 판정 정확도 계산
#
# 무엇을 하나요?
# - 실제 제품 상태를 올바르게 판정한 비율을 계산합니다.
#
# 계산식:
#
# Inspection Accuracy
# = 올바르게 판정한 이미지 수 / 전체 이미지 수
#
# 예:
#
# Inspection Correct = 6
# Total Images = 6
#
# Inspection Accuracy = 6 / 6
#                     = 1.0
#                     = 100%
#
# 왜 필요한가요?
# 제품의 실제 상태가 OK인지 NG인지
# 프로그램이 올바르게 판정했는지 확인하기 위해서입니다.
#
# 중요:
# OCR Accuracy와 Inspection Accuracy는
# 항상 같은 값이 나오는 것은 아닙니다.
# --------------------------------------------------

inspection_accuracy = (
    inspection_correct_count
    / count
)


# --------------------------------------------------
# 12. OCR 실패 및 제품 판정 실패 이미지 목록 찾기
#
# 무엇을 하나요?
# - OCR이 실제 SN을 잘못 읽은 이미지 파일명을 찾습니다.
# - 제품의 OK/NG 판정이 틀린 이미지 파일명을 찾습니다.
#
# ocr_failed_files:
# ocr_exact가 False인 이미지 목록
#
# inspection_failed_files:
# inspection_correct가 False인 이미지 목록
#
# 리스트 컴프리헨션:
# 조건에 맞는 항목만 골라 새로운 리스트를 만드는 문법
#
# if not as_bool(...):
# 검사 결과가 False인 경우에만 파일명을 가져옵니다.
#
# 왜 필요한가요?
# 전체 정확도만 확인하는 것이 아니라
# 어떤 제품에서 오류가 발생했는지
# 파일명으로 확인하기 위해서입니다.
#
# 주의:
# 현재 코드는 실패한 이미지의 파일명만 찾습니다.
# 이미지를 다른 폴더로 복사하거나
# 실패 원인을 자동 분석하지는 않습니다.
# --------------------------------------------------

ocr_failed_files = [
    row["file"]
    for row in rows
    if not as_bool(
        row["ocr_exact"]
    )
]


inspection_failed_files = [
    row["file"]
    for row in rows
    if not as_bool(
        row["inspection_correct"]
    )
]


# --------------------------------------------------
# 13. 실패 이미지 목록을 Markdown 문자열로 변환
#
# 무엇을 하나요?
# - OCR 실패 이미지 목록을 Markdown 형태로 만듭니다.
# - 제품 판정 실패 이미지 목록도 같은 방식으로 만듭니다.
#
# "\n".join():
# 여러 문자열을 줄바꿈으로 연결합니다.
#
# "- " + name:
# Markdown 목록 형식으로 파일명을 작성합니다.
#
# 예:
#
# - image_01.png
# - image_02.png
#
# if ocr_failed_files:
# OCR 실패 이미지가 존재하는지 확인합니다.
#
# 실패 이미지가 없다면:
# "- 없음"으로 기록합니다.
#
# 왜 필요한가요?
# Markdown 보고서에서 실패한 이미지 이름을
# 한 줄씩 읽기 쉽게 표시하기 위해서입니다.
# --------------------------------------------------

ocr_failure_text = (
    "\n".join(
        "- " + name
        for name in ocr_failed_files
    )
    if ocr_failed_files
    else "- 없음"
)


inspection_failure_text = (
    "\n".join(
        "- " + name
        for name in inspection_failed_files
    )
    if inspection_failed_files
    else "- 없음"
)


# --------------------------------------------------
# 14. 전체 OCR 성능 평가 보고서 작성
#
# 무엇을 하나요?
# - 전체 검사 이미지 수를 기록합니다.
# - OCR Exact Accuracy를 기록합니다.
# - Inspection Accuracy를 기록합니다.
# - 형식 검사 성공 개수를 기록합니다.
# - Expected SN 일치 개수를 기록합니다.
# - OCR 실패 이미지 목록을 기록합니다.
# - 제품 판정 실패 이미지 목록을 기록합니다.
#
# text:
# Markdown 보고서의 전체 내용이 저장되는 문자열
#
# f""" ... """:
# 여러 줄의 문자열 안에 변수 값을 넣는 방법입니다.
#
# {ocr_accuracy:.4f}:
# OCR 정확도를 소수점 아래 4자리까지 표시합니다.
#
# 예:
# 1.0 → 1.0000
#
# {count}:
# 전체 검사 이미지 수를 표시합니다.
#
# 왜 필요한가요?
# 45번에서 저장한 이미지별 결과를
# 하나의 성능 평가 보고서로 정리하기 위해서입니다.
#
# 주의:
# Expected Match는 제품 검사 정확도가 아닙니다.
#
# 실제 NG 제품에서는 Expected SN과 불일치하는 것이
# 의도한 검사 결과일 수 있습니다.
# --------------------------------------------------

text = f"""# Day 06 OCR Metrics

## Dataset

Images: {count}

## OCR Exact Accuracy

{ocr_accuracy:.4f}

OCR Correct: {ocr_correct} / {count}

## Inspection Accuracy

{inspection_accuracy:.4f}

Inspection Correct: {inspection_correct_count} / {count}

## Format

Format OK: {format_ok_count} / {count}

## Expected Match

Expected Match: {expected_match_count} / {count}

## OCR Failures

{ocr_failure_text}

## Inspection Failures

{inspection_failure_text}
"""


# --------------------------------------------------
# 15. Markdown 보고서 저장 및 터미널 출력
#
# 무엇을 하나요?
# - 평가 보고서를 저장할 폴더를 확인합니다.
# - 보고서 내용을 Markdown 파일로 저장합니다.
# - 보고서 전체 내용을 터미널에도 출력합니다.
# - 저장된 파일 경로를 출력합니다.
#
# METRIC_PATH.parent.mkdir():
# reports 폴더를 준비합니다.
#
# METRIC_PATH.write_text():
# Markdown 문자열을 파일로 저장합니다.
#
# encoding="utf-8":
# 한글을 포함한 내용을 UTF-8로 저장합니다.
#
# print(text):
# 전체 평가 결과를 터미널에 출력합니다.
#
# 저장 파일:
# reports/day06_ocr_metrics.md
#
# 왜 필요한가요?
# 전체 OCR 성능을 문서로 기록하여
# 이후 실패 분석과 Rule 성능 검토에 사용하기 위해서입니다.
#
# 주의:
# 같은 파일이 이미 존재한다면
# 기존 보고서 내용을 덮어씁니다.
#
# 다음 단계:
# 47_day06_make_challenge_variants.py
#
# 정상 SN 이미지를 Low Contrast, Blur,
# Tilt, Glare 상태로 변형하여
# OCR 검사기의 한계를 확인합니다.
# --------------------------------------------------

METRIC_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


METRIC_PATH.write_text(
    text,
    encoding="utf-8",
)


print(
    text
)

print(
    "Saved:",
    METRIC_PATH
)