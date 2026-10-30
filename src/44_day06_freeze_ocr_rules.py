# --------------------------------------------------
# 1. JSON 파일 처리 및 경로 설정에 필요한 라이브러리
#
# 무엇을 하나요?
# - JSON 파일을 읽고 저장할 기능을 불러옵니다.
# - 프로젝트 파일과 폴더 경로를 관리합니다.
#
# json:
# JSON 파일을 읽고 수정된 설정을 저장하는 라이브러리
#
# Path:
# 파일과 폴더 경로를 관리하는 클래스
#
# 왜 필요한가요?
# 앞에서 저장한 OCR Rule JSON을 읽은 뒤
# 실험 결과를 반영하여 다시 저장하기 위해서입니다.
# --------------------------------------------------

import json
from pathlib import Path


# --------------------------------------------------
# 2. 프로젝트 최상위 폴더와 OCR Rule JSON 경로 설정
#
# 무엇을 하나요?
# - 현재 Python 파일의 경로를 확인합니다.
# - 상위 폴더로 두 번 이동하여 프로젝트 Root를 찾습니다.
# - 수정할 OCR Rule JSON 파일의 경로를 지정합니다.
#
# __file__:
# 현재 Python 파일의 경로
#
# .resolve():
# 파일의 절대경로를 구합니다.
#
# .parent.parent:
# src 폴더의 상위인 프로젝트 Root로 이동합니다.
#
# CONFIG_PATH:
# OCR 검사 규칙을 저장할 JSON 파일 경로
#
# 프로젝트 구조:
#
# subject9_machinevision/
# ├── src/
# │   └── 44_day06_freeze_ocr_rules.py
# └── configs/
#     └── day06_ocr_rules.json
#
# 왜 필요한가요?
# 41번 실습에서 만든 기존 JSON 설정을
# 동일한 위치에서 읽고 수정하기 위해서입니다.
# --------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "day06_ocr_rules.json"
)


# --------------------------------------------------
# 3. 기존 OCR Rule JSON 파일 읽기
#
# 무엇을 하나요?
# - 41번에서 만든 day06_ocr_rules.json을 읽습니다.
# - JSON 문자열을 Python 딕셔너리로 변환합니다.
#
# CONFIG_PATH.read_text():
# JSON 파일에 저장된 내용을 문자열로 읽습니다.
#
# json.loads():
# JSON 문자열을 Python 딕셔너리로 변환합니다.
#
# rules:
# 기존 OCR 설정을 저장하는 딕셔너리
#
# 기존 JSON에는 다음 정보가 있습니다.
#
# baseline_version:
# 현재 Rule 버전
#
# reference_file:
# 대표 이미지 파일명
#
# roi_by_file:
# 제품 이미지별 ROI 좌표
#
# preprocess_mode:
# 사용할 전처리 방법
#
# ocr_psm / whitelist:
# Tesseract 실행 설정
#
# expected_pattern:
# SN 문자열 형식 검사 규칙
#
# 왜 필요한가요?
# 기존 ROI 좌표와 기본 설정을 유지하면서
# 필요한 항목만 수정하기 위해서입니다.
#
# 주의:
# JSON 파일이 없으면 FileNotFoundError가 발생합니다.
# 먼저 41번 실습을 완료해야 합니다.
# --------------------------------------------------

rules = json.loads(
    CONFIG_PATH.read_text(
        encoding="utf-8"
    )
)


# --------------------------------------------------
# 4. Baseline Rule 버전을 v1.0으로 확정
#
# 무엇을 하나요?
# - 기존 Rule 버전을 변경합니다.
#
# 변경 전:
# day06-ocr-draft
#
# 변경 후:
# day06-ocr-v1.0
#
# baseline_version:
# 현재 OCR 검사 기준의 버전을 기록하는 항목
#
# Draft:
# 아직 최종 결정되지 않은 실험용 설정
#
# v1.0:
# 이번 실습에서 처음 확정한 Baseline Rule
#
# 왜 필요한가요?
# 어떤 버전의 검사 기준을 사용했는지
# 이후 Batch 검사와 평가에서 확인하기 위해서입니다.
#
# 주의:
# 버전명을 변경한다고 파일 자체가
# 수정 불가능한 상태가 되는 것은 아닙니다.
# --------------------------------------------------

rules[
    "baseline_version"
] = "day06-ocr-v1.0"


# --------------------------------------------------
# 5. OCR 전처리 방법을 original로 확정
#
# 무엇을 하나요?
# - 43번 OCR 비교 실험에서 선택한
#   전처리 방법을 JSON에 기록합니다.
#
# 43번 실험에서 비교한 방법:
#
# original
# gray
# clahe
# otsu
# clahe_otsu
#
# 기존 교재의 대표 이미지 실험에서는
# Original, Gray, CLAHE가 정확하게 인식되었습니다.
#
# 따라서 복잡한 처리가 필요하지 않은
# original을 Baseline으로 선택합니다.
#
# original:
# SN ROI에 Gray, CLAHE, Otsu 등의
# 추가 전처리를 적용하지 않는 방법입니다.
#
# 왜 필요한가요?
# Batch 검사에서도 43번에서 선택한
# 동일한 전처리 방법을 사용하기 위해서입니다.
#
# 주의:
# 이 코드는 OCR 결과를 다시 비교하지 않습니다.
# 이미 선택한 전처리 방법을 기록할 뿐입니다.
# --------------------------------------------------

rules[
    "preprocess_mode"
] = "original"


# --------------------------------------------------
# 6. Tesseract PSM과 숫자 Whitelist 확정
#
# 무엇을 하나요?
# - Tesseract의 문자 배치 인식 방식을 설정합니다.
# - OCR에서 허용할 문자를 숫자로 제한합니다.
#
# ocr_psm = 7
#
# PSM(Page Segmentation Mode):
# Tesseract에게 입력 이미지의 문자 배치를
# 어떻게 해석할지 알려주는 설정입니다.
#
# PSM 7:
# 입력 이미지를 한 줄의 문자로 처리합니다.
#
# 이번 SN:
# 0637145699424
#
# 숫자가 한 줄로 인쇄되어 있으므로 PSM 7을 사용합니다.
#
# whitelist = 0123456789
#
# Whitelist:
# Tesseract가 인식할 수 있는 문자 종류를 제한합니다.
#
# 이번 실습에서는 숫자만 필요하므로
# 0부터 9까지만 허용합니다.
#
# 왜 필요한가요?
# SN의 문자 구성에 맞게 OCR 조건을 설정하여
# 이후 모든 제품 이미지에 동일하게 적용하기 위해서입니다.
# --------------------------------------------------

rules[
    "ocr_psm"
] = 7

rules[
    "whitelist"
] = "0123456789"


# --------------------------------------------------
# 7. SN 형식 검사 및 자동 문자 보정 여부 확정
#
# 무엇을 하나요?
# - OCR 문자열의 형식 검사 규칙을 정합니다.
# - OCR이 인식한 숫자를 임의로 변경하지 않도록 설정합니다.
#
# expected_pattern:
# 문자열 형식을 검사하기 위한 정규식(Regex)
#
# ^[0-9]{13}$
#
# ^:
# 문자열의 시작
#
# [0-9]:
# 숫자 0부터 9까지 중 한 개
#
# {13}:
# 숫자가 정확히 13개
#
# $:
# 문자열의 끝
#
# 즉 처음부터 끝까지
# 정확히 숫자 13자리인지 검사합니다.
#
# allow_context_correction = False
#
# Context Correction:
# OCR이 읽은 문자를 주변 정보나 기대값에 맞춰
# 자동으로 고치는 기능을 의미합니다.
#
# False:
# 이번 실습에서는 자동 문자 보정을 사용하지 않습니다.
#
# 예:
# OCR 결과 = 0637245697424
#
# Expected SN = 0637145699424
#
# 프로그램이 OCR 숫자를 임의로
# Expected SN으로 바꾸지 않습니다.
#
# 왜 필요한가요?
# OCR이 잘못 읽은 경우와
# 제품에 실제로 다른 번호가 인쇄된 경우를
# 구분할 수 있도록 하기 위해서입니다.
#
# 주의:
# 현재 코드는 이 설정을 JSON에 기록합니다.
# 실제 문자열 검사에서는 별도로 작성한
# day06_ocr_validation.py의 함수를 사용합니다.
# --------------------------------------------------

rules[
    "expected_pattern"
] = r"^[0-9]{13}$"

rules[
    "allow_context_correction"
] = False


# --------------------------------------------------
# 8. 확정된 OCR Rule을 기존 JSON 파일에 저장
#
# 무엇을 하나요?
# - 수정한 rules 딕셔너리를 JSON 문자열로 변환합니다.
# - 기존 day06_ocr_rules.json 파일에 다시 저장합니다.
#
# json.dumps():
# Python 딕셔너리를 JSON 문자열로 변환합니다.
#
# ensure_ascii=False:
# 한글을 읽을 수 있도록 저장합니다.
#
# indent=2:
# 들여쓰기를 적용하여 내용을 읽기 쉽게 만듭니다.
#
# write_text():
# 문자열을 파일에 저장합니다.
#
# 저장 파일:
# configs/day06_ocr_rules.json
#
# 중요한 점:
# 기존 ROI 좌표와 Reference 이미지 정보는
# 변경하지 않고 그대로 유지합니다.
#
# 변경되는 정보:
#
# baseline_version
# preprocess_mode
# ocr_psm
# whitelist
# expected_pattern
# allow_context_correction
#
# 왜 필요한가요?
# 이후 Batch 검사에서 동일한 OCR Rule을
# 불러와 사용할 수 있도록 하기 위해서입니다.
#
# 주의:
# 기존 JSON 파일에 덮어쓰기 방식으로 저장합니다.
# --------------------------------------------------

CONFIG_PATH.write_text(
    json.dumps(
        rules,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


# --------------------------------------------------
# 9. 최종 OCR Rule과 JSON 저장 경로 출력
#
# 무엇을 하나요?
# - Rule Freeze 완료 메시지를 출력합니다.
# - 확정된 설정값을 터미널에 표시합니다.
# - JSON 파일의 저장 경로를 출력합니다.
#
# 출력 항목:
#
# Version:
# 확정된 Baseline Rule 버전
#
# Preprocess:
# 사용할 전처리 방법
#
# PSM:
# Tesseract 문자 배치 설정
#
# Whitelist:
# OCR에서 허용할 문자
#
# Pattern:
# SN 문자열 형식 검사 정규식
#
# Context Correction:
# 자동 문자 보정 사용 여부
#
# Saved:
# JSON 파일 저장 경로
#
# 왜 필요한가요?
# 다음 Batch 실습을 시작하기 전에
# 검사 규칙이 올바르게 저장되었는지 확인하기 위해서입니다.
#
# 다음 실습:
# 45_day06_batch_ocr.py
#
# 제품 이미지 6장에 동일한 OCR Rule을 적용합니다.
# --------------------------------------------------

print(
    "Day06 OCR Rule Freeze"
)

print(
    "Version:",
    rules["baseline_version"]
)

print(
    "Preprocess:",
    rules["preprocess_mode"]
)

print(
    "PSM:",
    rules["ocr_psm"]
)

print(
    "Whitelist:",
    rules["whitelist"]
)

print(
    "Pattern:",
    rules["expected_pattern"]
)

print(
    "Context Correction:",
    rules["allow_context_correction"]
)

print(
    "Saved:",
    CONFIG_PATH
)