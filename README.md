# Subject 9 Machine Vision

OpenCV 기반 제조 머신비전 검사기술을 단계적으로 학습하는 프로젝트입니다.

---

## Day 01 — 머신비전 검사문제 설계와 ROI

1일차에는 완성된 OK / NG 검사기를 만드는 것이 아니라
이미지와 영상에서 **어디를 검사할 것인지 정하고,
정상과 불량의 차이를 어떻게 측정 가능한 특징으로 바꿀 것인지**
생각하는 과정을 학습했습니다.

### 핵심 흐름

```text
검사 문제 확인
    ↓
이미지 읽기
    ↓
ROI 설정
    ↓
여러 이미지에 같은 ROI 적용
    ↓
영상 Frame 처리
    ↓
영상 Frame에 ROI 적용
    ↓
스마트폰 실시간 영상 연결
    ↓
실시간 Frame에 ROI 적용
    ↓
새로운 검사 문제에 ROI 직접 적용
    ↓
결과와 판단 기록
```

### 주요 Python 파일

| 파일 | 역할 |
|---|---|
| `00_check_env.py` | Python · OpenCV · NumPy 환경 확인 |
| `01_read_image.py` | 이미지 한 장 읽기 |
| `02_image_info.py` | 이미지 크기와 채널 확인 |
| `03_roi_crop.py` | 이미지 한 장에서 ROI 설정·저장 |
| `04_batch_roi.py` | 여러 이미지에 동일한 ROI 적용 |
| `05_video_input.py` | 저장 영상을 Frame으로 읽고 일정 간격으로 저장 |
| `06_video_roi.py` | 저장 영상의 Frame에 ROI 적용·저장 |
| `07_webcam_input.py` | 스마트폰 IP Webcam 영상 입력 확인 |
| `08_phone_camera_roi.py` | 스마트폰 실시간 영상에서 ROI 선택·저장 |
| `03_roi_crop_challenge.py` | Mini Challenge: 주황 병뚜껑 ROI 설정 |
| `04_batch_roi_challenge.py` | Mini Challenge: 정상·불량 이미지에 같은 ROI 적용 |

### Mini Challenge — 주황색 병뚜껑 유무 검사

| 항목 | 내용 |
|---|---|
| 검사 대상 | 영양제 병의 주황색 병뚜껑 |
| 정상 | 병 입구에 뚜껑이 닫혀 있음 |
| 불량 | 뚜껑이 없어 투명한 병목이 보임 |
| ROI | `(x1, y1, x2, y2) = (240, 330, 730, 570)` |
| ROI 판단 | 고정 ROI 사용 가능 (카메라·병 위치를 고정한 조건) |
| 측정 후보 | ROI 안의 주황색 픽셀 비율 (HSV) |

- 처음 예시 좌표를 그대로 써서 ROI에 모니터만 들어왔고, Preview를 보고 뚜껑 위치로 수정했습니다.
- 같은 ROI로 정상(주황색 뚜껑)과 불량(투명한 병목)이 모두 잡혔습니다.
- 병 위치가 바뀌는 경우와 병 안 내용물의 색이 판정에 주는 영향은 아직 확인하지 않았습니다.

자세한 내용: `reports/day01_problem_definition.md`

### 배운 점

- 규칙 기반 머신비전에서는 **사람이** 정상·불량을 구분할 특징과 기준을 정한다.
- 이미지 전체가 아니라 **ROI**를 보면 검사와 관계없는 배경의 영향을 줄일 수 있다.
- OpenCV의 `image.shape`는 **(Height, Width, Channel)** 순서다.
- 한 이미지에서 맞던 ROI도 제품이나 카메라 위치가 바뀌면 어긋날 수 있다.
- 저장 영상과 실시간 영상도 결국 **Frame 단위**로 하나씩 처리된다.
- Calibration 데이터로 기준을 정하고, Test 데이터로는 확인만 한다. (`reports/dataset_roles.csv`)

### 결과물

| 경로 | 내용 |
|---|---|
| `outputs/roi/` | 샘플 이미지 ROI와 Preview |
| `outputs/frames/`, `outputs/video_roi/` | 영상 Frame과 Frame ROI |
| `outputs/live_roi/` | 스마트폰 실시간 ROI (`cap_on.jpg`, `cap_off.jpg`) |
| `outputs/day01_challenge/` | Mini Challenge ROI와 Preview |
| `reports/day01_problem_definition.md` | 검사 문제 정의 |
| `reports/dataset_roles.csv` | Dataset 폴더별 역할 |

> `samples/`, `data/`, `outputs/`는 `.gitignore`로 제외되어 GitHub에는 올라가지 않습니다.

---

## Day 02 — Image Enhancement와 검사 영상 품질 개선

1일차에 설정한 ROI를 이용하여
촬영조건과 영상 품질이 검사 결과에 어떤 영향을 주는지 확인했다.

### 실습 내용

- BGR / Gray / HSV 비교
- Brightness / Contrast
- Gamma Correction
- CLAHE
- Gaussian / Median / Bilateral Filter
- Sharpening
- Threshold / Otsu
- 전처리 전·후 객체 분리 결과 비교
- Mini Challenge를 통한 전처리 직접 선택

### 주요 Python 파일

| 파일 | 역할 |
|---|---|
| `09_color_space.py` | BGR · Gray · HSV 비교 |
| `10_brightness_contrast.py` | 밝기·대비 조절 |
| `11_gamma.py` | Gamma 보정 |
| `12_clahe.py` | 지역 대비 개선 (CLAHE) |
| `13_filter_compare.py` | Noise 제거 Filter 비교 |
| `14_sharpen.py` | 경계 강조 |
| `15_threshold_otsu.py` | 고정 Threshold와 Otsu 비교 |
| `16_preprocess_compare.py` | 전처리 전·후 종합 비교 |
| `17_day02_challenge.py` | Mini Challenge: 흰 병뚜껑 분리 |

### Mini Challenge — 배경과 밝기가 비슷한 흰 병뚜껑 분리

**문제 정의**
병뚜껑(Gray 약 129)보다 배경 벽(약 134)이 더 밝아,
Gray 기반 Threshold로는 병뚜껑과 배경이 함께 흰색이 되었다.

**시도했지만 효과가 없던 방법**

| 방법 | 결과 |
|---|---|
| Brightness (beta 0 / 50 / 80) | Otsu 결과가 원본과 완전히 같음 (임계값만 111 → 161 → 191로 이동) |
| Contrast / Gamma | 순서를 보존하는 변환이라 분리 개선 없음 |
| CLAHE | 처음으로 갈리기 시작했지만 Noise 조각 584 → 2,929 |
| Sharpening (3×3 Kernel) | Noise 조각 584 → 2,824 |

**해결**
HSV의 **S(채도) 채널**에서 병뚜껑 약 17, 배경 약 5로 차이가 났다.

```text
S > 9 (배경 제거) AND Gray > 118 (유리병 제거)
    ↓
Morphology → 덩어리 선별
    ↓
GrabCut (그림자 진 부분 보완)
    ↓
좌우 대칭 보완 → 윗면 / 링 분리
```

**한계**
- 링의 약 36%는 검출이 아니라 추정(대칭·Convex Hull)으로 채운 영역이며, 일부는 유리 부분까지 포함되었다.
- S 기준값을 9에서 8로만 바꿔도 배경이 섞일 만큼 민감하다.
- 근본적인 해결은 **검은 배경과 그림자 없는 조명으로 다시 촬영**하는 것이다.

### 핵심 결과

- `outputs/day02/compare/preprocess_compare.png`
- `outputs/day02/challenge/challenge_result.jpg`
- `reports/day02_parameter_table.csv`
- `reports/day02_preprocess_reason.md`

### 오늘의 핵심

전처리 기술을 먼저 선택하는 것이 아니라

문제 확인
→ 필요한 전처리 선택
→ Parameter 비교
→ Threshold / Otsu 결과 확인
→ 사용 여부 결정

순서로 판단한다.

전처리는 없는 정보를 만들어 내지 못한다.
밝기에 차이가 없으면 다른 채널(색상·채도)을 찾거나 촬영환경을 개선해야 한다.

### 다음 학습

Day 03에서는 Binary 이미지에
Morphology와 Contour를 적용하여
객체의 면적·개수·위치를 측정하고
Rule 기반 OK / NG 판정으로 연결한다.

---

## Day 03 — Morphology·Contour와 규칙 기반 판정

1일차에서 실제 촬영한 ROI 이미지와
2일차에서 학습한 Threshold / Otsu 방식을 이어서 사용하였다.
Binary Image를 Morphology로 정리하고,
Contour로 객체의 특징을 숫자로 측정한 뒤
실제 측정값을 근거로 Rule 기반 OK / NG 판정을 만들었다.

### 실습 흐름

```text
ROI Image → Gray → Otsu → Binary → Morphology → Contour
→ Feature 측정 → cap_on / cap_off 비교 → Candidate Rule → OK / NG
→ 새 촬영 이미지 Batch Test → 결과 분석 → Mini Challenge
```

### 주요 실습

- Opening / Closing 역할 확인, Kernel 3×3 · 5×5 · 9×9 비교
- Area Ratio / Count / BBox / Center 측정 (`cap_on` 0.661 vs `cap_off` 0.459)
- 두 값의 가운데로 Candidate Threshold (0.560) 생성
- 640×480으로 새로 찍은 6장으로 Batch Test (기준 0.361) → 4/4 일치

### 분석 결과

- 4/4 일치였지만 `cap_off_003`은 기준과 0.044 차이로 아슬아슬했다.
- 면적 비율은 ROI 크기에 따라 달라진다. (`cap_off_003`은 ROI가 가장 작아 비율이 커짐)
- 2일차 기준 이미지로 만든 기준값 0.560을 Batch 이미지에 적용하면 뚜껑 있는 3장 중 2장이 틀린다.
  → 기준값은 촬영 조건에 묶여 있으며, 조건이 바뀌면 다시 Calibration해야 한다.
- 덩어리 개수는 촬영 조건에 따라 대소 관계가 뒤집혀 단독 Feature로 쓰기 어렵다.

### Mini Challenge — 주황색 병뚜껑 존재 검사

| 항목 | 내용 |
|---|---|
| 처리 | ROI → HSV → 주황색 Mask → Morphology → Contour |
| Feature | `orange_ratio` (ROI 안 주황색 비율) |
| 측정값 | 정상 0.430 ~ 0.437 / 불량 0.0007 ~ 0.0012 |
| Rule | `orange_ratio >= 0.2159` → OK |
| Test | 3장 모두 일치, 가장 작은 Margin 0.214 |

색상이 뚜렷한 물체는 Gray 면적보다 HSV 색상 Mask가 훨씬 안정적이었다. (Margin 약 5배)

### 확장 실습

검사 로직을 `src/cap_inspector.py`로 분리하고,
스마트폰 IP Webcam 실시간 영상을 FastAPI 웹 화면에서 검사하도록 연결했다.
(`src/fastapi_cap_app.py`, `web/cap_inspection.html`, `tests/test_inspector.py`)

### 주요 결과 파일

- reports/day03_reference_features.csv
- reports/day03_inspection_results.csv
- reports/day03_challenge_results.csv
- reports/day03_rule_analysis.md

---

## Day 04 — Template · Feature Matching

### 학습 내용

광천김 포장지의 로고 영역을 기준으로 다음 방법을 실습했다.

- GrabCut, MeanShift
- Template Matching
- ORB, SIFT
- RANSAC, Homography

### 비교 데이터

normal / shift / rot15 / scale90 (full_front.jpg를 코드로 변형)

### 주요 결과

| 방법 | 결과 |
|---|---|
| Template Matching | 기준 상태에서는 잘 찾았지만 회전·크기 변화에서 Score 감소 |
| ORB | 위치·회전·크기 변화에서도 Matching이 어느 정도 유지 |
| SIFT | 회전·크기 변화에서도 많은 Matching 유지 |
| SIFT + Homography | 특징점 위치 관계로 로고 위치를 사각형으로 찾음 |

### Mini Challenge — 실제 다시 촬영한 이미지

선택한 방법: **SIFT + Homography**

| 이미지 | Good Match | Inlier | Inlier Ratio | 결과 |
|---|---:|---:|---:|---|
| shift | 347 | 305 | 0.879 | FOUND |
| rotate | 389 | 360 | 0.925 | FOUND |
| distance | 365 | 343 | 0.940 | FOUND |
| tilt_glare | 294 | 212 | 0.721 | FOUND (반사로 가장 불안정) |

- 같은 이미지에서 Template Matching은 로고 크기가 달라 3장 모두 엉뚱한 위치를 가리켰고,
  distance.jpg는 Template보다 작아 실행할 수 없었다.
- ORB는 tilt_glare에서 Good Match가 27개로 크게 줄었다.
- tilt_glare의 Homography 사각형은 사다리꼴로 그려져 포장지 기울기(원근)를 반영했다.

### 주요 결과 파일

- reports/day04_template_results.csv
- reports/day04_orb_results.csv
- reports/day04_sift_results.csv
- reports/day04_homography_results.csv
- reports/day04_challenge_results.csv
- reports/day04_method_selection.md

---

## Day 05 — Calibration · Rule Freeze · Test

### 검사 대상

광천김 포장지의 기준 로고 존재 및 정합 검사

### 사용한 기술

Template Matching, SIFT, RANSAC, Homography

### Calibration

- OK 6장 / NG 6장
- 측정값: Template Score, SIFT Good Match, Homography Inlier, Inlier Ratio

### Baseline Rule

- Homography Found: True
- SIFT Good Match >= 129
- Inlier Ratio >= 0.8664

### Test

- Test 이미지: 6장
- Accuracy: 0.5000
- TP: 3 / TN: 0 / FP: 3 / FN: 0
- NG Precision: 0.5000 / NG Recall: 1.0000
- FP 3건: 정상 이미지를 모두 NG로 판정 (Baseline Rule이 정상 Test 이미지에 너무 엄격함)

### Mini Challenge — LOTTE 드림카카오 72% 라벨

| 항목 | 내용 |
|---|---|
| Calibration | OK 3장 / NG 3장 |
| Feature 비교 | 네 Feature 모두 OK·NG 범위가 겹침 (gap 음수) |
| Rule | Homography 성공 AND Inlier Ratio >= 0.6543 |
| Test | TP 2 / TN 1 / FP 1 / FN 0 (Accuracy 0.75) |

- FP: 라벨 아래쪽을 가린 NG가 OK로 통과 (SIFT 83, Inlier Ratio 0.687).
  위쪽 글자만으로 특징점이 충분해 Homography가 로고 전체 위치를 추정했다.
- 특징점 개수·비율은 로고 일부만 가린 불량을 표현하기 어렵다.
- Calibration의 `ok_03`은 멀리서 위에서 내려다보며 찍어 정상인데도 점수가 낮았다.
  → 촬영 거리·각도를 고정해야 한다.

### 핵심 결과 파일

- reports/day05_calibration_measurements.csv
- reports/day05_rule_candidates.csv
- configs/day05_logo_rules.json
- reports/day05_test_results.csv
- reports/day05_test_metrics.md
- reports/day05_mini_challenge.md

### 오늘의 핵심

Calibration 데이터로 판정 기준을 정하고, 기준을 동결한 뒤 처음 보는 Test 데이터에 적용했다.
Accuracy뿐 아니라 FP와 FN을 확인하고, 틀린 결과의 원인을 기록했다.

---

## Day 06 — OCR 기반 제조 문자 검사

### 검사 대상

제품에 인쇄된 13자리 SN
(Mini Challenge: Ch.-B. 제품 코드)

### 사용 기술

OpenCV ROI, Gray / CLAHE / Otsu 비교, Tesseract OCR, pytesseract,
PSM, Whitelist, Text Cleaning, Regex, Expected SN 비교, Deskew

### 최종 OCR Rule

| 항목 | 값 |
|---|---|
| Version | day06-ocr-v1.0 |
| Preprocess | original |
| PSM | 7 |
| Whitelist | 0123456789 |
| Pattern | `^[0-9]{13}$` |

전처리 비교에서 Otsu는 `1`을 `2`로 잘못 읽었고, CLAHE+Otsu는 아무것도 읽지 못해 Original을 선택했다.

### Batch 검사 결과

- 검사 이미지: 6장 (OK 4장, 교육용 NG 2장)
- OCR Exact Accuracy: 100% (6/6)
- Inspection Accuracy: 100% (6/6)
- OCR Failure / Inspection Failure: 없음

### 환경 변화 테스트

| 조건 | OCR 결과 | 결과 |
|---|---|---|
| Low Contrast | `0637145699424` | 성공 |
| Blur | 빈 값 | 실패 |
| Tilt | `2` | 실패 |
| Tilt + Deskew | `4` | 실패 |
| Glare | 빈 값 | 실패 |

### Mini Challenge

검사 대상: Ch.-B.
OCR 성공 3 / 4. 흐린 이미지에서 D를 0으로 읽은 1장이 실패했다.
교재 Regex는 이 오류를 통과시켰지만, 실제 모양에 맞춘 엄격한 Regex는 잡아냈다.

### 주요 결과 파일

- configs/day06_ocr_rules.json
- reports/day06_preprocess_compare.csv
- reports/day06_ocr_results.csv
- reports/day06_ocr_metrics.md
- reports/day06_challenge_results.csv
- reports/day06_failure_analysis.md
- reports/day06_mini_challenge_results.csv
- reports/day06_mini_challenge.md

### 오늘 확인한 핵심

제품 이미지에서 필요한 문자 영역을 ROI로 선택하고,
OCR로 읽은 문자열을 형식과 Expected Value로 검사하여 OK / NG 판정으로 연결했다.
OCR이 정확하게 문자를 읽는 것과 제품 자체가 정상인 것은 서로 다른 문제이며,
영상 품질이 바뀌면 정상에서 100%였던 OCR도 쉽게 실패한다는 것을 확인했다.

---
