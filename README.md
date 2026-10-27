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
