# Day04 Mini Challenge

## 1. 검사 이미지와 조건

| 이미지 | 실제 촬영 조건 | 크기 (px) |
|---|---|---|
| shift.jpg | 제품 위치 이동 | 1346 × 584 |
| rotate.jpg | 제품 회전 | 861 × 417 |
| distance.jpg | 촬영 거리 변경 (로고가 작게 찍힘) | 387 × 157 |
| tilt_glare.jpg | 기울기 + 비닐 반사 | 1656 × 777 |

가장 크게 달라진 조건:
위치보다 **크기(촬영 거리)와 기울기·반사**가 크게 달라졌다.
Reference 로고(440 × 198)와 비교하면 challenge 이미지 속 로고는
최대 약 3배 크게, distance.jpg는 Reference보다 작게 찍혔다.

## 2. 선택한 방법

**SIFT + Homography**

## 3. 선택 이유

- 실제 촬영에서는 위치·회전·크기가 동시에 조금씩 달라진다.
- 앞의 실습에서 Template Matching은 회전·크기 변화에서 Score가 크게 떨어졌다.
  (rot15 0.475, scale90 0.624)
- SIFT는 회전·크기 변화에도 특징점이 유지되었고,
  Homography를 함께 쓰면 로고의 **실제 위치(사각형)**까지 확인할 수 있다.

## 4. 결과 (SIFT + Homography)

| 이미지 | Good Match | Inlier | Inlier Ratio | 결과 |
|---|---:|---:|---:|---|
| shift.jpg | 347 | 305 | 0.879 | FOUND, 로고 영역을 감쌈 |
| rotate.jpg | 389 | 360 | 0.925 | FOUND, 기울어진 사각형으로 로고를 감쌈 |
| distance.jpg | 365 | 343 | 0.940 | FOUND, 작은 이미지에서도 찾음 |
| tilt_glare.jpg | 294 | 212 | **0.721** | FOUND, 하지만 Inlier Ratio가 가장 낮음 |

Reference Keypoints: 675

## 5. 다른 방법과의 비교 (같은 이미지로 확인)

| 이미지 | Template Score | ORB Good | SIFT Good | SIFT+H Inlier Ratio |
|---|---:|---:|---:|---:|
| shift | 0.484 (위치 틀림) | 66 | 347 | 0.879 |
| rotate | 0.420 (위치 틀림) | 218 | 389 | 0.925 |
| distance | 비교 불가 | 226 | 365 | 0.940 |
| tilt_glare | 0.518 (위치 틀림) | **27** | 294 | 0.721 |

- **Template Matching:** 로고 크기가 Reference와 달라 세 장 모두 엉뚱한 위치(왼쪽 아래 배경)를 가리켰다.
  distance.jpg는 이미지가 Template보다 작아 `matchTemplate` 자체를 실행할 수 없었다.
- **ORB:** tilt_glare에서 Good Match가 27개로 크게 줄었다. 반사와 원근 변화에 약했다.
- **SIFT:** 네 장 모두 Good Match 290개 이상으로 가장 안정적이었다.

## 6. 관찰

- 크기가 달라지는 실제 촬영에서는 Template Matching의 Score가 낮아질 뿐 아니라 **위치 자체를 틀렸다.**
  Score만 보고 판단하면 안 되고, 찾은 위치가 실제 로고인지 이미지로 확인해야 한다.
- tilt_glare.jpg는 FOUND였지만 Inlier Ratio가 0.721로 다른 이미지(0.88~0.94)보다 낮았다.
  비닐 반사로 일부 특징점이 잘못 연결된 것으로 보인다.

## 7. 한계

- 반사가 더 강하거나 로고 일부가 가려지면 Inlier가 더 줄어들 수 있다.
- Challenge 이미지가 이미 로고 주변만 자른 사진이라,
  넓은 화면에서 로고를 찾는 경우는 아직 확인하지 않았다.
- 사각형이 볼록한지만 확인했고, 사각형의 크기·각도가 합리적인지는 아직 검사하지 않았다.

## 8. 실제 검사에 쓴다면 더 테스트할 조건

- 반사가 더 심한 사진, 로고 일부가 가려진 사진
- 로고가 없는 포장지 (오검출이 생기는지)
- 여러 장을 같은 조건으로 찍어 Inlier Ratio가 얼마나 흔들리는지

## 9. Threshold

Day05 Calibration 후 결정
(예: Inlier 수 또는 Inlier Ratio의 기준값을 정상 사진 여러 장으로 정한다)
