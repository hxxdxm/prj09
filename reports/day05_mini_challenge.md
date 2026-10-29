# Day05 Mini Challenge

## 검사 대상
LOTTE 드림카카오 72% 통의 노란 라벨 (DREAM CACAO 72%)

## OK 기준
라벨 로고 전체가 가려지지 않고 보인다.

## NG 기준
라벨 로고의 일부 또는 대부분이 가려져 보이지 않는다.

## Calibration 데이터
- OK: 3장
- NG: 3장

## 확인한 Feature
- Feature 1: SIFT Good Match
- Feature 2: Inlier Ratio
- (함께 비교: Template Score, Homography Inlier)

## Calibration 결과
| Feature | OK 범위 | NG 범위 | gap (OK 최솟값 - NG 최댓값) |
|---|---|---|---:|
| Template Score | 0.337 ~ 0.977 | 0.343 ~ 0.426 | -0.090 |
| SIFT Good Match | 101 ~ 164 | 44 ~ 124 | -23 |
| Homography Inlier | 57 ~ 129 | 16 ~ 81 | -24 |
| Inlier Ratio | 0.564 ~ 0.787 | 0.364 ~ 0.653 | -0.089 |

### OK
ok_01은 Template 0.977, SIFT 164로 가장 높았다. (Reference를 이 사진에서 잘라 자기 자신과 비교한 것으로 보임)
ok_03은 SIFT 101, Inlier Ratio 0.564로 정상인데도 낮았다.
ok_03은 다른 OK 사진보다 멀리서, 위에서 내려다보는 각도로 찍어
라벨이 작고 비스듬하게 나왔다.
그래서 정상인데도 SIFT 101, Inlier Ratio 0.564로 가장 낮았고,
Calibration에서 유일하게 NG로 잘못 판정되었다(FN).

### NG
ng_02(SIFT 44, Ratio 0.364)는 낮았지만,
ng_01(SIFT 114, Ratio 0.640)과 ng_03(SIFT 124, Ratio 0.653)은 OK인 ok_03보다 높았다.
로고 일부만 가려져 남은 부분에서 특징점이 충분히 나온 것으로 보인다.

→ 네 Feature 모두 gap이 음수로, Calibration에서 OK와 NG를 완전히 나눌 수 없었다.

## 선택한 Rule
Homography 계산 성공
AND
Inlier Ratio >= 0.6543

Calibration 결과: FP 0, FN 1 (ok_03), margin 0.003

## Test 결과
- TP: 1   (오른쪽을 가린 NG를 NG로 잡음)
- TN: 2   (정상 2장을 OK로)
- FP: 0
- FN: 1   (아래쪽을 가린 NG를 OK로 통과시킴)
- Accuracy: 0.75
- NG Recall: 0.5 (실제 NG 2장 중 1장만 잡음)

## 가장 눈에 띄는 결과
파일: test_ng_02.jpg
Actual: NG (라벨 아래쪽 72% 부분을 종이로 가림)
Predicted: OK
측정값: SIFT 83, Inlier 57, Inlier Ratio 0.687

왜 이런 결과가 나왔다고 생각하는가?
가려지지 않은 위쪽 DREAM CACAO 글자에서 특징점이 충분히 연결되어 Homography가 계산되었다.
Homography는 보이는 부분의 관계로 로고 전체 위치를 추정하기 때문에, 결과 사각형이 가려진 종이 위까지 그려졌다.
Inlier Ratio 0.687은 기준 0.6543보다 높았다.
기준값이 Calibration NG(ng_03 0.653)에 거의 붙어 있어(margin 0.003), 그보다 조금만 높은 NG도 통과할 수 있는 Rule이었다.

## 한 줄 정리
SIFT·Inlier Ratio는 "로고가 보이는가"는 판단할 수 있지만,
로고 일부만 가린 불량은 남은 부분의 특징점 때문에 정상과 구분하기 어려웠다.

## 다음에 개선할 점 (Rule은 Test 후 수정하지 않음)
- Reference를 Calibration·Test에 쓰지 않는 별도 사진에서 자른다.
- Calibration NG를 완전 가림 / 일부 가림으로 나누어 더 준비한다.
- 로고 영역을 부분별로 나누어 각 부분이 보이는지 확인하는 Feature를 검토한다.
- 모든 사진을 같은 거리·같은 높이(정면)에서 찍도록 카메라 위치를 고정한다.