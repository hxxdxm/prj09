from pathlib import Path

import cv2
import numpy as np


# ==================================================
# challenge.py  (Day02 Mini Challenge)
#
# 문제 정의
#   병뚜껑(Gray 약 129)보다 배경 벽(약 134)이 더 밝아
#   밝기·대비·CLAHE·Sharpening으로는 분리되지 않는다.
#
# 해결 아이디어
#   [1차] S(채도)와 Gray 조건을 AND로 묶어
#         "확실한 병뚜껑" 영역을 찾는다.
#         → 그림자 진 왼쪽은 채도가 낮아 빠진다.
#   [2차] 1차 결과를 힌트로 GrabCut을 실행해
#         빠진 윗면 왼쪽까지 채운다.
#   [3차] 병뚜껑은 좌우 대칭이므로, 윗면 가운데를 기준으로
#         Mask를 뒤집어 합친 뒤, 윗면과 링을 따로 나눠
#         각각 Convex Hull로 다듬는다.
# ==================================================


# --------------------------------------------------
# 1. 입력 이미지와 결과 폴더를 정한다.
# --------------------------------------------------

image_path = Path(
    "samples/day02/challenge.jpg"
)

output_dir = Path(
    "outputs/day02/challenge"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 2. Parameter를 한곳에 모아 둔다.
# --------------------------------------------------

BLUR_SIZE = (21, 21)     # S·Gray의 블록 Noise 제거 (홀수)
S_THRESHOLD = 9          # S가 이보다 크면 병뚜껑 후보 (8이면 배경 섞임)
GRAY_THRESHOLD = 118     # Gray가 이보다 크면 병뚜껑 후보 (유리병 제거)
KERNEL_SIZE = (15, 15)   # Morphology 크기
MIN_AREA_RATIO = 0.02    # 이미지의 2%보다 작은 덩어리는 버림
GRABCUT_PAD = 120        # GrabCut이 찾아볼 범위 (힌트 영역 바깥 여유)
GRABCUT_ITER = 10         # GrabCut 반복 횟수


# --------------------------------------------------
# 3. 이미지를 읽고 정상적으로 읽혔는지 확인한다.
# --------------------------------------------------

image = cv2.imread(
    str(image_path)
)

if image is None:
    raise FileNotFoundError(
        f"이미지를 읽지 못했습니다: {image_path}"
    )

height, width = image.shape[:2]
image_area = height * width


# --------------------------------------------------
# 4. [비교 기준] 전처리 없이 Gray + Otsu
# --------------------------------------------------

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY,
)

otsu_value, original_otsu = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
)

print(f"[비교] 원본 Gray Otsu 임계값: {otsu_value:.0f}")


# --------------------------------------------------
# 5. [1차] 조건 1: S 채널로 배경을 제거한다.
# --------------------------------------------------

hsv = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2HSV,
)

s_blur = cv2.GaussianBlur(
    hsv[:, :, 1],
    BLUR_SIZE,
    0,
)

_, s_mask = cv2.threshold(
    s_blur,
    S_THRESHOLD,
    255,
    cv2.THRESH_BINARY,
)


# --------------------------------------------------
# 6. [1차] 조건 2: Gray로 유리병을 제거한다.
# --------------------------------------------------

gray_blur = cv2.GaussianBlur(
    gray,
    BLUR_SIZE,
    0,
)

_, gray_mask = cv2.threshold(
    gray_blur,
    GRAY_THRESHOLD,
    255,
    cv2.THRESH_BINARY,
)


# --------------------------------------------------
# 7. [1차] 두 조건을 AND로 합치고 Morphology로 정리한다.
# --------------------------------------------------

combined = cv2.bitwise_and(
    s_mask,
    gray_mask,
)

kernel = np.ones(
    KERNEL_SIZE,
    np.uint8,
)

combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)


# --------------------------------------------------
# 8. [1차] 병뚜껑 덩어리만 고른다.
#
# 버리는 덩어리
#   - 너무 작은 것  : 배경의 작은 점
#   - 테두리에 닿는 것: 배경 구석, 유리병 아래쪽
#
# 남는 것: 병뚜껑 윗면 + 아래 링 (그림자 때문에 둘로 나뉨)
# --------------------------------------------------

def touches_border(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2


contours, _ = cv2.findContours(
    combined,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

cap_parts = [
    c for c in contours
    if cv2.contourArea(c) > image_area * MIN_AREA_RATIO
    and not touches_border(c)
]

if not cap_parts:
    raise RuntimeError(
        "병뚜껑 후보를 찾지 못했습니다. Threshold 값을 확인하세요."
    )

first_mask = np.zeros_like(combined)

cv2.drawContours(
    first_mask,
    cap_parts,
    -1,
    255,
    cv2.FILLED,
)

print(f"[1차] 병뚜껑 덩어리 수: {len(cap_parts)}")
print(f"[1차] 병뚜껑 면적 비율: {(first_mask > 0).mean() * 100:.1f}%")


# --------------------------------------------------
# 9. [2차] GrabCut에 줄 힌트 지도를 만든다.
#
# GrabCut은 "여기는 확실히 병뚜껑, 여기는 확실히 배경"이라는
# 힌트를 주면, 색 분포와 경계를 보고 나머지를 스스로 판단한다.
#
#   GC_BGD     : 확실한 배경   → 찾아볼 범위 바깥
#   GC_PR_BGD  : 아마 배경     → 찾아볼 범위 안쪽
#   GC_PR_FGD  : 아마 병뚜껑   → 1차 결과를 감싼 영역 (Convex Hull)
#   GC_FGD     : 확실한 병뚜껑 → 1차 결과를 조금 깎은 안쪽
# --------------------------------------------------

grabcut_mask = np.full(
    (height, width),
    cv2.GC_BGD,
    np.uint8,
)

x, y, w, h = cv2.boundingRect(np.vstack(cap_parts))

x1 = max(0, x - GRABCUT_PAD)
y1 = max(0, y - 20)
x2 = min(width, x + w + GRABCUT_PAD)
y2 = min(height, y + h + 20)

grabcut_mask[y1:y2, x1:x2] = cv2.GC_PR_BGD

hull = cv2.convexHull(np.vstack(cap_parts))
hull_mask = np.zeros_like(first_mask)
cv2.drawContours(hull_mask, [hull], -1, 255, cv2.FILLED)
grabcut_mask[hull_mask > 0] = cv2.GC_PR_FGD

sure_cap = cv2.erode(first_mask, np.ones((25, 25), np.uint8))
grabcut_mask[sure_cap > 0] = cv2.GC_FGD


# --------------------------------------------------
# 10. [2차] GrabCut을 실행한다.
# --------------------------------------------------

# GrabCut은 내부에서 무작위 값을 사용하므로
# Seed를 고정해 실행할 때마다 같은 결과가 나오게 한다.
cv2.setRNGSeed(0)

bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

cv2.grabCut(
    image,
    grabcut_mask,
    None,
    bgd_model,
    fgd_model,
    GRABCUT_ITER,
    cv2.GC_INIT_WITH_MASK,
)

# "확실한 병뚜껑" 또는 "아마 병뚜껑"으로 판단된 픽셀만 흰색
result_mask = np.where(
    (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
    255,
    0,
).astype(np.uint8)

result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_OPEN, kernel)


# --------------------------------------------------
# 11. GrabCut 결과에서 병뚜껑 덩어리만 남긴다.
#
# 병뚜껑 윗면과 아래 링이 그림자 때문에 두 덩어리로 나온다.
# --------------------------------------------------

grabcut_contours, _ = cv2.findContours(
    result_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

grabcut_parts = [
    c for c in grabcut_contours
    if cv2.contourArea(c) > image_area * MIN_AREA_RATIO
]

grabcut_mask_clean = np.zeros_like(result_mask)
cv2.drawContours(grabcut_mask_clean, grabcut_parts, -1, 255, cv2.FILLED)

print(f"[2차] GrabCut 덩어리 수: {len(grabcut_parts)}")
print(f"[2차] GrabCut 면적 비율: {(grabcut_mask_clean > 0).mean() * 100:.1f}%")


# --------------------------------------------------
# 12. [3차] 좌우 대칭으로 빠진 부분을 보완한다.
#
# 규칙: 병뚜껑은 좌우 대칭이다.
#
# GrabCut 후에도 아래 링의 왼쪽 끝은 그림자 때문에 빠진다.
# 온전하게 잡힌 윗면의 가운데를 기준선으로 삼아
# Mask를 좌우로 뒤집은 뒤 원래 Mask와 합친다(OR).
# --------------------------------------------------

top_part = min(
    grabcut_parts,
    key=lambda c: cv2.boundingRect(c)[1],   # 가장 위에 있는 덩어리 = 윗면
)

top_x, _, top_w, _ = cv2.boundingRect(top_part)
center_x = top_x + top_w / 2

# x → 2 * center_x - x 로 옮기는 좌우 반전 변환
flip_matrix = np.float32([
    [-1, 0, 2 * center_x],
    [0, 1, 0],
])

flipped = cv2.warpAffine(
    grabcut_mask_clean,
    flip_matrix,
    (width, height),
)

symmetric_mask = cv2.bitwise_or(
    grabcut_mask_clean,
    flipped,
)

print(f"[3차] 대칭 기준선 x = {center_x:.0f}")


# --------------------------------------------------
# 13. [3차] 윗면과 링을 따로 나눠 각각 Convex Hull로 다듬는다.
#
# 대칭 보완 후에도 윗면과 링 사이에는 그림자 틈이 있어
# 두 덩어리로 나뉜다. 위에 있는 것이 윗면, 아래가 링이다.
# --------------------------------------------------

symmetric_contours, _ = cv2.findContours(
    symmetric_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

symmetric_parts = [
    c for c in symmetric_contours
    if cv2.contourArea(c) > image_area * MIN_AREA_RATIO
]

if len(symmetric_parts) < 2:
    raise RuntimeError(
        "윗면과 링이 두 덩어리로 나뉘지 않았습니다. "
        "Morphology 크기(KERNEL_SIZE)를 줄여 보세요."
    )

# 위쪽(y가 작은) 순서로 정렬 → [윗면, 링]
symmetric_parts = sorted(
    symmetric_parts,
    key=lambda c: cv2.boundingRect(c)[1],
)

top_hull = cv2.convexHull(symmetric_parts[0])
ring_hull = cv2.convexHull(symmetric_parts[1])


def filled_mask(contour):
    target = np.zeros((height, width), np.uint8)
    cv2.drawContours(target, [contour], -1, 255, cv2.FILLED)
    return target


top_mask = filled_mask(top_hull)
ring_mask = filled_mask(ring_hull)

# 두 Hull이 살짝 겹치면 겹친 부분은 윗면으로 본다.
ring_mask[top_mask > 0] = 0

# 전체 병뚜껑 Mask = 윗면 + 링
result_mask = cv2.bitwise_or(top_mask, ring_mask)

TOP_COLOR = (0, 255, 0)      # 윗면: 초록
RING_COLOR = (255, 128, 0)   # 링  : 파랑

result = image.copy()
cv2.drawContours(result, [top_hull], -1, TOP_COLOR, 4)
cv2.drawContours(result, [ring_hull], -1, RING_COLOR, 4)

cv2.putText(result, "TOP", tuple(top_hull[top_hull[:, :, 1].argmin()][0] + [0, 50]),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, TOP_COLOR, 3)
cv2.putText(result, "RING", tuple(ring_hull[ring_hull[:, :, 0].argmin()][0] + [20, 10]),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, RING_COLOR, 3)

top_area = (top_mask > 0).sum()
ring_area = (ring_mask > 0).sum()

print(f"[최종] 윗면 면적: {top_area} ({top_area / image_area * 100:.1f}%)")
print(f"[최종] 링   면적: {ring_area} ({ring_area / image_area * 100:.1f}%)")


# --------------------------------------------------
# 14. 결과를 저장한다.
# --------------------------------------------------

cv2.imwrite(str(output_dir / "challenge_original_otsu.jpg"), original_otsu)
cv2.imwrite(str(output_dir / "challenge_first_mask.jpg"), first_mask)
cv2.imwrite(str(output_dir / "challenge_grabcut_mask.jpg"), grabcut_mask_clean)
cv2.imwrite(str(output_dir / "challenge_result_mask.jpg"), result_mask)
cv2.imwrite(str(output_dir / "challenge_top_mask.jpg"), top_mask)
cv2.imwrite(str(output_dir / "challenge_ring_mask.jpg"), ring_mask)
cv2.imwrite(str(output_dir / "challenge_result.jpg"), result)


# --------------------------------------------------
# 15. [확인용] 원본 | Mask | 겹쳐 보기 를 나란히 저장한다.
# --------------------------------------------------

def make_side_view(source, target_mask):

    mask_bgr = cv2.cvtColor(target_mask, cv2.COLOR_GRAY2BGR)

    painted = source.copy()
    painted[target_mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(source, 0.6, painted, 0.4, 0)

    side = cv2.hconcat([source, mask_bgr, overlay])

    return cv2.resize(side, None, fx=0.5, fy=0.5)


side_views = {
    "challenge_s_side.jpg": s_mask,
    "challenge_gray_side.jpg": gray_mask,
    "challenge_first_side.jpg": first_mask,
    "challenge_grabcut_side.jpg": grabcut_mask_clean,
    "challenge_result_side.jpg": result_mask,
}

for filename, target_mask in side_views.items():
    cv2.imwrite(
        str(output_dir / filename),
        make_side_view(image, target_mask),
    )


# 윗면(초록)과 링(파랑)을 다른 색으로 겹쳐 본다.
parts_mask_view = np.zeros_like(image)
parts_mask_view[top_mask > 0] = TOP_COLOR
parts_mask_view[ring_mask > 0] = RING_COLOR

parts_overlay = cv2.addWeighted(image, 0.6, parts_mask_view, 0.4, 0)
parts_overlay[(top_mask == 0) & (ring_mask == 0)] = image[(top_mask == 0) & (ring_mask == 0)]

parts_side = cv2.hconcat([image, parts_mask_view, parts_overlay])
parts_side = cv2.resize(parts_side, None, fx=0.5, fy=0.5)

cv2.imwrite(str(output_dir / "challenge_parts_side.jpg"), parts_side)


# --------------------------------------------------
# 16. 단계별 비교 이미지를 한 장으로 저장한다.
#
# 윗줄 : 원본 | 원본 Otsu (전처리 전) | 1차 결과 (S AND Gray)
# 아랫줄: 2차 결과 (GrabCut) | 3차 (좌우 대칭 보완) | 최종 결과 (윗면/링)
# --------------------------------------------------

views = [
    ("original", image),
    ("before: gray otsu", original_otsu),
    ("1st: S AND gray", first_mask),
    ("2nd: grabcut", grabcut_mask_clean),
    ("3rd: + mirror", symmetric_mask),
    ("final: top / ring", result),
]

tiles = []

for name, view in views:

    if len(view.shape) == 2:
        view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)

    tile = cv2.resize(view, None, fx=0.4, fy=0.4)

    cv2.putText(
        tile, name, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
    )

    tiles.append(tile)

compare = cv2.vconcat([
    cv2.hconcat(tiles[:3]),
    cv2.hconcat(tiles[3:]),
])

compare_path = output_dir / "challenge_compare.jpg"

cv2.imwrite(str(compare_path), compare)

print()
print("입력 :", image_path)
print("비교 :", compare_path)
print("Challenge 완료")