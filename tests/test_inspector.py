
# ============================================================
# ① 테스트에 필요한 라이브러리 불러오기
# - OpenCV, NumPy : 가상 이미지 생성 및 검사
# - TestClient    : 실제 브라우저 없이 FastAPI 요청 테스트
# - cap_inspector : 앞에서 만든 병뚜껑 검사 함수
# - app, state    : FastAPI 애플리케이션과 검사 상태
# ============================================================

from pathlib import Path
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.cap_inspector import (
    inspect_roi,
    judge_cap,
    load_rule,
    measure_area_ratio
)
from src.fastapi_cap_app import app, state


# ============================================================
# ② 테스트용 가상 병뚜껑 이미지 생성
# - 실제 사진 대신 사각형 패턴 이미지를 만듭니다.
# - cap_on=True  : 큰 검은색 사각형
# - cap_on=False : 작은 검은색 사각형
# - 실제 병뚜껑 판정 정확도를 평가하는 이미지는 아닙니다.
# ============================================================

def synthetic_roi(cap_on: bool) -> np.ndarray:
    """실제 사진 없이 검사 규칙을 확인하는 인공 패턴."""

    image = np.full((160, 200, 3), 235, np.uint8)

    if cap_on:
        cv2.rectangle(
            image, (35, 25), (165, 105), (15, 15, 15), -1
        )
    else:
        cv2.rectangle(
            image, (65, 90), (135, 120), (15, 15, 15), -1
        )

    return image


# ============================================================
# ③ 검사 기준값 생성 테스트 준비
# - 가상 cap_on / cap_off 이미지를 JPG로 저장합니다.
# - load_rule()을 이용하여 판정 기준값을 만듭니다.
# - tmp_path는 pytest가 제공하는 임시 폴더입니다.
# ============================================================

def test_rule(tmp_path: Path):

    cv2.imwrite(
        str(tmp_path / 'cap_on.jpg'),
        synthetic_roi(True)
    )

    cv2.imwrite(
        str(tmp_path / 'cap_off.jpg'),
        synthetic_roi(False)
    )

    rule = load_rule(tmp_path)


    # ========================================================
    # ④ OpenCV 검사 로직 테스트
    # - cap_on은 OK, cap_off는 NG가 나와야 합니다.
    # - 두 이미지의 Area Ratio와 기준값을 비교합니다.
    # - assert는 예상 결과와 다르면 테스트를 실패시킵니다.
    # ========================================================

    on = inspect_roi(synthetic_roi(True), rule)
    off = inspect_roi(synthetic_roi(False), rule)

    assert on['decision'] == 'OK'
    assert off['decision'] == 'NG'

    assert (
        rule.cap_on_ratio
        > rule.threshold
        > rule.cap_off_ratio
    )

    assert (
        measure_area_ratio(synthetic_roi(True))[0]
        > measure_area_ratio(synthetic_roi(False))[0]
    )

    assert judge_cap(rule.threshold, rule)[0] == 'OK'


# ============================================================
# ⑤ FastAPI 통합 테스트 준비
# - 테스트용 이미지 2장으로 가상 판정 기준을 만듭니다.
# - 이후 FastAPI에서도 이 기준을 사용합니다.
# ============================================================

def test_api_and_fake_camera(tmp_path: Path):

    cv2.imwrite(
        str(tmp_path / 'cap_on.jpg'),
        synthetic_roi(True)
    )

    cv2.imwrite(
        str(tmp_path / 'cap_off.jpg'),
        synthetic_roi(False)
    )

    fake_rule = load_rule(tmp_path)


    # ========================================================
    # ⑥ 실제 카메라 대신 사용할 준비
    # - FastAPI 서버 모듈을 불러옵니다.
    # - 기존 VideoCapture 함수를 보관합니다.
    # - 테스트 시작 시 cap_on 이미지를 사용하도록 설정합니다.
    # ========================================================

    import src.fastapi_cap_app as server

    original = server.cv2.VideoCapture
    which = {'on': True}


    # ========================================================
    # ⑦ 가짜 카메라 FakeCapture 만들기
    # - 실제 스마트폰 카메라를 사용하지 않습니다.
    # - isOpened()는 연결 성공 상태를 반환합니다.
    # - read()는 가상의 영상 Frame을 반환합니다.
    # - 카메라가 계속 영상을 보내는 상황을 흉내 냅니다.
    # ========================================================

    class FakeCapture:

        def __init__(self, url):
            self.url = url

        def isOpened(self):
            return True

        def read(self):
            time.sleep(.04)
            return True, synthetic_roi(which['on'])

        def release(self):
            pass


    # ========================================================
    # ⑧ FastAPI 테스트 환경 실행
    # - 실제 VideoCapture를 FakeCapture로 임시 교체합니다.
    # - TestClient로 FastAPI를 테스트합니다.
    # - 가상의 병뚜껑 판정 기준값을 적용합니다.
    # ========================================================

    server.cv2.VideoCapture = FakeCapture

    try:
        with TestClient(app) as client:

            with state.lock:
                state.rule = fake_rule
                state.rule_error = ''


            # =================================================
            # ⑨ API 화면과 카메라 URL 검증
            # - GET / : HTML 화면 요청이 성공하는지 확인
            # - 잘못된 URL : HTTP 400 오류가 나와야 함
            # - 올바른 URL : HTTP 200 응답이 나와야 함
            # =================================================

            assert client.get('/').status_code == 200

            assert client.post(
                '/api/camera',
                json={'url': 'https://example.com/video'}
            ).status_code == 400

            assert client.post(
                '/api/camera',
                json={
                    'url': 'http://192.168.0.30:8080/video'
                }
            ).status_code == 200


            # =================================================
            # ⑩ ROI 데이터 검증
            # - 잘못된 좌표를 보내면 HTTP 400 오류 확인
            # - 정상 좌표를 보내면 HTTP 200 응답 확인
            # - 이 과정에서 Pydantic 및 API 검증을 확인
            # =================================================

            assert client.post(
                '/api/roi',
                json={
                    'x1': 0.4,
                    'y1': 0.3,
                    'x2': 0.3,
                    'y2': 0.4
                }
            ).status_code == 400

            assert client.post(
                '/api/roi',
                json={
                    'x1': 0,
                    'y1': 0,
                    'x2': 1,
                    'y2': 1
                }
            ).status_code == 200


            # =================================================
            # ⑪ 검사 결과를 기다리는 함수
            # - GET /api/status로 현재 검사 상태를 조회합니다.
            # - 원하는 판정값이 나올 때까지 반복 확인합니다.
            # - 최대 60회 확인하며, 실패하면 오류를 발생시킵니다.
            # =================================================

            def wait_for(d):

                for _ in range(60):

                    status = client.get(
                        '/api/status'
                    ).json()

                    if (
                        status['decision'] == d
                        and status['camera_status'] == 'ONLINE'
                    ):
                        return status

                    time.sleep(.05)

                raise AssertionError(
                    f'No status {d}: {status}'
                )


            # =================================================
            # ⑫ 실시간 검사 결과 테스트
            # - cap_on Frame이 들어오면 OK인지 확인합니다.
            # - cap_off Frame으로 변경하면 NG인지 확인합니다.
            # - Area Ratio도 기준값과 올바르게 비교되는지 확인
            # =================================================

            assert (
                wait_for('OK')['ratio']
                > fake_rule.threshold
            )

            # 가짜 카메라를 뚜껑 없는 상태로 변경
            which['on'] = False

            assert (
                wait_for('NG')['ratio']
                < fake_rule.threshold
            )


            # =================================================
            # ⑬ ROI 초기화 및 테스트 환경 복구
            # - DELETE /api/roi 요청으로 ROI를 초기화합니다.
            # - 테스트 종료 후 기존 카메라 함수를 복구합니다.
            # =================================================

            assert client.delete(
                '/api/roi'
            ).json()['ok']

    finally:
        server.cv2.VideoCapture = original
        state.set_camera('')