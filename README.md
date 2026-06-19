# 진동 기반 침입 탐지 시스템

카메라나 마이크를 사용하지 않고, 문 표면의 3축 가속도 진동 데이터만으로 침입 전조 상황(노크, 타격, 밀침 등)을 감지하는 저비용 엣지-서버 하이브리드 보안 시스템.

## 시스템 구조

```
[ESP32-S3 + MPU6050]
  100Hz 수집 → 200샘플(2s) 슬라이딩 윈도우 → 1D-CNN 추론
        │
        ├─ confidence ≥ 0.6 ──→ POST /result ──┐
        │                                       │
        └─ confidence < 0.6 ──→ POST /refine ──┤
                                                ▼
                                  [FastAPI 중개 서버] (main_server.py)
                                                │
                                   is_confident=false인 경우
                                                ▼
                                  [2D-CNN 재검증 서버] (main.py)
                                  readings → 스펙트로그램 → softmax
                                                │
                                                ▼
                                  results_log.csv 저장 (중개 서버)
```

## 파일 구성

| 파일 | 실행 환경 | 역할 |
|---|---|---|
| `intrusion_detector.ino` | ESP32-S3 (펌웨어) | MPU6050 100Hz 데이터 수집, 200샘플 슬라이딩 윈도우(100샘플 오버랩) 구성, Z-score 정규화, TFLite Micro 1D-CNN 추론, confidence 0.6 기준 분기 후 `/result` 또는 `/refine`으로 전송 |
| `main_server.py` | 클라우드/로컬 서버 | **FastAPI 중개 서버.** `/result`(확실한 결과 저장), `/refine`(불확실한 결과를 2D-CNN 서버로 전달), `/recent`(최근 5개 결과 조회). 최종 결과를 `results_log.csv`에 기록 |
| `main.py` | 별도 서버 | **2D-CNN 분석 서버.** 중개 서버로부터 원본 readings 수신, 고역통과 필터 및 스펙트로그램 변환(`SPEC_SHAPE=(13,26,4)`) 후 `intrusion_model_2s.keras`로 추론, softmax 결과 반환 |
| `intrusion_model_2s.keras` | 2D-CNN 서버용 모델 | 2초 윈도우 스펙트로그램(13×26×4) 입력, 정상/주의/위험 3-class softmax 출력 |

> **주의:** 파일명과 역할이 직관적으로 매칭되지 않는다. `main.py`는 2D-CNN 분석 서버이고, `main_server.py`는 FastAPI 중개 서버이다.

`main.py`는 추론 후 자체적으로 `SERVER1_URL`(중개 서버 주소)의 `/result`에도 결과를 콜백 전송한다. 이는 2D-CNN 재검증 결과를 중개 서버 로그에 한 번 더 남기기 위한 경로이다.

## API 명세

### `main_server.py` (중개 서버)

**POST /result** — 확실한 1D-CNN 결과 저장
```json
{
  "device_id": "esp32_12345",
  "window_id": 3,
  "is_confident": true,
  "result": [0.607, 0.081, 0.312]
}
```

**POST /refine** — 불확실한 결과 → 2D-CNN 서버로 전달
```json
{
  "device_id": "esp32_12345",
  "window_id": 3,
  "is_confident": false,
  "result": "주의",
  "confidence": 0.3123,
  "readings": [{"t": 0.00, "x": 1.2345, "y": 2.3456, "z": 3.4567}]
}
```

**GET /recent** — 최근 5개 결과 조회

### `main.py` (2D-CNN 서버)

**POST /refine** — 스펙트로그램 변환 후 재검증, softmax 반환

**GET /health** — 모델 로드 상태 확인

### 클래스 정의
```python
CLASSES = ["정상", "주의", "위험"]
```

## 실행 방법

```bash
# FastAPI 중개 서버
uvicorn main_server:app --host 0.0.0.0 --port 8000

# 2D-CNN 서버 (main.py)
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 환경변수

| 변수 | 위치 | 설명 |
|---|---|---|
| `MODEL_PATH` | main.py | 2D-CNN 모델 파일 경로 (기본값: `intrusion_model_2s.keras`) |
| `SERVER1_URL` | main.py | 중개 서버 주소 (결과 콜백 전송용) |
| `SECOND_SERVER_URL` | main_server.py | 2D-CNN 서버 주소 (상단에 하드코딩되어 있음, 필요 시 환경변수로 분리 권장) |

## 알려진 한계

- 사용자 알림은 콘솔 출력과 CSV 로그뿐이며, 실제 앱 푸시는 구현되지 않음
- 특정 문 재질·부착 위치에서 수집한 데이터 기반이라 다른 환경에서의 일반화 성능은 미검증
- 2D-CNN 서버 의존 구조이나 재시도/장애 대응 로직 없음
- confidence 임계값 0.6은 추가적인 데이터 기반 검증 필요
# data_analysis_capstone
# data_analysis_capstone
