from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Union
import requests
import csv
import os
import ast
from datetime import datetime

app = FastAPI()

# =========================
# 2D CNN 서버 주소
# =========================
SECOND_SERVER_URL = "https://phqb277wuu8x3r-8000.proxy.runpod.net/refine"

CSV_FILE = "results_log.csv"

CLASSES = ["정상", "주의", "위험"]


# =========================
# /result 요청 형식
# =========================
class ResultRequest(BaseModel):
    device_id: str
    window_id: int
    is_confident: bool
    result: List[float]  # [정상확률, 주의확률, 위험확률]


# =========================
# /refine 요청 형식
# =========================
class RefineReading(BaseModel):
    t: float
    x: float
    y: float
    z: float


class RefineRequest(BaseModel):
    device_id: str
    window_id: int
    is_confident: bool
    result: str
    confidence: float
    readings: List[RefineReading]


# =========================
# 공통 유틸 함수
# =========================
def validate_softmax(softmax: List[float]) -> List[float]:
    if len(softmax) != 3:
        raise ValueError("softmax는 [정상확률, 주의확률, 위험확률] 형태의 3개 값이어야 합니다.")

    return [float(v) for v in softmax]


def parse_softmax_output(value) -> List[float]:
    """
    2D CNN 서버 응답의 softmax_output을 리스트로 변환.
    가능한 형식:
    - [0.1, 0.2, 0.7]
    - "[0.1, 0.2, 0.7]"
    - "[정상확률, 주의확률, 위험확률]" 같은 설명 문자열은 실제 확률이 아니므로 에러 처리
    """
    if isinstance(value, list):
        return validate_softmax(value)

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return validate_softmax(parsed)
        except Exception:
            pass

    raise ValueError(f"softmax_output 형식을 해석할 수 없습니다: {value}")


def get_label_from_softmax(softmax: List[float]) -> tuple[str, float]:
    softmax = validate_softmax(softmax)
    max_idx = max(range(3), key=lambda i: softmax[i])
    return CLASSES[max_idx], softmax[max_idx]


def get_message(label: str) -> str:
    if label == "정상":
        return "이상 없음"
    elif label == "주의":
        return "주의 상황 감지"
    elif label == "위험":
        return "위험 상황 감지"
    else:
        return "분류 불확실"


def save_result(
    device_id,
    window_id,
    is_confident,
    result,
    confidence,
    source,
    message,
    softmax_0=None,
    softmax_1=None,
    softmax_2=None,
    sample_count=0,
    mean=None,
    std=None,
    fs=None,
    win_2s=None
):
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "server_timestamp",
                "device_id",
                "window_id",
                "is_confident",
                "result",
                "confidence",
                "source",
                "message",
                "softmax_0_normal",
                "softmax_1_caution",
                "softmax_2_danger",
                "sample_count",
                "mean",
                "std",
                "fs",
                "win_2s"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            device_id,
            window_id,
            is_confident,
            result,
            confidence,
            source,
            message,
            softmax_0,
            softmax_1,
            softmax_2,
            sample_count,
            mean,
            std,
            fs,
            win_2s
        ])


def send_user_notification(device_id, result, confidence, message):
    print("====== 사용자 알림 ======")
    print(f"device_id: {device_id}")
    print(f"result: {result}")
    print(f"confidence: {confidence}")
    print(f"message: {message}")
    print("========================")
    return True


# =========================
# 서버 상태 확인
# =========================
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "First server is running",
        "endpoints": {
            "direct_1d_result": "POST /result",
            "ambiguous_refine": "POST /refine",
            "recent_logs": "GET /recent"
        },
        "result_format": {
            "device_id": "esp32_12345",
            "window_id": 3,
            "is_confident": True,
            "result": "[정상확률, 주의확률, 위험확률]"
        },
        "refine_format": {
            "device_id": "esp32_12345",
            "window_id": 3,
            "is_confident": False,
            "result": "주의",
            "confidence": 0.3123,
            "readings": [
                {"t": 0.00, "x": 1.2345, "y": 2.3456, "z": 3.4567}
            ]
        }
    }


# =========================
# 최근 5개 결과 확인
# =========================
@app.get("/recent")
def get_recent_results():
    if not os.path.isfile(CSV_FILE):
        return {
            "status": "success",
            "count": 0,
            "recent_results": []
        }

    with open(CSV_FILE, mode="r", encoding="utf-8-sig") as f:
        reader = list(csv.DictReader(f))

    recent_results = reader[-5:]

    return {
        "status": "success",
        "count": len(recent_results),
        "recent_results": recent_results
    }


# =========================
# 1D CNN 확실한 결과 수신
# =========================
@app.post("/result")
def receive_result(data: ResultRequest):
    """
    요청 예시:
    {
      "device_id": "esp32_12345",
      "window_id": 3,
      "is_confident": true,
      "result": [0.607, 0.081, 0.312]
    }
    """

    if data.is_confident is not True:
        return {
            "status": "error",
            "message": "/result는 is_confident=true인 경우에 사용하세요. 애매한 경우는 /refine으로 보내야 합니다."
        }

    try:
        softmax = validate_softmax(data.result)
    except ValueError as e:
        return {
            "status": "error",
            "message": str(e)
        }

    final_result, final_confidence = get_label_from_softmax(softmax)
    message = get_message(final_result)

    save_result(
        device_id=data.device_id,
        window_id=data.window_id,
        is_confident=data.is_confident,
        result=final_result,
        confidence=final_confidence,
        source="1d_cnn",
        message=message,
        softmax_0=softmax[0],
        softmax_1=softmax[1],
        softmax_2=softmax[2],
        sample_count=0
    )

    send_user_notification(
        device_id=data.device_id,
        result=final_result,
        confidence=final_confidence,
        message=message
    )

    return {
        "status": "success",
        "type": "direct_1d_result",
        "device_id": data.device_id,
        "window_id": data.window_id,
        "final_result": final_result,
        "confidence": final_confidence,
        "softmax": softmax,
        "message": message
    }


# =========================
# 1D CNN 애매한 결과 수신 → 2D CNN 서버로 전달
# =========================
@app.post("/refine")
def receive_refine(data: RefineRequest):
    """
    요청 예시:
    {
      "device_id": "esp32_12345",
      "window_id": 3,
      "is_confident": false,
      "result": "주의",
      "confidence": 0.3123,
      "readings": [
        {"t": 0.00, "x": 1.2345, "y": 2.3456, "z": 3.4567}
      ]
    }
    """

    if data.is_confident is not False:
        return {
            "status": "error",
            "message": "/refine은 is_confident=false인 경우에 사용하세요. 확실한 경우는 /result로 보내야 합니다."
        }

    if data.readings is None or len(data.readings) == 0:
        return {
            "status": "error",
            "message": "readings 데이터가 필요합니다."
        }

    # 2D CNN 서버로 보낼 payload
    payload = {
        "device_id": data.device_id,
        "window_id": data.window_id,
        "is_confident": data.is_confident,
        "result": data.result,
        "confidence": data.confidence,
        "readings": [
            {
                "t": r.t,
                "x": r.x,
                "y": r.y,
                "z": r.z
            }
            for r in data.readings
        ]
    }

    try:
        response = requests.post(
            SECOND_SERVER_URL,
            json=payload,
            timeout=30
        )

        response.raise_for_status()
        second_result = response.json()

    except requests.exceptions.RequestException as e:
        # 2D 서버 실패도 로그에 저장
        save_result(
            device_id=data.device_id,
            window_id=data.window_id,
            is_confident=data.is_confident,
            result="2d_server_error",
            confidence=data.confidence,
            source="server_error",
            message=str(e),
            sample_count=len(data.readings)
        )

        return {
            "status": "error",
            "message": "2D CNN 서버 요청 실패",
            "detail": str(e)
        }

    # 2D CNN 서버 응답 예시:
    # {
    #   "mean": 0.0046,
    #   "std": 0.0405,
    #   "fs": 100,
    #   "win_2s": 200,
    #   "classes": ["정상", "주의", "위험"],
    #   "softmax_output": "[정상확률, 주의확률, 위험확률]"
    # }

    try:
        softmax = parse_softmax_output(second_result.get("softmax_output"))
        classes = second_result.get("classes", CLASSES)

        max_idx = max(range(3), key=lambda i: softmax[i])
        final_result = classes[max_idx]
        final_confidence = softmax[max_idx]
        message = get_message(final_result)

    except Exception as e:
        save_result(
            device_id=data.device_id,
            window_id=data.window_id,
            is_confident=data.is_confident,
            result="2d_response_parse_error",
            confidence=data.confidence,
            source="server_error",
            message=str(e),
            sample_count=len(data.readings)
        )

        return {
            "status": "error",
            "message": "2D CNN 응답 파싱 실패",
            "detail": str(e),
            "raw_response": second_result
        }

    save_result(
        device_id=data.device_id,
        window_id=data.window_id,
        is_confident=data.is_confident,
        result=final_result,
        confidence=final_confidence,
        source="2d_cnn",
        message=message,
        softmax_0=softmax[0],
        softmax_1=softmax[1],
        softmax_2=softmax[2],
        sample_count=len(data.readings),
        mean=second_result.get("mean"),
        std=second_result.get("std"),
        fs=second_result.get("fs"),
        win_2s=second_result.get("win_2s")
    )

    send_user_notification(
        device_id=data.device_id,
        result=final_result,
        confidence=final_confidence,
        message=message
    )

    return {
        "status": "success",
        "type": "refined_2d_result",
        "device_id": data.device_id,
        "window_id": data.window_id,
        "one_d_result": data.result,
        "one_d_confidence": data.confidence,
        "final_result": final_result,
        "confidence": final_confidence,
        "softmax": softmax,
        "message": message,
        "sample_count": len(data.readings),
        "two_d_info": {
            "mean": second_result.get("mean"),