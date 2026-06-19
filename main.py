import os
import numpy as np
import httpx
from scipy import signal
from scipy.ndimage import zoom
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import tensorflow as tf

# ── 설정 ──
MODEL_PATH = os.environ.get("MODEL_PATH", "intrusion_model_2s.keras")
SERVER1_URL = os.environ.get("SERVER1_URL", "http://3.27.105.83:8000")
FS = 100
WIN_2S = 200
CLASS_NAMES = ["정상", "주의", "위험"]
SPEC_SHAPE = (13, 26, 4)

app = FastAPI(title="2D CNN Intrusion Detector")
model = None


@app.on_event("startup")
def load_model():
    global model
    model = tf.keras.models.load_model(MODEL_PATH)
    dummy = np.zeros((1, *SPEC_SHAPE), dtype=np.float32)
    model.predict(dummy, verbose=0)
    print(f"Model loaded: {MODEL_PATH}, input={model.input_shape}")


# ── 전처리 (노트북 v2 make_spectrogram 동일) ──

def make_spectrogram(segment_4ch: np.ndarray, fs: int) -> np.ndarray:
    seg3 = segment_4ch[:, :3]

    nyquist = 0.5 * fs
    b, a = signal.butter(3, 0.5 / nyquist, btype="high", analog=False)

    # ch3: 정규화 전 원신호 에너지 스펙트로그램
    raw_filt = signal.filtfilt(b, a, seg3, axis=0)
    n = len(raw_filt)
    nperseg = max(8, n // 8)
    noverlap = nperseg * 3 // 4

    energy_specs = []
    for ch in range(raw_filt.shape[1]):
        _, _, Sxx = signal.spectrogram(
            raw_filt[:, ch], fs=fs, nperseg=nperseg, noverlap=noverlap
        )
        energy_specs.append(np.log1p(np.abs(Sxx)))
    energy_ch = np.mean(np.stack(energy_specs, axis=-1), axis=-1, keepdims=True)

    # ch0~2: 윈도우별 정규화 후 패턴 스펙트로그램
    mean = np.mean(seg3, axis=0)
    std = np.std(seg3, axis=0)
    std[std < 1e-6] = 1e-6
    seg3_norm = (seg3 - mean) / std

    filtered = signal.filtfilt(b, a, seg3_norm, axis=0)
    pattern_specs = []
    for ch in range(filtered.shape[1]):
        _, _, Sxx = signal.spectrogram(
            filtered[:, ch], fs=fs, nperseg=nperseg, noverlap=noverlap
        )
        pattern_specs.append(np.log1p(np.abs(Sxx)))
    pattern_ch = np.stack(pattern_specs, axis=-1)

    return np.concatenate([pattern_ch, energy_ch], axis=-1)


def readings_to_input(readings: list[dict]) -> np.ndarray:
    ax = np.array([r["x"] for r in readings], dtype=np.float64)
    ay = np.array([r["y"] for r in readings], dtype=np.float64)
    az = np.array([r["z"] for r in readings], dtype=np.float64)
    aabs = np.sqrt(ax**2 + ay**2 + az**2)
    segment = np.column_stack([ax, ay, az, aabs])

    spec = make_spectrogram(segment, FS)

    th, tw = SPEC_SHAPE[0], SPEC_SHAPE[1]
    if spec.shape[:2] != (th, tw):
        spec = zoom(spec, [th / spec.shape[0], tw / spec.shape[1], 1])

    return spec.astype(np.float32)


# ── API ──

class Reading(BaseModel):
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
    readings: list[Reading]


class RefineResponse(BaseModel):
    device_id: str
    window_id: int
    softmax_output: list[float]
    predicted_class: str
    predicted_index: int


@app.post("/refine", response_model=RefineResponse)
def refine(req: RefineRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    if len(req.readings) < 10:
        raise HTTPException(422, f"Too few readings: {len(req.readings)}")

    readings_dict = [r.model_dump() for r in req.readings]
    spec = readings_to_input(readings_dict)
    batch = np.expand_dims(spec, axis=0)

    probs = model.predict(batch, verbose=0)[0]
    softmax_list = [round(float(p), 6) for p in probs]
    pred_idx = int(np.argmax(probs))

    # 서버1의 /result로 결과 전송
    all_vals = [r.x for r in req.readings] + [r.y for r in req.readings] + [r.z for r in req.readings]
    result_payload = {
        "device_id": req.device_id,
        "window_id": req.window_id,
        "is_confident": True,
        "result": softmax_list,
        "mean": float(np.mean(all_vals)),
        "std": float(np.std(all_vals)),
        "fs": FS,
        "win_2s": WIN_2S,
        "classes": CLASS_NAMES,
        "softmax_output": softmax_list,
    }
    try:
        resp = httpx.post(f"{SERVER1_URL}/result", json=result_payload, timeout=5.0)
        print(f"[→ server1 /result] HTTP {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"[→ server1 /result] Failed: {e}")

    return RefineResponse(
        device_id=req.device_id,
        window_id=req.window_id,
        softmax_output=softmax_list,
        predicted_class=CLASS_NAMES[pred_idx],
        predicted_index=pred_idx,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
