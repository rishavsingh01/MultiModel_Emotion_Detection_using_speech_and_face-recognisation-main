import numpy as np
import librosa
import pickle
from pathlib import Path

# Define paths for persisted model artifacts.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "speech_emotion_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

# Configure default time windows for chunk-based analysis.
DEFAULT_CHUNK_DURATION = 2.5
DEFAULT_HOP_DURATION = 1.0

# Load trained model objects once at module import time.
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)


# Normalize every audio clip to a fixed duration expected by feature extractors.
def _fix_length(audio, sample_rate, target_duration=DEFAULT_CHUNK_DURATION):
    target_len = int(target_duration * sample_rate)
    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)))
    if len(audio) > target_len:
        return audio[:target_len]
    return audio


# Build a lightweight MFCC-only feature vector for legacy model compatibility.
def extract_feature_legacy(audio, sample_rate):
    audio = _fix_length(audio, sample_rate)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.hstack((np.mean(mfcc, axis=1), np.std(mfcc, axis=1))).astype(np.float32)


# Build a richer multi-feature vector using spectral and temporal descriptors.
def extract_feature_rich(audio, sample_rate):
    audio = _fix_length(audio, sample_rate)
    hop_length = 256
    n_fft = 1024

    stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate, hop_length=hop_length)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=64,
    )
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)
    rms = librosa.feature.rms(S=stft, frame_length=n_fft, hop_length=hop_length)

    feature_parts = []
    for matrix in (mfcc, chroma, mel, contrast, zcr, rms):
        feature_parts.append(np.mean(matrix, axis=1))
        feature_parts.append(np.std(matrix, axis=1))

    return np.hstack(feature_parts).astype(np.float32)


# Select and shape features so they match the scaler/model input dimensionality.
def extract_feature(audio, sample_rate):
    expected = getattr(scaler, "n_features_in_", None)
    legacy = extract_feature_legacy(audio, sample_rate)
    rich = extract_feature_rich(audio, sample_rate)

    if expected is None:
        return rich
    if expected == len(rich):
        return rich
    if expected == len(legacy):
        return legacy

    chosen = rich if abs(len(rich) - expected) <= abs(len(legacy) - expected) else legacy
    if len(chosen) > expected:
        chosen = chosen[:expected]
    elif len(chosen) < expected:
        chosen = np.pad(chosen, (0, expected - len(chosen)))

    return chosen.astype(np.float32)


# Split long audio into overlapping chunks and attach per-chunk timing and energy.
def split_audio(audio, sr, chunk_duration=DEFAULT_CHUNK_DURATION, hop_duration=DEFAULT_HOP_DURATION):
    audio = np.asarray(audio, dtype=np.float32)
    chunk_samples = max(1, int(chunk_duration * sr))
    hop_samples = max(1, int(hop_duration * sr))

    if len(audio) == 0:
        return []

    if len(audio) <= chunk_samples:
        padded = np.pad(audio, (0, chunk_samples - len(audio)))
        rms = float(np.sqrt(np.mean(np.square(padded))))
        return [{"chunk": padded, "start": 0.0, "end": chunk_duration, "energy": rms}]

    starts = list(range(0, len(audio) - chunk_samples + 1, hop_samples))
    final_start = len(audio) - chunk_samples
    if starts[-1] != final_start:
        starts.append(final_start)

    chunks = []
    for start in starts:
        end = start + chunk_samples
        chunk = audio[start:end]
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        chunks.append(
            {
                "chunk": chunk,
                "start": round(start / sr, 2),
                "end": round(end / sr, 2),
                "energy": rms,
            }
        )

    return chunks


# Convert raw model scores into normalized probabilities.
def _softmax(scores):
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


# Generate class probabilities across different classifier interfaces.
def _predict_probabilities(feature):
    feature = np.asarray(feature, dtype=np.float32).reshape(1, -1)

    if hasattr(model, "predict_proba"):
        return model.predict_proba(feature)[0].astype(np.float32)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(feature)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        return _softmax(scores)[0].astype(np.float32)

    prediction = int(model.predict(feature)[0])
    probabilities = np.zeros(len(label_encoder.classes_), dtype=np.float32)
    probabilities[prediction] = 1.0
    return probabilities


# Predict the dominant speech emotion for one audio segment.
def predict_emotion(audio, sr, return_details=False):
    feature = extract_feature(audio, sr)
    feature = scaler.transform([feature])

    probabilities = _predict_probabilities(feature)
    top_index = int(np.argmax(probabilities))
    emotion = str(label_encoder.classes_[top_index])
    confidence = float(probabilities[top_index])

    sorted_idx = np.argsort(probabilities)[::-1]
    margin = float(
        probabilities[sorted_idx[0]] - probabilities[sorted_idx[1]]
    ) if len(sorted_idx) > 1 else float(probabilities[sorted_idx[0]])

    if return_details:
        return {
            "emotion": emotion,
            "confidence": confidence,
            "margin": margin,
            "probabilities": {
                str(label): float(probabilities[i])
                for i, label in enumerate(label_encoder.classes_)
            },
        }

    return emotion


# Run chunk-level inference and return a timeline of voiced emotional cues.
def analyze_emotion(audio, sr, predict_function):
    chunks = split_audio(audio, sr)
    if not chunks:
        return []

    energies = np.array([chunk["energy"] for chunk in chunks], dtype=np.float32)
    energy_threshold = max(0.004, float(np.percentile(energies, 35)))

    voiced_chunks = [chunk for chunk in chunks if chunk["energy"] >= energy_threshold]
    if not voiced_chunks:
        voiced_chunks = [max(chunks, key=lambda item: item["energy"])]

    timeline = []
    for chunk_info in voiced_chunks:
        prediction = predict_function(chunk_info["chunk"], sr, return_details=True)

        if isinstance(prediction, dict):
            emotion = prediction.get("emotion", "unknown")
            confidence = float(prediction.get("confidence", 0.0))
            margin = float(prediction.get("margin", 0.0))
            probabilities = prediction.get("probabilities", {})
        else:
            emotion = str(prediction)
            confidence = 0.0
            margin = 0.0
            probabilities = {}

        timeline.append(
            {
                "time": f"{chunk_info['start']:.1f}-{chunk_info['end']:.1f} sec",
                "emotion": emotion,
                "confidence": round(confidence, 4),
                "margin": round(margin, 4),
                "energy": round(float(chunk_info["energy"]), 5),
                "probabilities": probabilities,
            }
        )

    return timeline
