from collections import defaultdict
from threading import Lock

from flask import Flask, render_template, jsonify, Response
import sounddevice as sd
import numpy as np
import cv2

from speech_emotion import analyze_emotion, predict_emotion
from facial_emotion import detect_face_emotion

RECORD_SECONDS = 6
SAMPLE_RATE = 22050
SPEECH_MODEL_WEIGHT = 0.7
FACE_MODEL_WEIGHT = 0.3

FACE_TO_SPEECH_EMOTION = {
    "angry": "angry",
    "sad": "sad",
    "happy": "happy",
    "neutral": "neutral",
    "fear": "sad",
    "fearful": "sad",
    "disgust": "angry",
    "disgusted": "angry",
    "surprise": "happy",
    "surprised": "happy",
    "calm": "neutral"
}

latest_face_state = {
    "emotion": "No face detected",
    "confidence": 0.0,
    "scores": {}
}
face_state_lock = Lock()

app = Flask(__name__)


def record_audio(duration=RECORD_SECONDS, fs=SAMPLE_RATE):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return audio.flatten(), fs


def normalize_face_emotion(face_emotion):
    if not face_emotion:
        return None
    return FACE_TO_SPEECH_EMOTION.get(str(face_emotion).lower())


def summarize_speech_timeline(timeline):
    if not timeline:
        return "unknown", 0.0, {}

    score_by_emotion = defaultdict(float)
    for chunk in timeline:
        chunk_energy = float(chunk.get("energy", 0.01))
        chunk_weight = max(chunk_energy, 0.01)
        probabilities = chunk.get("probabilities", {}) or {}

        if probabilities:
            for emotion, probability in probabilities.items():
                score_by_emotion[str(emotion)] += chunk_weight * float(probability)
            continue

        emotion = chunk.get("emotion", "unknown")
        confidence = float(chunk.get("confidence", 0.0))
        score_by_emotion[emotion] += chunk_weight * (confidence if confidence > 0 else 1.0)

    total = float(sum(score_by_emotion.values()))
    if total <= 0:
        return "unknown", 0.0, {}

    normalized_scores = {
        emotion: round(score / total, 4)
        for emotion, score in score_by_emotion.items()
    }

    sorted_scores = sorted(
        normalized_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )
    dominant_emotion = max(score_by_emotion, key=score_by_emotion.get)
    dominant_confidence = normalized_scores.get(dominant_emotion, 0.0)

    # Low-margin predictions are often unstable and biased to one class.
    if len(sorted_scores) > 1:
        top_score = float(sorted_scores[0][1])
        second_score = float(sorted_scores[1][1])
        if "neutral" in normalized_scores and (top_score < 0.45 or (top_score - second_score) < 0.08):
            dominant_emotion = "neutral"
            dominant_confidence = float(normalized_scores["neutral"])

    return dominant_emotion, dominant_confidence, normalized_scores


def final_emotion(speech_emotion, speech_confidence, face_emotion, face_confidence):
    face_mapped = normalize_face_emotion(face_emotion)
    if face_mapped is None:
        return {
            "emotion": speech_emotion,
            "confidence": speech_confidence,
            "source": "speech_only"
        }

    combined_scores = defaultdict(float)
    combined_scores[speech_emotion] += SPEECH_MODEL_WEIGHT * max(speech_confidence, 0.01)
    combined_scores[face_mapped] += FACE_MODEL_WEIGHT * max(face_confidence, 0.01)

    final_label = max(combined_scores, key=combined_scores.get)
    total = float(sum(combined_scores.values()))
    final_confidence = round(combined_scores[final_label] / total, 4) if total else 0.0

    return {
        "emotion": final_label,
        "confidence": final_confidence,
        "source": "agreement" if speech_emotion == face_mapped else "weighted_fusion"
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    try:
        audio, sr = record_audio(duration=RECORD_SECONDS)
        rms_energy = float(np.sqrt(np.mean(np.square(audio))))
        if rms_energy < 0.005:
            return jsonify({
                "error": "Audio level is too low. Please speak louder and try again."
            }), 400

        timeline = analyze_emotion(audio, sr, predict_emotion)
        speech_emotion, speech_confidence, speech_scores = summarize_speech_timeline(timeline)

        with face_state_lock:
            face_state = dict(latest_face_state)

        face_emotion = face_state.get("emotion", "No face detected")
        face_confidence = float(face_state.get("confidence", 0.0))

        fusion_result = final_emotion(
            speech_emotion,
            speech_confidence,
            face_emotion,
            face_confidence
        )

        return jsonify({
            "speech_timeline": timeline,
            "speech_emotion": speech_emotion,
            "speech_confidence": round(speech_confidence, 4),
            "speech_scores": speech_scores,
            "face_emotion": face_emotion,
            "face_confidence": round(face_confidence, 4),
            "final_emotion": fusion_result["emotion"],
            "final_confidence": fusion_result["confidence"],
            "fusion_source": fusion_result["source"]
        })
    except Exception as exc:
        print(f"Error processing emotions: {exc}")
        return jsonify({"error": str(exc)}), 500


def generate_frames():
    cap = cv2.VideoCapture(0)
    frame_counter = 0
    current_emotion = "No face detected"
    current_confidence = 0.0

    if not cap.isOpened():
        with face_state_lock:
            latest_face_state.update({
                "emotion": "Webcam unavailable",
                "confidence": 0.0,
                "scores": {}
            })
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Run face analysis less frequently to keep the video stream smooth.
            if frame_counter % 5 == 0:
                face_details = detect_face_emotion(frame, return_details=True)
                current_emotion = face_details.get("emotion", "No face detected")
                current_confidence = float(face_details.get("confidence", 0.0))

                with face_state_lock:
                    latest_face_state.update(face_details)

            frame_counter += 1

            cv2.putText(
                frame,
                f"Emotion: {current_emotion}",
                (10, 30),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                f"Confidence: {current_confidence:.2f}",
                (10, 65),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        cap.release()


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/face_status")
def face_status():
    with face_state_lock:
        return jsonify(dict(latest_face_state))


if __name__ == "__main__":
    app.run(debug=True)
