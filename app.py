from collections import defaultdict
from threading import Lock
import platform

from flask import Flask, render_template, jsonify, Response
import sounddevice as sd
import numpy as np
import cv2

from speech_emotion import analyze_emotion, predict_emotion
from facial_emotion import detect_face_emotion

# Configure audio capture, fusion weights, and fallback camera indexes.
RECORD_SECONDS = 9
SAMPLE_RATE = 22050
SPEECH_MODEL_WEIGHT = 0.7
FACE_MODEL_WEIGHT = 0.3
CAMERA_INDEX_CANDIDATES = (0, 1, 2)

# Map face model labels into the speech model's 8-class label space.
FACE_TO_SPEECH_EMOTION = {
    "angry": "angry",
    "sad": "sad",
    "happy": "happy",
    "neutral": "neutral",
    "calm": "calm",
    "fear": "fearful",
    "fearful": "fearful",
    "disgust": "disgusted",
    "disgusted": "disgusted",
    "surprise": "surprised",
    "surprised": "surprised",
}

# Store the latest face inference snapshot shared across requests.
latest_face_state = {
    "emotion": "No face detected",
    "confidence": 0.0,
    "scores": {}
}
face_state_lock = Lock()

app = Flask(__name__)


# Open webcam using platform-specific backends with index fallbacks.
def open_webcam():
    system_name = platform.system().lower()

    if system_name == "windows":
        backend_candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backend_candidates = [cv2.CAP_ANY]

    for index in CAMERA_INDEX_CANDIDATES:
        for backend in backend_candidates:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                return cap
            cap.release()

    return None


# Record mono audio for emotion analysis.
def record_audio(duration=RECORD_SECONDS, fs=SAMPLE_RATE):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return audio.flatten(), fs


# Normalize external face labels into internal speech emotion labels.
def normalize_face_emotion(face_emotion):
    if not face_emotion:
        return None
    return FACE_TO_SPEECH_EMOTION.get(str(face_emotion).lower())


# Aggregate chunk-level speech outputs into a single stabilized speech summary.
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
    dominant_emotion = str(sorted_scores[0][0])
    dominant_confidence = float(sorted_scores[0][1])

    # Stabilize low-margin predictions by preferring calm/neutral when confidence is weak.
    if len(sorted_scores) > 1:
        top_score = dominant_confidence
        second_score = float(sorted_scores[1][1])
        margin = top_score - second_score
        calm_score = float(normalized_scores.get("calm", 0.0))
        neutral_score = float(normalized_scores.get("neutral", 0.0))
        fallback_score = max(calm_score, neutral_score)

        if fallback_score > 0 and (top_score < 0.4 or margin < 0.06):
            dominant_emotion = "calm" if calm_score >= neutral_score else "neutral"
            dominant_confidence = fallback_score

    return dominant_emotion, dominant_confidence, normalized_scores


# Fuse speech and face evidence into one final emotion prediction.
def final_emotion(speech_emotion, speech_confidence, speech_scores, face_emotion, face_confidence):
    face_mapped = normalize_face_emotion(face_emotion)

    # Blend full speech score distribution with face confidence for more stable fusion.
    combined_scores = defaultdict(float)
    if speech_scores:
        for emotion, score in speech_scores.items():
            combined_scores[str(emotion)] += SPEECH_MODEL_WEIGHT * max(float(score), 0.0)
    else:
        combined_scores[speech_emotion] += SPEECH_MODEL_WEIGHT * max(speech_confidence, 0.01)

    if face_mapped is not None:
        combined_scores[face_mapped] += FACE_MODEL_WEIGHT * max(face_confidence, 0.01)

    if not combined_scores:
        return {
            "emotion": speech_emotion,
            "confidence": speech_confidence,
            "source": "speech_only"
        }

    final_label = str(max(combined_scores, key=combined_scores.get))
    total = float(sum(combined_scores.values()))
    final_confidence = round(combined_scores[final_label] / total, 4) if total else 0.0

    return {
        "emotion": final_label,
        "confidence": final_confidence,
        "source": (
            "agreement"
            if face_mapped is not None and speech_emotion == face_mapped
            else "weighted_fusion" if face_mapped is not None else "speech_only"
        )
    }


@app.route("/")
def home():
    # Render the dashboard and pass backend recording duration to the frontend.
    return render_template("index.html", record_seconds=RECORD_SECONDS)


@app.route("/process", methods=["POST"])
def process():
    try:
        # Capture audio and reject near-silent recordings.
        audio, sr = record_audio(duration=RECORD_SECONDS)
        rms_energy = float(np.sqrt(np.mean(np.square(audio))))
        if rms_energy < 0.005:
            return jsonify({
                "error": "Audio level is too low. Please speak louder and try again."
            }), 400

        # Run speech timeline inference and summarize it into one speech label.
        timeline = analyze_emotion(audio, sr, predict_emotion)
        speech_emotion, speech_confidence, speech_scores = summarize_speech_timeline(timeline)

        # Read latest face snapshot and fuse it with speech prediction.
        with face_state_lock:
            face_state = dict(latest_face_state)

        face_emotion = face_state.get("emotion", "No face detected")
        face_confidence = float(face_state.get("confidence", 0.0))

        fusion_result = final_emotion(
            speech_emotion,
            speech_confidence,
            speech_scores,
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
    # Stream webcam frames while periodically refreshing face emotion inference.
    cap = open_webcam()
    frame_counter = 0
    current_emotion = "No face detected"
    current_confidence = 0.0

    if cap is None:
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
    # Serve MJPEG stream with no-cache headers to keep browser feed fresh.
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.route("/face_status")
def face_status():
    # Expose the latest cached face inference for frontend polling.
    with face_state_lock:
        return jsonify(dict(latest_face_state))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
