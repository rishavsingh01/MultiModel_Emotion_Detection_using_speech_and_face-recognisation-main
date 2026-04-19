from deepface import DeepFace


# Detect facial emotion and optionally return calibrated confidence details.
def detect_face_emotion(frame, return_details=False):
    try:
        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        if isinstance(result, list):
            face_result = result[0] if result else {}
        elif isinstance(result, dict):
            face_result = result
        else:
            face_result = {}

        emotion = str(face_result.get("dominant_emotion", "No face detected"))
        emotion_scores = face_result.get("emotion", {}) or {}
        confidence = float(emotion_scores.get(emotion, 0.0)) / 100.0
    except Exception:
        emotion = "No face detected"
        confidence = 0.0
        emotion_scores = {}

    if return_details:
        return {
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "scores": {
                str(label): float(score) / 100.0
                for label, score in emotion_scores.items()
            }
        }

    return emotion
