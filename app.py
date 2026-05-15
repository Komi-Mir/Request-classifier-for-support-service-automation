import joblib
from flask import Flask, jsonify, request

from preprocess import preprocess_text

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "TfIdfVectorizer.pkl"
ENCODER_PATH = "label_encoder.pkl"

PRIORITY_MAP = {
    "Оплата": "Высокий",
    "Техническая ошибка": "Высокий",
    "Доставка": "Средний",
    "Возврат и обмен": "Средний",
    "Спам": "Низкий",
}

app = Flask(__name__)

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
label_encoder = joblib.load(ENCODER_PATH)


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "service": "Support request classifier",
            "status": "ok",
            "endpoint": "/predict",
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    raw_text = data.get("text", "").strip()

    if not raw_text:
        return jsonify({"error": "Text is empty"}), 400

    cleaned_text = preprocess_text(raw_text)

    if not cleaned_text:
        return jsonify({"error": "Text is empty after preprocessing"}), 400

    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]

    category = label_encoder.inverse_transform([prediction])[0]
    priority = PRIORITY_MAP.get(category, "Средний")

    return jsonify(
        {
            "category": category,
            "priority": priority,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=8000)
