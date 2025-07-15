from flask import Flask, render_template, request
import joblib
import os

# ------------------------------------------------------------------
# 0) Setup Flask
# ------------------------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------------------------
# 1) Load ML model (if you have one) otherwise simple rule
# ------------------------------------------------------------------
MODEL_PATH = os.path.join("model", "mental_health_model.pkl")

def simple_rule_based_prediction(inputs):
    sleep, stress, depression, appetite, focus = inputs
    score = (10 - sleep) + int(stress) + 2*int(depression) + int(appetite) + int(focus)
    if score < 8:
        return "Healthy"
    elif score < 14:
        return "Mild Stress"
    else:
        return "Needs Attention"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    def predict(inputs):
        return model.predict([inputs])[0]
else:
    def predict(inputs):
        return simple_rule_based_prediction(inputs)

# ------------------------------------------------------------------
# 2) Routes
# ------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Receives form data, calls predict(), and shows suggestions page.
    """
    # ---- Get form values ----
    sleep       = float(request.form["sleep"])
    stress      = int(request.form["stress"])
    depression  = int(request.form["depression"])
    appetite    = int(request.form["appetite"])
    focus       = int(request.form["focus"])

    # ---- Prediction ----
    result = predict([sleep, stress, depression, appetite, focus])

    # ---- Suggestions ----
    if "Healthy" in result:
        tips = [
            "Maintain your regular sleep schedule.",
            "Stay socially connected with friends and family.",
            "Keep up positive hobbies like music or reading."
        ]
    elif "Mild Stress" in result:
        tips = [
            "Practice deep‑breathing or 5‑minute meditation.",
            "Get outside for a short walk or stretch.",
            "Share feelings with a friend or write in a journal."
        ]
    else:  # Needs Attention
        tips = [
            "Talk to someone you trust—or a counselor—soon.",
            "Create a simple routine with balanced meals and sleep.",
            "Avoid isolation; stay in touch with supportive people."
        ]

    # ---- Show the result page ----
    return render_template("result.html", status=result, tips=tips)

# ------------------------------------------------------------------
# 3) Run the server
# ------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
