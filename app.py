from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
# import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os

# 🔕 TensorFlow warnings kam karne ke liye
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# ==============================
# 📁 IMAGE UPLOAD SETUP
# ==============================
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# ✅ LOAD DATASET (SCALER)
# ==============================
try:
    dataset = pd.read_csv("upi_fraud_dataset.csv")
    X = dataset.iloc[:, :10].values

    scaler = StandardScaler()
    scaler.fit(X)

    print("✅ Dataset loaded")

except Exception as e:
    print("❌ Dataset load error:", e)
    scaler = None

# ==============================
# ✅ LOAD MODEL
# ==============================
try:
    # model = tf.keras.models.load_model("filesuse/project_model1.h5")
    print("✅ Model loaded")

except Exception as e:
    print("❌ Model load error:", e)
    model = None

# ==============================
# 🏠 HOME
# ==============================
@app.route('/')
def first():
    return render_template('first.html')

# ==============================
# 🔐 LOGIN
# ==============================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if username == "admin" and password == "admin":
            return redirect(url_for('upload'))
        else:
            return render_template('login.html', error="Invalid Credentials")

    return render_template('login.html')

# ==============================
# 📤 UPLOAD PAGE
# ==============================
@app.route('/upload')
def upload():
    return render_template('upload.html')

# ==============================
# 📂 PREVIEW (IMAGE)
# ==============================
@app.route('/preview', methods=['POST'])
def preview():
    file = request.files.get('imagefile')

    if not file:
        return "No file uploaded ❌"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    return render_template('preview.html', image_path=filepath)

# ==============================
# 🤖 PREDICTION FORM
# ==============================
@app.route('/prediction1')
def prediction1():
    return render_template('index.html')

# ==============================
# 📊 CHART PAGE
# ==============================
@app.route('/chart')
def chart():
    return render_template('chart.html')

# ==============================
# 🎯 ML DETECTION
# ==============================
@app.route('/detect', methods=['POST'])
def detect():
    try:
        if scaler is None or model is None:
            return "Model or dataset not loaded ❌"

        trans_datetime = pd.to_datetime(request.form.get("trans_datetime"))

        v1 = trans_datetime.hour
        v2 = trans_datetime.day
        v3 = trans_datetime.month
        v4 = trans_datetime.year
        v5 = int(request.form.get("category"))
        v6 = float(request.form.get("card_number"))
        dob = pd.to_datetime(request.form.get("dob"))
        v7 = (trans_datetime - dob).days / 365.25
        v8 = float(request.form.get("trans_amount"))
        v9 = int(request.form.get("state"))
        v10 = int(request.form.get("zip"))

        x_test = np.array([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]).reshape(1, -1)

        x_scaled = scaler.transform(x_test)

        prediction = model.predict(x_scaled)

        # 🔥 CONFIDENCE %
        prob = float(prediction[0][0]) * 100

        if prob > 50:
            result = "FRAUD TRANSACTION 🚨"
        else:
            result = "VALID TRANSACTION ✅"

        # 🔥 SIMPLE REASON (DEMO)
        if v8 > 50000:
            reason = "High transaction amount"
        else:
            reason = "Normal transaction pattern"

        return render_template(
            'result.html',
            OUTPUT=result,
            PROB=round(prob, 2),
            REASON=reason
        )

    except Exception as e:
        return f"Error: {e}"

# ==============================
# 🚀 RUN SERVER
# ==============================
if __name__ == "__main__":
    print("🚀 ML App Running...")
    app.run(debug=True)