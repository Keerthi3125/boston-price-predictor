from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model, scaler = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            crim = float(request.form["crim"])
            rm = float(request.form["rm"])
            lstat = float(request.form["lstat"])
            ptratio = float(request.form["ptratio"])
            age = float(request.form["age"])
            input_data = scaler.transform([[crim, rm, lstat, ptratio, age] + [0]*8])
            prediction = model.predict(input_data)[0]
        except:
            prediction = "Invalid input."
    return render_template("index.html", prediction=prediction)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
