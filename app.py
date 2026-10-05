import pandas as pd
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)
saved = pickle.load(open("model.pkl", "rb"))

scaler, knn = saved["scaler"], saved["knn"]
FEATURES, CLASSES = saved["features"], saved["classes"]

def key_of(feature):
    return feature.replace(" ", "_")

def pretty(feature):
    for suffix in ("_mean", "_se", "_worst"):
        feature = feature.replace(suffix, "")

    return feature.replace("_", " ").title()

GROUPS = []
for title, suffix in [("Mean values", "_mean"), ("Standard error values", "_se"), ("Worst values", "_worst")]:
    GROUPS.append({"title":title,
                   "fields": [{"key": key_of(f), "feature": f, "label": pretty(f)}
                              for f in FEATURES if f.endswith(suffix)]})

data = pd.read_csv("diagnosis.csv")
samples = {
    "benign": {key_of(f): v for f, v in data[data.diagnosis == "B"][FEATURES].median().round(4).items()},
    "malignant": {key_of(f): v for f,v in data[data.diagnosis == "M"][FEATURES].median().round(4).items()},
}

@app.route("/", methods = ["POST", "GET"])
def home():
    result, error = None, None
    if request.method == "POST":
        try:
            values = {}
            for f in FEATURES:
                v = float(request.form[key_of(f)])

                if v < 0:
                    raise ValueError
                values[f] = v
            row = pd.DataFrame([values])[FEATURES]
            row_scaled = scaler.transform(row)
            pred = int(knn.predict(row_scaled)[0])
            prob = float(knn.predict_proba(row_scaled)[0][pred]) * 100
            is_malignant = CLASSES[pred] == "M"

            result = {"label": "Malignant" if is_malignant else "Benign", 
                      "is_malignant": is_malignant, "confidence": round(prob,1)}

        except (ValueError, KeyError):
            error = "Please fill every field with a valid non-negative number."

    return render_template("index.html", groups = GROUPS, result=result, error=error,
                           samples = samples, k=knn.n_neighbors,
                           accuracy=round(saved["accuracy"] * 100,1), n_features=len(FEATURES))


if __name__ == "__main__":
    app.run(debug=True)

