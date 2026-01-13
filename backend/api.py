from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf

# -------------------------
# Load model and scalers
# -------------------------
model = tf.keras.models.load_model("stone_column_ann_model.h5")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

app = FastAPI(title="Stone Column Design Backend")

# -------------------------
# Input schema
# -------------------------
class DesignInput(BaseModel):
    cu: float
    diameter: float
    length: float
    spacing_ratio: float
    encasement_stiffness: float = 0.0

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
def predict(data: DesignInput):

    X = np.array([[  
        data.cu,
        data.diameter,
        data.length,
        data.spacing_ratio,
        data.encasement_stiffness
    ]])

    X_scaled = scaler_X.transform(X)
    y_pred = model.predict(X_scaled)
    y = scaler_y.inverse_transform(y_pred)

    ultimate_stress = float(y[0][0])
    service_load = float(y[0][1])
    fos = float(y[0][2])

    # Safety logic
    if fos >= 2.5:
        status = "Safe"
    elif fos >= 2.0:
        status = "Marginal"
    else:
        status = "Unsafe"

    return {
        "ultimate_stress": round(ultimate_stress, 2),
        "service_load": round(service_load, 2),
        "factor_of_safety": round(fos, 2),
        "status": status
    }
