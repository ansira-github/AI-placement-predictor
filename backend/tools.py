# backend/tools.py

import pandas as pd
import joblib
import os

# -----------------------------
# Load trained model + scaler
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "placement_model.pkl")
loaded_obj = joblib.load(MODEL_PATH)

# If pickle contains dict (model + scaler), unpack it
if isinstance(loaded_obj, dict):
    model = loaded_obj.get("model")
    scaler = loaded_obj.get("scaler", None)
else:
    model = loaded_obj
    scaler = None


# -----------------------------
# Prediction for single student
# -----------------------------
def predict_placement(cgpa, iq):
    """
    Predicts placement status based on CGPA and IQ.
    
    Args:
        cgpa (float): Student's CGPA (0-10)
        iq (int): Student's IQ score
        
    Returns:
        tuple: (prediction, probability, key_influence_factor)
    """
    input_data = pd.DataFrame([[cgpa, iq]], columns=['cgpa', 'iq'])

    # Apply scaler if available
    if scaler is not None:
        input_data = scaler.transform(input_data)

    # Make prediction
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])  # probability of being placed

    # Determine key influence factor
    try:
        coefficients = model.coef_[0]
        cgpa_influence = abs(coefficients[0] * cgpa)
        iq_influence = abs(coefficients[1] * iq)
        key_factor = "CGPA" if cgpa_influence > iq_influence else "IQ"
    except Exception:
        key_factor = "CGPA" if cgpa > 7.0 else "IQ"

    return prediction, probability, key_factor


# -----------------------------
# Personalized advice
# -----------------------------
def get_placement_advice(cgpa, iq, prediction):
    advice = []

    if prediction == 0:  # Not placed
        advice.append("🎯 **Focus Areas for Improvement:**")

        if cgpa < 7.0:
            advice.append(f"• Your CGPA ({cgpa:.2f}) is below the typical threshold. Aim for 7.5+")
        if iq < 110:
            advice.append(f"• Work on problem-solving skills (current IQ: {iq})")

        advice.append("• Build projects and gain practical experience")
        advice.append("• Practice coding on platforms like LeetCode, HackerRank")
        advice.append("• Develop soft skills and communication")
    else:  # Placed
        advice.append("✅ **You're on track! Keep it up:**")
        advice.append(f"• Strong CGPA: {cgpa:.2f}")
        advice.append(f"• Good aptitude score: {iq}")
        advice.append("• Continue building your portfolio")
        advice.append("• Prepare for technical interviews")

    return "\n".join(advice)


# -----------------------------
# Scenario analysis
# -----------------------------
def analyze_improvement_scenarios(cgpa, iq):
    return {
        "current": predict_placement(cgpa, iq),
        "cgpa_improved": predict_placement(min(cgpa + 0.5, 10.0), iq),
        "iq_improved": predict_placement(cgpa, iq + 10),
        "both_improved": predict_placement(min(cgpa + 0.5, 10.0), iq + 10),
    }


# -----------------------------
# Bulk CSV prediction (optional)
# -----------------------------
def bulk_predict(file_path):
    """
    Predicts placement outcomes for multiple students in a CSV file.
    CSV must have columns: cgpa, iq
    """
    df = pd.read_csv(file_path)

    if scaler is not None:
        features = scaler.transform(df[['cgpa', 'iq']])
    else:
        features = df[['cgpa', 'iq']]

    df["prediction"] = model.predict(features)
    df["probability"] = model.predict_proba(features)[:, 1]

    return df
