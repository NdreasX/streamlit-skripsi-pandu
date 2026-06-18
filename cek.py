import joblib

# Load encodernya
encoders = joblib.load("label_encoders.pkl")

# Intip isi aslinya
print("=========================================")
print("INI LABEL ASLINYA BRAY:")
print(encoders["funnel"].classes_)
print("=========================================")