import pickle

# Save
with open('models/crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Load (in app.py)
with open('models/crop_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
