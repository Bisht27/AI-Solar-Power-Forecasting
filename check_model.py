import joblib

MODEL_PATH = "model/best_random_forest.pkl"

model = joblib.load(MODEL_PATH)

print("=" * 60)
print("MODEL INFORMATION")
print("=" * 60)

print("Model type:")
print(type(model))

print("\nNumber of features:")
print(getattr(model, "n_features_in_", "Not available"))

print("\nFeature names:")
print(getattr(model, "feature_names_in_", "Not available"))

print("\nModel parameters:")
print(model.get_params())

print("\nModel:")
print(model)