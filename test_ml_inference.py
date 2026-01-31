from src.ml.ml_guidance import ml_guidance

print("ML Guidance Status:", ml_guidance.status())

# Test F1 Feature
f1_feats = {
    "driver": 0, "period": 10, "need": 5, 
    "local_load": 2, "driver_load": 200, 
    "demand_gap": 3, "total_workers": 10
}
score_f1 = ml_guidance.f1_predict_score(f1_feats)
print(f"F1 Prediction: {score_f1}")

if score_f1 is None:
    print("❌ F1 Prediction failed (None)")
else:
    print("✅ F1 Prediction successful")
    
# Test F2 Feature
f2_feats = {
    "num_periods": 12, "uncovered_need": 5, 
    "avg_load": 4.5, "load_variance": 0.5, 
    "heur_total_workers": 10, "opt_total_workers": 9
}
score_f2 = ml_guidance.f2_predict_score(f2_feats)
print(f"F2 Prediction: {score_f2}")

if score_f2 is None:
    print("❌ F2 Prediction failed (None)")
else:
    print("✅ F2 Prediction successful")
