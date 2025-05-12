from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# ✅ 모델 로드
try:
    model = joblib.load("LGBMClassifier.pkl")
    feature_names = getattr(model, "feature_name_", None)
    if feature_names:
        print("✅ Model Features:", feature_names)
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# ✅ sigmoid 함수 (확률 변환용)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        df = pd.DataFrame(data)

        # ✅ 특성 검증
        if feature_names and set(df.columns) != set(feature_names):
            return jsonify({
                'error': 'Feature mismatch!',
                'expected_features': feature_names,
                'received_features': list(df.columns),
                'missing_features': list(set(feature_names) - set(df.columns)),
                'extra_features': list(set(df.columns) - set(feature_names))
            }), 400

        # ✅ 예측
        raw_preds = model.predict(df)
        probs = sigmoid(raw_preds)

        # ✅ 임계값 수정: 0.5로 설정
        final_preds = (probs > 0.5).astype(int)

        return jsonify({
            'prediction': final_preds.tolist(),
            'prediction_proba': probs.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run('127.0.0.1', port=9999, debug=True)
