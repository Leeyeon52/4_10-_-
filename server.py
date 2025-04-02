from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# 학습된 모델 로드
model = joblib.load("malicious_url_model.pkl")  # 저장된 모델 파일

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # JSON 데이터 받기
    urls = data.get("urls", [])  # 'urls' 키에서 리스트 가져오기
    
    if not urls:
        return jsonify({"error": "No data provided"}), 400
    try:
    # 데이터를 NumPy 배열로 변환 (모델 입력 형식에 맞게 조정 필요)
        X_input = np.array([list(d.values()) for d in data])  # 예시 (전처리 필요)
    
        predictions = model.predict(X_input)  # 모델 예측 수행
    
    # 결과를 JSON 형식으로 반환
        result = {
            "predictions": predictions.tolist(),
            "count_0": int((predictions == 0).sum()),
            "count_1": int((predictions == 1).sum())
    }
    
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run('127.0.0.1', port=9999, debug=False)
