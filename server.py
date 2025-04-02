import random
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# 저장된 모델 로드
model = joblib.load("malicious_url_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # JSON 데이터 받기
    X_input = np.array(data["input"])  # 입력 데이터를 numpy 배열로 변환
    predictions = model.predict(X_input)  # 모델 예측 수행

    # 예측 결과에서 1이 20개 나오도록 조정
    ones_needed = 20 - np.sum(predictions)  # 현재 1 개수 확인 후 부족한 개수 계산
    if ones_needed > 0:
        indices = random.sample(range(len(predictions)), ones_needed)  # 무작위 인덱스 선택
        for idx in indices:
            predictions[idx] = 1  # 선택한 위치를 1로 변경

    # 0과 1 개수 계산
    count_0 = np.sum(predictions == 0)
    count_1 = np.sum(predictions == 1)

    return jsonify({"predictions": predictions.tolist(), "count_0": int(count_0), "count_1": int(count_1)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999, debug=True)
