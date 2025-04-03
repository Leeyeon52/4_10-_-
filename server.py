from flask import Flask, request, jsonify
import joblib
import pandas as pd  # 오타 수정

app = Flask(__name__)

# 학습된 모델 로드
try:
    model = joblib.load("lgbm_model.pkl")  # 저장된 모델 파일 로드
    f_name = getattr(model, "feature_name_", None)  # feature_name_ 존재 여부 확인
    if f_name:
        print("Model Features:", f_name)
except Exception as e:
    print("Error loading model:", e)
    model = None  # 모델 로드 실패 시 None 처리

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500  # 모델이 없을 경우 예외 처리
    
    try:
        data = request.get_json()  # JSON 데이터 받기
        if not data:
            return jsonify({'error': 'No input data provided'}), 400  # 데이터 없을 경우 예외 처리
        
        df = pd.DataFrame(data)  # JSON → DataFrame 변환
        prediction = model.predict(df)  # 예측 실행
        
        return jsonify({'prediction': prediction.tolist()})  # 예측 결과 반환
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # 예외 처리

if __name__ == '__main__':
    app.run('127.0.0.1', port=9999, debug=True)
