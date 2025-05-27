from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

DB_USER = "root"
DB_PASSWORD = "1436"
DB_HOST = "192.168.0.187"
DB_PORT = "3306"
DB_NAME = "test"
# 학습된 모델 로드
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "파일이 없습니다!"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "파일 이름이 없습니다!"}), 400
    
    try:
        df = pd.read_csv(file)
        df.columns = [
            'ID', '설립연도', '국가', '분야', '투자단계', '직원 수', '인수여부', 
            '상장여부', '고객수(백만명)', '총 투자금(억원)', '연매출(억원)', 
            'SNS 팔로우 수(백만명)', '기업가치(백억원)', '성공확률'
        ]  # 모델 예측 수행
    
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
