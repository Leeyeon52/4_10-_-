import requests
import time
import pandas as pd

# ✅ API URL
url = "http://127.0.0.1:9999/predict"

# ✅ 테스트용 샘플 (전체 29개 피처 포함)
sample_input = [{
    "Amount_Scaled": 0.15,
    "V1": -0.04, "V2": 0.12, "V3": -0.45, "V4": 0.67, "V5": -0.78,
    "V6": 0.89, "V7": -1.2, "V8": 0.34, "V9": -0.56, "V10": 0.76,
    "V11": -0.98, "V12": 0.45, "V13": -0.23, "V14": 0.78, "V15": -1.1,
    "V16": 0.36, "V17": -0.49, "V18": 0.27, "V19": -0.15, "V20": 0.89,
    "V21": -0.34, "V22": 0.12, "V23": -0.44, "V24": 0.67, "V25": -0.78,
    "V26": 0.89, "V27": -1.2, "V28": 0.34  # ✅ V28 포함 (총 29개 피처)
}]

# ✅ API 테스트 요청
try:
    res = requests.post(url, json=sample_input)
    print("✅ result:", res.json())

    if res.status_code == 200:
        print("🎉 정상 응답")
    else:
        print("❌ HTTP Status Code:", res.status_code)

except requests.exceptions.RequestException as e:
    print("❌ Request error:", e)

# ✅ CSV 데이터 반복 요청
try:
    df = pd.read_csv("X_train_over.csv")
    print("✅ X_train_over.csv 로드 성공")

    # ✅ 필요 없는 라벨 컬럼('Class')이 있다면 제거
    if 'Class' in df.columns:
        df = df.drop(columns=['Class'])

except FileNotFoundError:
    print("❌ X_train_over.csv 파일이 없습니다.")
    df = pd.DataFrame()

cnt_ok = 0
cnt_fraud = 0

s_time = time.time()

# ✅ 정확한 컬럼 순서로 요청 (총 29개)
columns = ['Amount_Scaled'] + [f"V{i}" for i in range(1, 29)]  # ✅ V28까지 포함

for index, row in df.iterrows():
    input_json = [row[columns].to_dict()]  # 반드시 리스트로 보냄

    try:
        res = requests.post(url, json=input_json)
        if res.status_code == 200:
            response_data = res.json()
            prediction = response_data.get("prediction", [])
            for p in prediction:
                if p == 0:
                    cnt_ok += 1
                elif p == 1:
                    cnt_fraud += 1
        else:
            print(f"❌ 오류 응답: {res.status_code}, 내용: {res.text}")

    except Exception as err:
        print("❌ API 요청 중 예외 발생:", err)

e_time = time.time()
print(f"⏳ 실행 시간: {e_time - s_time:.2f} 초")
print(f"✅ 정상 거래: {cnt_ok}, 🚨 사기 거래: {cnt_fraud}")
