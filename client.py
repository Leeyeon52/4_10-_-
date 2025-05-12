import requests
import time
import pandas as pd

url = "http://127.0.0.1:9999/predict"

# ✅ 단일 테스트 예측
sample_input = [{
    "V1": -0.0403, "V2": 0.1, "V3": 0.2, "V4": 0.3, "V5": 0.4, "V6": 0.5, "V7": 0.6,
    "V8": 0.7, "V9": 0.8, "V10": 0.9, "V11": 1.0, "V12": 0.2, "V13": 0.3, "V14": 0.4,
    "V15": 0.5, "V16": 0.6, "V17": 0.7, "V18": 0.8, "V19": 0.9, "V20": 1.0,
    "V21": 0.1, "V22": 0.2, "V23": 0.3, "V24": 0.4, "V25": 0.5, "V26": 0.6,
    "V27": 0.7, "V28": 0.8, "Amount": 10.0
}]

try:
    res = requests.post(url, json=sample_input)
    response = res.json()
    prediction = response.get("prediction", [])
    probability = response.get("prediction_proba", [])

    print("▶ 단일 예측 결과:")
    print(" - 예측 (0: 정상, 1: 사기):", prediction[0])
    if probability:
        print(" - 예측 확률:", round(probability[0], 4))

except Exception as e:
    print("❌ 단일 예측 오류:", e)

# ✅ 반복 요청 처리 (최대 10,000건)
try:
    df = pd.read_csv("X_train_over.csv")

    if "Class" not in df.columns:
        raise ValueError("❌ 'Class' 컬럼이 데이터에 없습니다. 사기 여부를 알 수 없습니다.")

    cnt_ok = 0
    cnt_fraud = 0
    total_count = 0

    s_time = time.time()

    for idx, (_, row) in enumerate(df.iterrows()):
        if idx >= 10000:
            break

        input_json = row.drop("Class").to_dict()
        label = int(row["Class"])
        total_count += 1

        res = requests.post(url, json=[input_json])
        if res.status_code == 200:
            result = res.json()
            pred = result.get("prediction", [])[0]

            if pred == 1:
                cnt_fraud += 1
            else:
                cnt_ok += 1

            print(f"[{idx+1}] ✅ 정답: {label} → 예측: {pred} (확률: {round(result.get('prediction_proba', [0])[0], 4)})")

        else:
            print(f"[{idx+1}] ❌ HTTP Error:", res.status_code)

    e_time = time.time()
    print("\n📊 결과 요약")
    print(f"총 처리 건수: {total_count}")
    print(f"✅ 정상 거래 예측: {cnt_ok}")
    print(f"🚨 사기 거래 예측: {cnt_fraud}")
    print(f"🕒 실행 시간: {e_time - s_time:.2f}초")

except Exception as e:
    print("❌ 반복 처리 오류:", e)
