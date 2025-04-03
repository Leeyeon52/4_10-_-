import requests
import time
import pandas as pd

# API URL
url = "http://127.0.0.1:9999/predict"

# 샘플 입력 데이터 (JSON 형식 오류 수정)
input_data = [{
    "V1": -0.0402962145973447, 
    "V29": 0.034481133146
}]

# API 요청 테스트
try:
    res = requests.post(url, json=input_data)
    print("result:", res.json())

    if res.status_code == 200:
        print("ok")
    else:
        print("HTTP Status Code:", res.status_code)

except requests.exceptions.RequestException as e:
    print("Request error:", e)


# ---- 데이터프레임을 API에 반복 요청 ----
df = pd.DataFrame()  # 여기에 실제 데이터 로드 필요 (예: pd.read_csv("data.csv"))

cnt_ok = 0
cnt_fraud = 0

s_time = time.time()

for index, row in df.iterrows():
    input_json = {
        "Amount_Scaled": row["Amount_Scaled"],
        "V1": row["V1"],
        "V2": row["V2"],
        "V3": row["V3"],
        "V4": row["V4"],
        "V5": row["V5"],
        "V6": row["V6"],
        "V7": row["V7"],
        "V8": row["V8"],
        "V9": row["V9"],
        "V10": row["V10"],
        "V11": row["V11"],
        "V12": row["V12"],
        "V13": row["V13"],
        "V14": row["V14"],
        "V15": row["V15"],
        "V16": row["V16"],
        "V17": row["V17"],
        "V18": row["V18"],
        "V19": row["V19"],
        "V20": row["V20"],
        "V21": row["V21"],
        "V22": row["V22"],
        "V23": row["V23"],
        "V24": row["V24"],
        "V25": row["V25"],  # V22 중복 오류 수정
        "V26": row["V26"],
        "V27": row["V27"],
        "V28": row["V28"],
        "V29": row["V29"],
    }

    try:
        res = requests.post(url, json=input_json)
        if res.status_code == 200:
            response_data = res.json()
            print("result:", response_data)
            
            prediction = response_data.get("prediction", [])
            for p in prediction:
                if p == 0:
                    cnt_ok += 1
                elif p == 1:
                    cnt_fraud += 1
        else:
            print("HTTP Status Code:", res.status_code)

    except Exception as err:
        print("Error:", err)

e_time = time.time()
print(f"Execution Time: {e_time - s_time:.2f} sec")
print(f"Normal Transactions: {cnt_ok}, Fraud Transactions: {cnt_fraud}")
