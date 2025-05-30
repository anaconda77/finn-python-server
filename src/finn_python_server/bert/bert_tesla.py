import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import softmax

# 1. CSV 파일 불러오기
df = pd.read_csv("~/Downloads/tsla_news.csv")

# 2. 뉴스 텍스트 결합
df["text"] = df["headline"].fillna("") + ". " + df["summary"].fillna("")

# 3. BERT 모델 및 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.eval()

# 4. 레이블 매핑: 모델은 0~2 출력 → 우리가 원하는 형식으로 변환
label_map = {0: 0, 1: 1, 2: -1}  # 중립, 긍정, 부정

# 5. 배치 단위 추론 (너무 많은 경우 메모리 제한 가능성 대비)
batch_size = 16
results = []

for i in range(0, len(df), batch_size):
    batch_texts = df["text"].iloc[i:i+batch_size].tolist()
    inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    mapped = [label_map[p.item()] for p in preds]
    results.extend(mapped)

# 6. 결과 저장
df["predicted_sentiment"] = results
df.to_csv("tsla_news_with_sentiment.csv", index=False)

print("✔️ 감정 예측 완료! → tsla_news_with_sentiment.csv 저장됨")
