import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from torch.nn.functional import softmax

# 1. CSV 파일 불러오기
df = pd.read_csv("~/Downloads/tsla_news.csv")

# 2. 뉴스 텍스트 결합 (headline + summary)
df["text"] = df["headline"].fillna("") + ". " + df["summary"].fillna("")

# 3. DistilBERT 모델 및 토크나이저 로드
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
model.eval()

# 4. 레이블 매핑: 모델의 출력(0~2) → 우리가 원하는 값으로 변환
label_map = {0: 0, 1: 1, 2: -1}  # 중립, 긍정, 부정

# 5. 배치 단위 추론 (배치 크기 조정 가능)
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
