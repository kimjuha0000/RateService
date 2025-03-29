# train_model.py
import os
import urllib.request
import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def download_data():
    """
    NSMC 데이터셋(리뷰 파일)을 로컬에 다운로드합니다.
    """
    if not os.path.exists("ratings_train.txt"):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
            filename="ratings_train.txt"
        )
    if not os.path.exists("ratings_test.txt"):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
            filename="ratings_test.txt"
        )
    print("데이터 다운로드 완료.")

def load_data():
    """
    NSMC 데이터셋을 로드합니다.
    ratings_train.txt와 ratings_test.txt는 탭 구분 파일이며,
    'document' (리뷰 텍스트)와 'label' (1: 긍정, 0: 부정) 컬럼을 포함합니다.
    """
    train_data = pd.read_csv("ratings_train.txt", sep="\t", encoding="utf-8")
    test_data = pd.read_csv("ratings_test.txt", sep="\t", encoding="utf-8")
    return train_data, test_data

def create_embeddings_for_dataset_batched(texts, tokenizer, model, device, batch_size=32):
    """
    주어진 텍스트 리스트에 대해 배치 단위로 DistilBERT 임베딩을 생성합니다.
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # 마지막 은닉 상태의 평균값(pooling) → (batch_size, hidden_size)
        batch_embeds = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeds)
    embeddings = np.vstack(embeddings)
    return embeddings

def train_classifier():
    """
    데이터를 다운로드 및 로드한 후 DistilBERT 임베딩을 생성하고,
    로지스틱 회귀 분류기를 학습합니다.
    
    학습이 완료되면 분류기를 'classifier.pkl'로 저장합니다.
    """
    download_data()
    train_data, test_data = load_data()
    
    X_train_texts = train_data['document'].astype(str).tolist()
    y_train = train_data['label'].values
    X_test_texts = test_data['document'].astype(str).tolist()
    y_test = test_data['label'].values
    
    device = torch.device("cpu")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model_bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model_bert.to(device)
    model_bert.eval()
    
    print("임베딩 생성 중 (배치 사이즈 32)...")
    X_train_embeddings = create_embeddings_for_dataset_batched(X_train_texts, tokenizer, model_bert, device, batch_size=32)
    X_test_embeddings = create_embeddings_for_dataset_batched(X_test_texts, tokenizer, model_bert, device, batch_size=32)
    print("임베딩 생성 완료.")
    
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train_embeddings, y_train)
    
    y_pred = classifier.predict(X_test_embeddings)
    acc = accuracy_score(y_test, y_pred)
    print("로지스틱 회귀 정확도 (DistilBERT 임베딩 사용): {:.2f}%".format(acc * 100))
    print(classification_report(y_test, y_pred))
    
    # 학습된 분류기를 저장
    joblib.dump(classifier, "classifier.pkl")
    print("학습된 분류기가 'classifier.pkl'로 저장되었습니다.")
    
    return classifier, tokenizer, model_bert, device

def main():
    train_classifier()

if __name__ == "__main__":
    main()
