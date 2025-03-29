# predict_text.py
import torch
import joblib
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel

def get_review_embedding(text, tokenizer, model, device):
    """
    단일 리뷰 텍스트에 대해 DistilBERT 임베딩을 생성합니다.
    마지막 은닉 상태의 평균값(pooling)을 임베딩으로 사용합니다.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
    return embedding

def compute_score(text, classifier, tokenizer, model, device):
    """
    입력된 리뷰 텍스트에 대해 분류기의 긍정 확률을 계산하고,
    0~100 범위 점수 및 이진 판정을 반환합니다.
    """
    emb = get_review_embedding(text, tokenizer, model, device)
    prob = classifier.predict_proba(emb.reshape(1, -1))[0, 1]
    score = prob * 100
    binary_pred = 1 if prob >= 0.5 else 0
    return score, binary_pred

def main():
    device = torch.device("cpu")
    # 저장된 분류기 로드
    classifier = joblib.load("classifier.pkl")
    # DistilBERT 토크나이저와 모델 로드
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.to(device)
    model.eval()
    
    print("텍스트 입력을 통해 감성 점수를 테스트합니다. (종료하려면 'q' 입력)")
    while True:
        text = input("리뷰 텍스트를 입력하세요: ")
        if text.strip().lower() == 'q':
            print("프로그램을 종료합니다.")
            break
        score, binary_pred = compute_score(text, classifier, tokenizer, model, device)
        sentiment = "긍정" if binary_pred == 1 else "부정"
        print(f"감성 점수: {score:.2f}% | 최종 판정: {sentiment}\n")

if __name__ == "__main__":
    main()
