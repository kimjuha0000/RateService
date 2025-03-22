import re
import pickle
import torch
import torch.nn as nn
from konlpy.tag import Okt
from src.train import SentimentLSTM, pad_sequences_custom, sentiment_predict  # 기존에 정의된 함수 사용

# 불용어 리스트 (학습 시 사용한 것과 동일해야 함)
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
max_len = 30

# Okt 객체 재생성
okt = Okt()

# vocab 로드 (학습 시 저장한 vocabulary 파일)
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# 모델 생성 및 학습된 파라미터 로드
vocab_size = len(vocab) + 1
embed_size = 100
hidden_size = 128
num_layers = 1
dropout = 0.0
model = SentimentLSTM(vocab_size, embed_size, hidden_size, num_layers, dropout)
model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device('cpu')))
model.eval()

def main():
    print("문장을 입력하면 감성(긍정/부정)을 예측합니다. 종료하려면 'exit'를 입력하세요.\n")
    while True:
        sentence = input("문장 입력: ")
        if sentence.lower() == 'exit':
            break
        # train.py에 정의된 sentiment_predict 함수 사용
        sentiment_predict(sentence, vocab, model, okt, stopwords, max_len)

if __name__ == '__main__':
    main()
