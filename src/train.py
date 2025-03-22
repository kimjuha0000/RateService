# train.py
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, TensorDataset, random_split

from data_loader import download_data, load_data, clean_data
from preprocess import preprocess_text, pad_sequences_custom

# PyTorch 모델 정의: Embedding -> LSTM -> Dense
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, 
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(h_n[-1])
        out = self.sigmoid(out)
        return out

def train_model(model, X_train, y_train, batch_size=64, epochs=15, patience=4, learning_rate=0.001):
    X_tensor = torch.tensor(X_train, dtype=torch.long)
    y_tensor = torch.tensor(y_train, dtype=torch.float).unsqueeze(1)
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # 80:20 비율로 훈련/검증 분할
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= train_size
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= val_size
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} / Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("최적 모델 저장됨.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

def evaluate_model(model, X_test, y_test, batch_size=64):
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.long)
    y_tensor = torch.tensor(y_test, dtype=torch.float).unsqueeze(1)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            outputs = model(batch_x)
            preds = (outputs > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    accuracy = correct / total
    print(f"\n테스트 정확도: {accuracy:.4f}")

def sentiment_predict(new_sentence, vocab, model, okt, stopwords, max_len):
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', new_sentence)
    tokens = okt.morphs(new_sentence, stem=True)
    filtered = [word for word in tokens if word not in stopwords]
    seq = [vocab[word] for word in filtered if word in vocab]
    if len(seq) == 0:
        print("입력된 문장이 너무 짧거나 분석할 수 없습니다.")
        return
    seq_pad = pad_sequences_custom([seq], max_len)
    seq_tensor = torch.tensor(seq_pad, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        output = model(seq_tensor)
    score = output.item()
    if score > 0.5:
        print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))

def main():
    # 1. 데이터 다운로드 및 로드/클리닝
    download_data()
    train_data, test_data = load_data()
    train_data = clean_data(train_data)
    test_data = clean_data(test_data)
    
    # 2. 전처리 (토큰화, 시퀀스 변환, 패딩 등)
    (X_train, y_train), (X_test, y_test), vocab, okt, stopwords, max_len = preprocess_text(
        train_data, test_data, max_len=30, threshold=3
    )
    
    # 3. 모델 생성 및 훈련
    vocab_size = len(vocab) + 1
    embed_size = 100
    hidden_size = 128
    num_layers = 1
    dropout = 0.0
    model = SentimentLSTM(vocab_size, embed_size, hidden_size, num_layers, dropout)
    model = train_model(model, X_train, y_train, batch_size=64, epochs=15, patience=4, learning_rate=0.001)
    
    # 4. 평가
    evaluate_model(model, X_test, y_test, batch_size=64)
    
    # 5. vocab.pkl 저장 (전처리 시 생성한 단어 집합 저장)
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print("vocab.pkl 파일이 저장되었습니다.")
    
    # 6. 예측 테스트
    print("\n예측 결과:")
    sentiment_predict('이 영화 개꿀잼 ㅋㅋㅋ', vocab, model, okt, stopwords, max_len)
    sentiment_predict('이 영화 핵노잼 ㅠㅠ', vocab, model, okt, stopwords, max_len)
    sentiment_predict('이딴게 영화냐 ㅉㅉ', vocab, model, okt, stopwords, max_len)
    sentiment_predict('감독 뭐하는 놈이냐?', vocab, model, okt, stopwords, max_len)
    sentiment_predict('와 개쩐다 정말 세계관 최강자들의 영화다', vocab, model, okt, stopwords, max_len)
    sentiment_predict('엥 이거 뭐냐', vocab, model, okt, stopwords, max_len)
    sentiment_predict('집에가고싶다', vocab, model, okt, stopwords, max_len)
    sentiment_predict('재미없있네요', vocab, model, okt, stopwords, max_len)

if __name__ == '__main__':
    main()
