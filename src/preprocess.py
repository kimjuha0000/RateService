# preprocess.py
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from konlpy.tag import Okt

def build_vocab(tokenized_texts, threshold=3):
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    vocab = {}
    for word, count in counter.items():
        if count >= threshold:
            # 인덱스 0은 패딩용으로 예약
            vocab[word] = len(vocab) + 1
    return vocab, counter

def texts_to_sequences(tokenized_texts, vocab):
    sequences = []
    for tokens in tokenized_texts:
        seq = [vocab[word] for word in tokens if word in vocab]
        sequences.append(seq)
    return sequences

def pad_sequences_custom(sequences, maxlen, padding='pre', truncating='pre'):
    padded = []
    for seq in sequences:
        if len(seq) < maxlen:
            pad_len = maxlen - len(seq)
            pad = [0] * pad_len
            seq = pad + seq if padding == 'pre' else seq + pad
        elif len(seq) > maxlen:
            seq = seq[-maxlen:] if truncating == 'pre' else seq[:maxlen]
        padded.append(seq)
    return np.array(padded)

def preprocess_text(train_data, test_data, max_len=30, threshold=3):
    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
    okt = Okt()
    
    X_train = []
    for sentence in tqdm(train_data['document'], desc="Train 토큰화"):
        tokens = okt.morphs(sentence, stem=True)
        filtered = [word for word in tokens if word not in stopwords]
        X_train.append(filtered)
        
    X_test = []
    for sentence in tqdm(test_data['document'], desc="Test 토큰화"):
        tokens = okt.morphs(sentence, stem=True)
        filtered = [word for word in tokens if word not in stopwords]
        X_test.append(filtered)
    
    # Vocabulary 구축
    vocab, counter = build_vocab(X_train, threshold=threshold)
    vocab_size = len(vocab) + 1  # 0번 인덱스는 패딩
    print("Vocabulary size:", vocab_size)
    
    X_train_seq = texts_to_sequences(X_train, vocab)
    X_test_seq = texts_to_sequences(X_test, vocab)
    
    # 빈 시퀀스 제거 및 레이블 인덱스 맞추기
    train_indices = [i for i, seq in enumerate(X_train_seq) if len(seq) > 0]
    X_train_seq = [X_train_seq[i] for i in train_indices]
    y_train = train_data['label'].values[train_indices]
    
    test_indices = [i for i, seq in enumerate(X_test_seq) if len(seq) > 0]
    X_test_seq = [X_test_seq[i] for i in test_indices]
    y_test = test_data['label'].values[test_indices]
    
    # 리뷰 길이 분포 시각화 (옵션)
    lengths = [len(seq) for seq in X_train_seq]
    print("최대 리뷰 길이:", max(lengths))
    print("평균 리뷰 길이:", sum(lengths) / len(lengths))
    plt.hist(lengths, bins=50)
    plt.xlabel('샘플 길이')
    plt.ylabel('샘플 개수')
    plt.title("리뷰 길이 분포")
    plt.show()
    
    below_threshold = sum(1 for seq in X_train_seq if len(seq) <= max_len)
    print(f"전체 샘플 중 길이가 {max_len} 이하인 샘플 비율: {below_threshold/len(X_train_seq)*100:.2f}%")
    
    X_train_pad = pad_sequences_custom(X_train_seq, max_len)
    X_test_pad = pad_sequences_custom(X_test_seq, max_len)
    
    return (X_train_pad, y_train), (X_test_pad, y_test), vocab, okt, stopwords, max_len



if __name__ == '__main__':
    print("이 파일은 모듈로 사용하세요.")
