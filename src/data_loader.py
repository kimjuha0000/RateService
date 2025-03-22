# data_loader.py
import os
import urllib.request
import pandas as pd
import numpy as np
import re

def download_data():
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
    train_data = pd.read_table("ratings_train.txt")
    test_data = pd.read_table("ratings_test.txt")
    print("훈련 샘플 개수:", len(train_data))
    print("테스트 샘플 개수:", len(test_data))
    return train_data, test_data

def clean_data(data):
    # 중복 제거 및 결측치 제거
    data.drop_duplicates(subset=['document'], inplace=True)
    data = data.dropna(how='any')
    # 한글과 공백만 남기기
    data['document'] = data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)
    data['document'] = data['document'].str.replace('^ +', "", regex=True)
    data['document'].replace('', np.nan, inplace=True)
    data = data.dropna(how='any')
    return data

if __name__ == '__main__':
    download_data()
    train, test = load_data()
    train = clean_data(train)
    test = clean_data(test)
    print("데이터 로드 및 클리닝 완료.")



