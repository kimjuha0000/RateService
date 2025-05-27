import os
import urllib.request
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    BertTokenizer, BertForSequenceClassification,
    pipeline, Trainer, TrainingArguments
)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class NSMCDataset(Dataset):
    """NSMC 데이터셋을 위한 PyTorch Dataset 클래스"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class EnhancedSentimentAnalyzer:
    """향상된 감성 분석 클래스"""
    
    def __init__(self, model_name='klue/bert-base', use_pretrained_pipeline=True):
        """
        Args:
            model_name: 사용할 모델명 (기본값: klue/bert-base)
            use_pretrained_pipeline: 사전 훈련된 파이프라인 사용 여부
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_pretrained_pipeline = use_pretrained_pipeline
        
        if use_pretrained_pipeline:
            # 사전 훈련된 한국어 감성 분석 파이프라인 사용
            try:
                self.sentiment_pipeline = pipeline(
                    "text-classification",
                    model="snunlp/KR-ELECTRA-discriminator",
                    tokenizer="snunlp/KR-ELECTRA-discriminator",
                    device=0 if torch.cuda.is_available() else -1
                )
                print("KR-ELECTRA 모델 로드 완료")
            except:
                # 대안 모델 사용
                self.sentiment_pipeline = pipeline(
                    "text-classification",
                    model="klue/bert-base",
                    device=0 if torch.cuda.is_available() else -1
                )
                print("KLUE-BERT 모델 로드 완료")
        else:
            self.tokenizer = None
            self.model = None
    
    def download_data(self):
        """NSMC 데이터셋 다운로드"""
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
    
    def load_data(self):
        """NSMC 데이터셋 로드"""
        train_data = pd.read_csv("ratings_train.txt", sep="\t", encoding="utf-8")
        test_data = pd.read_csv("ratings_test.txt", sep="\t", encoding="utf-8")
        
        # 결측값 제거
        train_data = train_data.dropna()
        test_data = test_data.dropna()
        
        return train_data, test_data
    
    def train_custom_model(self, sample_size=10000):
        """커스텀 BERT 모델 훈련"""
        self.download_data()
        train_data, test_data = self.load_data()
        
        # 샘플 데이터 사용 (전체 데이터는 너무 크므로)
        if sample_size:
            train_data = train_data.sample(n=min(sample_size, len(train_data)))
            test_data = test_data.sample(n=min(sample_size//5, len(test_data)))
        
        # 토크나이저와 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=2
        )
        
        # 데이터셋 생성
        train_dataset = NSMCDataset(
            train_data['document'].values,
            train_data['label'].values,
            self.tokenizer
        )
        
        test_dataset = NSMCDataset(
            test_data['document'].values,
            test_data['label'].values,
            self.tokenizer
        )
        
        # 훈련 설정
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
        
        # 모델 훈련
        print("모델 훈련 시작...")
        trainer.train()
        
        # 평가
        results = trainer.evaluate()
        print(f"평가 결과: {results}")
        
        return self.model, self.tokenizer, test_data
    
    def analyze_sentiment_detailed(self, text):
        """
        텍스트의 상세한 감성 분석
        
        Returns:
            dict: {
                'sentiment': 'positive' or 'negative',
                'confidence': 신뢰도 (0-1),
                'score': 감성 점수 (0-100),
                'detailed_scores': {
                    'positive': 긍정 확률,
                    'negative': 부정 확률
                }
            }
        """
        if self.use_pretrained_pipeline:
            try:
                result = self.sentiment_pipeline(text)
                
                # 결과 정규화
                if isinstance(result, list):
                    result = result[0]
                
                label = result['label'].lower()
                confidence = result['score']
                
                # 감성 점수 계산 (0-100 스케일)
                if 'pos' in label or label == 'label_1':
                    sentiment = 'positive'
                    score = confidence * 100
                else:
                    sentiment = 'negative' 
                    score = (1 - confidence) * 100
                
                return {
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'score': score,
                    'detailed_scores': {
                        'positive': confidence if sentiment == 'positive' else 1 - confidence,
                        'negative': confidence if sentiment == 'negative' else 1 - confidence
                    }
                }
                
            except Exception as e:
                print(f"파이프라인 분석 오류: {e}")
                return self._fallback_analysis(text)
        
        else:
            return self._custom_model_analysis(text)
    
    def _custom_model_analysis(self, text):
        """커스텀 훈련된 모델을 사용한 분석"""
        if not self.model or not self.tokenizer:
            raise ValueError("모델이 훈련되지 않았습니다. train_custom_model()을 먼저 실행하세요.")
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            negative_prob = probabilities[0][0].item()
            positive_prob = probabilities[0][1].item()
        
        sentiment = 'positive' if positive_prob > negative_prob else 'negative'
        confidence = max(positive_prob, negative_prob)
        score = positive_prob * 100
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'score': score,
            'detailed_scores': {
                'positive': positive_prob,
                'negative': negative_prob
            }
        }
    
    def _fallback_analysis(self, text):
        """간단한 키워드 기반 fallback 분석"""
        positive_words = ['좋', '훌륭', '최고', '재미', '감동', '완벽', '추천', '만족']
        negative_words = ['나쁜', '최악', '실망', '지루', '형편없', '별로', '아쉬', '후회']
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count > neg_count:
            sentiment = 'positive'
            score = min(70 + pos_count * 5, 100)
        elif neg_count > pos_count:
            sentiment = 'negative'
            score = max(30 - neg_count * 5, 0)
        else:
            sentiment = 'neutral'
            score = 50
        
        confidence = min(0.6 + abs(pos_count - neg_count) * 0.1, 0.9)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'score': score,
            'detailed_scores': {
                'positive': score / 100,
                'negative': (100 - score) / 100
            }
        }
    
    def compute_movie_score_enhanced(self, reviews):
        """
        영화 리뷰들의 향상된 감성 점수 계산
        
        Returns:
            dict: {
                'overall_score': 전체 평균 점수,
                'weighted_score': 신뢰도 가중 평균 점수,
                'sentiment_distribution': 감성 분포,
                'review_count': 리뷰 개수,
                'confidence_avg': 평균 신뢰도
            }
        """
        if not reviews:
            return {
                'overall_score': 50,
                'weighted_score': 50,
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'review_count': 0,
                'confidence_avg': 0
            }
        
        results = []
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        print(f"총 {len(reviews)}개 리뷰 분석 중...")
        
        for i, review in enumerate(reviews):
            if i % 50 == 0:
                print(f"진행률: {i}/{len(reviews)} ({i/len(reviews)*100:.1f}%)")
            
            try:
                analysis = self.analyze_sentiment_detailed(str(review))
                results.append(analysis)
                sentiment_counts[analysis['sentiment']] += 1
            except Exception as e:
                print(f"리뷰 분석 오류 (인덱스 {i}): {e}")
                # 오류 시 중립 점수 할당
                results.append({
                    'sentiment': 'neutral',
                    'confidence': 0.5,
                    'score': 50,
                    'detailed_scores': {'positive': 0.5, 'negative': 0.5}
                })
                sentiment_counts['neutral'] += 1
        
        if not results:
            return {
                'overall_score': 50,
                'weighted_score': 50,
                'sentiment_distribution': sentiment_counts,
                'review_count': 0,
                'confidence_avg': 0
            }
        
        # 점수 계산
        scores = [r['score'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        overall_score = np.mean(scores)
        
        # 신뢰도 가중 평균
        total_weight = sum(confidences)
        if total_weight > 0:
            weighted_score = sum(score * conf for score, conf in zip(scores, confidences)) / total_weight
        else:
            weighted_score = overall_score
        
        return {
            'overall_score': round(overall_score, 2),
            'weighted_score': round(weighted_score, 2),
            'sentiment_distribution': sentiment_counts,
            'review_count': len(reviews),
            'confidence_avg': round(np.mean(confidences), 3)
        }

def main():
    """메인 실행 함수"""
    # 향상된 감성 분석기 초기화
    analyzer = EnhancedSentimentAnalyzer()
    
    # 테스트 데이터로 성능 검증
    analyzer.download_data()
    _, test_data = analyzer.load_data()
    
    # 샘플 리뷰로 테스트
    sample_reviews = test_data['document'].sample(n=100, random_state=42).astype(str).tolist()
    
    print("향상된 감성 분석 시작...")
    movie_analysis = analyzer.compute_movie_score_enhanced(sample_reviews)
    
    print("\n[향상된 영화 감성 분석 결과]")
    print(f"전체 평균 점수: {movie_analysis['overall_score']}/100")
    print(f"신뢰도 가중 평균 점수: {movie_analysis['weighted_score']}/100")
    print(f"리뷰 개수: {movie_analysis['review_count']}")
    print(f"평균 신뢰도: {movie_analysis['confidence_avg']}")
    print(f"감성 분포:")
    for sentiment, count in movie_analysis['sentiment_distribution'].items():
        percentage = (count / movie_analysis['review_count']) * 100
        print(f"  {sentiment}: {count}개 ({percentage:.1f}%)")
    
    # 개별 리뷰 분석 예시
    print("\n[개별 리뷰 분석 예시]")
    sample_text = "이 영화 정말 재미있고 감동적이었어요! 최고의 작품입니다."
    detailed_result = analyzer.analyze_sentiment_detailed(sample_text)
    print(f"리뷰: {sample_text}")
    print(f"감성: {detailed_result['sentiment']}")
    print(f"점수: {detailed_result['score']:.2f}/100")
    print(f"신뢰도: {detailed_result['confidence']:.3f}")
    print(f"상세 점수: 긍정 {detailed_result['detailed_scores']['positive']:.3f}, "
          f"부정 {detailed_result['detailed_scores']['negative']:.3f}")

if __name__ == "__main__":
    main()