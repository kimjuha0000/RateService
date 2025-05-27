# enhanced_aggregate_movie_sentiments.py
import cx_Oracle
from database import conn# database.py에 DB 연결 객체(conn)가 정의되어 있다고 가정
from new_train_model import EnhancedSentimentAnalyzer
import json
import time

def aggregate_movie_sentiments_enhanced(conn, analyzer):
    """
    movie_reviews 테이블에 저장된 각 리뷰의 감성 점수를 향상된 AI 모델로 재계산하여 업데이트한 후,
    영화별로 상세한 집계를 movie_sentiment_scores 테이블에 저장합니다.
    
    새로운 컬럼들:
      - sentiment_score: 기본 감성 점수 (0-100)
      - confidence_score: 예측 신뢰도 (0-1)
      - sentiment_label: 감성 라벨 ('positive', 'negative', 'neutral')
      - detailed_analysis: JSON 형태의 상세 분석 결과
    
    사전 작업:
      - movie_reviews 테이블에 새로운 컬럼들이 추가되어 있어야 합니다.
      - movie_sentiment_scores 테이블도 새로운 컬럼들을 포함해야 합니다.
    """
    cursor = conn.cursor()
    
    # 1. 테이블 컬럼 존재 여부 확인 및 추가
    print("테이블 스키마 확인 및 업데이트...")
    try:
        # movie_reviews 테이블에 새 컬럼 추가 (이미 존재하면 에러 무시)
        cursor.execute("ALTER TABLE movie_reviews ADD confidence_score NUMBER(5,3)")
        cursor.execute("ALTER TABLE movie_reviews ADD sentiment_label VARCHAR2(20)")
        cursor.execute("ALTER TABLE movie_reviews ADD detailed_analysis CLOB")
        print("movie_reviews 테이블에 새 컬럼 추가 완료")
    except cx_Oracle.DatabaseError as e:
        if "already exists" in str(e) or "name is already used" in str(e):
            print("movie_reviews 컬럼들이 이미 존재합니다.")
        else:
            print(f"movie_reviews 컬럼 추가 중 오류: {e}")
    
    try:
        # movie_sentiment_scores 테이블에 새 컬럼 추가
        cursor.execute("ALTER TABLE movie_sentiment_scores ADD weighted_avg_score NUMBER(5,2)")
        cursor.execute("ALTER TABLE movie_sentiment_scores ADD avg_confidence NUMBER(5,3)")
        cursor.execute("ALTER TABLE movie_sentiment_scores ADD positive_count NUMBER")
        cursor.execute("ALTER TABLE movie_sentiment_scores ADD negative_count NUMBER")
        cursor.execute("ALTER TABLE movie_sentiment_scores ADD neutral_count NUMBER")
        cursor.execute("ALTER TABLE movie_sentiment_scores ADD sentiment_distribution CLOB")
        print("movie_sentiment_scores 테이블에 새 컬럼 추가 완료")
    except cx_Oracle.DatabaseError as e:
        if "already exists" in str(e) or "name is already used" in str(e):
            print("movie_sentiment_scores 컬럼들이 이미 존재합니다.")
        else:
            print(f"movie_sentiment_scores 컬럼 추가 중 오류: {e}")
    
    # 2. 모든 리뷰에 대해 향상된 감성 점수 계산 및 업데이트
    select_sql = "SELECT review_id, review_text FROM movie_reviews"
    cursor.execute(select_sql)
    rows = cursor.fetchall()
    total_reviews = len(rows)
    print(f"총 {total_reviews}개의 리뷰에 대해 향상된 감성 분석을 수행합니다.")
    
    update_sql = """
    UPDATE movie_reviews 
    SET sentiment_score = :score, 
        confidence_score = :confidence,
        sentiment_label = :label,
        detailed_analysis = :details
    WHERE review_id = :rid
    """
    
    batch_size = 50  # 배치 단위로 처리
    start_time = time.time()
    
    for idx, (review_id, review_text) in enumerate(rows, start=1):
        try:
            # cx_Oracle.LOB인 경우 문자열로 변환
            review_str = str(review_text) if review_text else ""
            
            if not review_str.strip():
                # 빈 리뷰인 경우 기본값 할당
                analysis_result = {
                    'sentiment': 'neutral',
                    'confidence': 0.5,
                    'score': 50,
                    'detailed_scores': {'positive': 0.5, 'negative': 0.5}
                }
            else:
                # 향상된 감성 분석 수행
                analysis_result = analyzer.analyze_sentiment_detailed(review_str)
            
            # 상세 분석 결과를 JSON으로 저장
            detailed_json = json.dumps(analysis_result, ensure_ascii=False)
            
            cursor.execute(update_sql, {
                "score": analysis_result['score'],
                "confidence": analysis_result['confidence'],
                "label": analysis_result['sentiment'],
                "details": detailed_json,
                "rid": review_id
            })
            
            # 진행률 표시
            if idx % batch_size == 0 or idx == total_reviews:
                elapsed_time = time.time() - start_time
                avg_time_per_review = elapsed_time / idx
                estimated_total_time = avg_time_per_review * total_reviews
                remaining_time = estimated_total_time - elapsed_time
                
                print(f"{idx}/{total_reviews} 리뷰 분석 완료 "
                      f"({(idx/total_reviews)*100:.2f}%) "
                      f"- 예상 남은 시간: {remaining_time/60:.1f}분")
                
                # 배치마다 커밋
                conn.commit()
        
        except Exception as e:
            print(f"리뷰 ID {review_id} 분석 중 오류: {e}")
            # 오류 시 기본값으로 업데이트
            cursor.execute(update_sql, {
                "score": 50,
                "confidence": 0.5,
                "label": 'neutral',
                "details": '{"error": "analysis_failed"}',
                "rid": review_id
            })
    
    conn.commit()
    print("모든 리뷰의 향상된 감성 점수 업데이트 완료.")
    
    # 3. 영화별 상세 집계
    print("영화별 상세 집계 작업 시작...")
    
    # 기존 데이터 삭제
    cursor.execute("DELETE FROM movie_sentiment_scores")
    
    # 영화별 집계 쿼리
    aggregate_sql = """
    SELECT 
        movie_id,
        ROUND(AVG(sentiment_score), 2) as avg_score,
        ROUND(AVG(CASE WHEN confidence_score IS NOT NULL THEN sentiment_score * confidence_score END) / 
              AVG(CASE WHEN confidence_score IS NOT NULL THEN confidence_score END), 2) as weighted_avg_score,
        ROUND(AVG(confidence_score), 3) as avg_confidence,
        COUNT(*) as total_count,
        SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
        SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count,
        SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count
    FROM movie_reviews
    WHERE sentiment_score IS NOT NULL
    GROUP BY movie_id
    """
    
    cursor.execute(aggregate_sql)
    movie_stats = cursor.fetchall()
    
    # movie_sentiment_scores 테이블에 집계 결과 삽입
    insert_sql = """
    INSERT INTO movie_sentiment_scores (
        movie_id, avg_score, review_count, weighted_avg_score, avg_confidence,
        positive_count, negative_count, neutral_count, sentiment_distribution
    ) VALUES (
        :movie_id, :avg_score, :review_count, :weighted_avg_score, :avg_confidence,
        :positive_count, :negative_count, :neutral_count, :sentiment_distribution
    )
    """
    
    for stats in movie_stats:
        (movie_id, avg_score, weighted_avg_score, avg_confidence, total_count, 
         positive_count, negative_count, neutral_count) = stats
        
        # 감성 분포 JSON 생성
        sentiment_distribution = {
            "positive": {
                "count": int(positive_count or 0),
                "percentage": round((positive_count or 0) / total_count * 100, 2)
            },
            "negative": {
                "count": int(negative_count or 0),
                "percentage": round((negative_count or 0) / total_count * 100, 2)
            },
            "neutral": {
                "count": int(neutral_count or 0),
                "percentage": round((neutral_count or 0) / total_count * 100, 2)
            }
        }
        
        cursor.execute(insert_sql, {
            "movie_id": movie_id,
            "avg_score": avg_score,
            "review_count": total_count,
            "weighted_avg_score": weighted_avg_score,
            "avg_confidence": avg_confidence,
            "positive_count": positive_count or 0,
            "negative_count": negative_count or 0,
            "neutral_count": neutral_count or 0,
            "sentiment_distribution": json.dumps(sentiment_distribution, ensure_ascii=False)
        })
    
    conn.commit()
    print(f"총 {len(movie_stats)}개 영화의 향상된 감성 점수 집계 완료.")
    cursor.close()

def get_movie_sentiment_report(conn, movie_id=None, top_n=10):
    """
    영화 감성 분석 리포트 생성
    
    Args:
        movie_id: 특정 영화 ID (None이면 전체 상위 영화)
        top_n: 상위 N개 영화 조회
    """
    cursor = conn.cursor()
    
    if movie_id:
        # 특정 영화 상세 리포트
        cursor.execute("""
        SELECT m.movie_id, m.avg_score, m.weighted_avg_score, m.avg_confidence,
               m.review_count, m.positive_count, m.negative_count, m.neutral_count,
               m.sentiment_distribution
        FROM movie_sentiment_scores m
        WHERE m.movie_id = :movie_id
        """, {"movie_id": movie_id})
        
        result = cursor.fetchone()
        if result:
            print(f"\n[영화 ID {movie_id} 감성 분석 리포트]")
            print(f"평균 감성 점수: {result[1]}/100")
            print(f"신뢰도 가중 평균: {result[2]}/100")
            print(f"평균 신뢰도: {result[3]}")
            print(f"총 리뷰 수: {result[4]}")
            print(f"긍정 리뷰: {result[5]}개")
            print(f"부정 리뷰: {result[6]}개")
            print(f"중립 리뷰: {result[7]}개")
            
            if result[8]:
                distribution = json.loads(result[8])
                print("감성 분포:")
                for sentiment, data in distribution.items():
                    print(f"  {sentiment}: {data['count']}개 ({data['percentage']}%)")
    else:
        # 상위 영화들 리포트
        cursor.execute(f"""
        SELECT movie_id, avg_score, weighted_avg_score, review_count, avg_confidence
        FROM movie_sentiment_scores
        ORDER BY weighted_avg_score DESC, review_count DESC
        FETCH FIRST {top_n} ROWS ONLY
        """)
        
        results = cursor.fetchall()
        print(f"\n[상위 {top_n}개 영화 감성 점수 랭킹]")
        print("순위 | 영화ID | 평균점수 | 가중평균 | 리뷰수 | 신뢰도")
        print("-" * 50)
        for i, (movie_id, avg_score, weighted_avg, review_count, confidence) in enumerate(