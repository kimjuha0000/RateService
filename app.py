import cx_Oracle
from flask import Flask, jsonify, request, send_from_directory, g
from flask_cors import CORS
from train_model import train_model, compute_movie_score
from collections import Counter
import re

# ─── Flask 앱 생성 ───────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder='.',       # 현재 디렉터리를 정적 파일 루트로 설정
    static_url_path=''       # URL 경로를 파일명과 1:1 매핑
)
# 대시보드와 React(또는 파일://) 호출을 모두 허용
CORS(app, origins=["http://localhost:3000", "http://localhost:5000", "file://"])

# ─── DB 연결 정보 ──────────────────────────────────────────────────────────────
DB_USER = "SYSTEM"
DB_PASSWORD = "kk1801"
DB_DSN = "127.0.0.1:1521/XE"

def get_connection():
    """요청마다 새로운 Oracle DB 커넥션 생성"""
    return cx_Oracle.connect(user=DB_USER, password=DB_PASSWORD, dsn=DB_DSN)

# ─── 요청 전/후 DB 관리 ─────────────────────────────────────────────────────────
@app.before_request
def open_db():
    g.conn = get_connection()
    g.cursor = g.conn.cursor()

@app.teardown_request
def close_db(exception=None):
    cursor = getattr(g, 'cursor', None)
    if cursor:
        cursor.close()
    conn = getattr(g, 'conn', None)
    if conn:
        conn.close()

# ─── 모델 초기화 ────────────────────────────────────────────────────────────────
model = None
vectorizer = None

def init_model():
    """앱 시작 시 AI 모델·벡터라이저 로드"""
    global model, vectorizer
    print("AI 모델 로딩 중.")
    model, vectorizer, _ = train_model()
    print("AI 모델 로딩 완료")

# ─── 정적 대시보드 제공 ────────────────────────────────────────────────────────
@app.route('/')
def serve_dashboard():
    # 현재 디렉터리에서 dashboard1.html 반환
    return send_from_directory('.', 'dashboard1.html')

# ─── 대시보드용 API 엔드포인트 ─────────────────────────────────────────────────
@app.route('/api/dashboard/summary')
def get_dashboard_summary():
    """총 영화 수·리뷰 수·평균 감성 점수·분석 완료율 반환"""
    try:
        c = g.cursor
        c.execute("SELECT COUNT(*) FROM movies")
        total_movies = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM movie_reviews")
        total_reviews = c.fetchone()[0]
        c.execute("SELECT AVG(avg_score) FROM movie_sentiment_scores")
        avg_sentiment = round(float(c.fetchone()[0] or 0), 1)
        c.execute(
            "SELECT COUNT(CASE WHEN sentiment_score IS NOT NULL THEN 1 END) * 100.0 / COUNT(*) FROM movie_reviews"
        )
        analysis_rate = round(float(c.fetchone()[0] or 0), 0)
        return jsonify({
            'totalMovies': total_movies,
            'totalReviews': total_reviews,
            'avgSentiment': avg_sentiment,
            'analysisRate': int(analysis_rate)
        })
    except Exception as e:
        print(f"Dashboard summary error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/movies/recent')
def get_recent_movies():
    """최근 등록된 영화 10개 목록 반환"""
    try:
        c = g.cursor
        c.execute("""
            SELECT m.movie_id, m.title, m.insert_date,
                   mss.avg_score, mss.review_count
            FROM movies m
            LEFT JOIN movie_sentiment_scores mss ON m.movie_id = mss.movie_id
            ORDER BY m.insert_date DESC
            FETCH FIRST 10 ROWS ONLY
        """)
        movies = []
        for movie_id, title, insert_date, avg_score, review_count in c.fetchall():
            movies.append({
                'id': movie_id,
                'title': title,
                'date': insert_date.strftime('%Y-%m-%d') if insert_date else '',
                'score': round(float(avg_score or 0), 1),
                'reviewCount': review_count or 0
            })
        return jsonify(movies)
    except Exception as e:
        print(f"Recent movies error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/movies/top')
def get_top_movies():
    """리뷰 10개 이상인 영화 중 감성 점수 상위 5개 반환"""
    try:
        c = g.cursor
        c.execute("""
            SELECT m.title, mss.avg_score
            FROM movies m
            JOIN movie_sentiment_scores mss ON m.movie_id = mss.movie_id
            WHERE mss.review_count >= 10
            ORDER BY mss.avg_score DESC
            FETCH FIRST 5 ROWS ONLY
        """)
        top = []
        for idx, (title, avg_score) in enumerate(c.fetchall(), 1):
            top.append({
                'rank': idx,
                'title': title,
                'score': round(float(avg_score), 1)
            })
        return jsonify(top)
    except Exception as e:
        print(f"Top movies error: {e}")
        return jsonify({'error': str(e)}), 500

# ★ 누락된 API 추가 ★
@app.route('/api/movies/search')
def search_movies():
    """영화 제목으로 검색"""
    try:
        query = request.args.get('query', '').strip()
        if not query or len(query) < 2:
            return jsonify([])
            
        c = g.cursor
        c.execute("""
            SELECT m.movie_id, m.title
            FROM movies m
            WHERE UPPER(m.title) LIKE UPPER(:query)
            ORDER BY m.title
            FETCH FIRST 10 ROWS ONLY
        """, {"query": f"%{query}%"})
        
        results = []
        for movie_id, title in c.fetchall():
            results.append({
                'id': movie_id,
                'title': title
            })
        return jsonify(results)
    except Exception as e:
        print(f"Search movies error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sentiment/trend')
def get_sentiment_trend():
    """최근 30일 감성 점수 트렌드(일별 평균·건수) 반환"""
    try:
        c = g.cursor
        c.execute("""
            SELECT TRUNC(review_date) as day,
                   AVG(sentiment_score), COUNT(*)
            FROM movie_reviews
            WHERE review_date >= SYSDATE - 30 AND sentiment_score IS NOT NULL
            GROUP BY TRUNC(review_date)
            ORDER BY day
        """)
        trend = []
        for day, avg_s, cnt in c.fetchall():
            trend.append({
                'date': day.strftime('%Y-%m-%d'),
                'sentiment': round(float(avg_s), 1),
                'count': cnt
            })
        return jsonify(trend)
    except Exception as e:
        print(f"Sentiment trend error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sentiment/distribution')
def get_sentiment_distribution():
    """긍정·중립·부정 리뷰 건수 분포 반환"""
    try:
        c = g.cursor
        c.execute("""
            SELECT 
              COUNT(CASE WHEN sentiment_score >= 70 THEN 1 END),
              COUNT(CASE WHEN sentiment_score < 70 AND sentiment_score >= 40 THEN 1 END),
              COUNT(CASE WHEN sentiment_score < 40 THEN 1 END)
            FROM movie_reviews
            WHERE sentiment_score IS NOT NULL
        """)
        pos, neu, neg = c.fetchone()
        return jsonify({'positive': pos, 'neutral': neu, 'negative': neg})
    except Exception as e:
        print(f"Sentiment distribution error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/movies/wordcloud/<int:movie_id>')
def get_movie_wordcloud(movie_id):
    """워드클라우드용 데이터: 한글·영어 모두 추출"""
    c = g.cursor
    try:
        c.execute("""
                SELECT review_text FROM movie_reviews
                WHERE movie_id = :id AND sentiment_score >= 70
                ORDER BY review_date DESC
                FETCH FIRST 100 ROWS ONLY
        """, {"id": movie_id})
        
        # ★ LOB 데이터 처리 수정 ★
        texts = []
        for row in c.fetchall():
            if row[0]:
                # LOB 객체인 경우 .read()로 문자열 변환
                text = row[0].read() if hasattr(row[0], 'read') else str(row[0])
                texts.append(text)
        
        all_text = " ".join(texts)

        # 한글(2자 이상) 또는 영어 단어(2자 이상) 추출
        words = re.findall(r"\b[가-힣A-Za-z]{2,}\b", all_text)

        # 불용어 제거
        stopwords = {
            '영화','정말','진짜','너무','완전','이거','그냥','좀',
            '것','때','수','더','또','잘','많이'
        }
        filtered = [w for w in words if w not in stopwords]

        if not filtered:
            return jsonify([])

        freq = Counter(filtered).most_common(50)
        return jsonify([{'text': w, 'value': cnt} for w, cnt in freq])

    except Exception as e:
        print(f"Wordcloud error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/movies/list')
def get_movies_list():
    """워드클라우드용 리뷰가 있는 영화 ID·제목 목록 반환"""
    try:
        c = g.cursor
        c.execute("""
            SELECT DISTINCT m.movie_id, m.title
            FROM movies m
            JOIN movie_reviews mr ON m.movie_id = mr.movie_id
            WHERE mr.review_text IS NOT NULL
            ORDER BY m.title
        """)
        movies = [{'id': r[0], 'title': r[1]} for r in c.fetchall()]
        print(f"영화 목록 수: {len(movies)}")
        return jsonify(movies)
    except Exception as e:
        print(f"Movies list error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/movies/reviews/<int:movie_id>')
def get_movie_reviews(movie_id):
    """특정 영화의 최근 리뷰 10개 반환"""
    try:
        c = g.cursor
        c.execute("""
            SELECT review_text, sentiment_score, review_date
            FROM movie_reviews
            WHERE movie_id = :id AND review_text IS NOT NULL
            ORDER BY review_date DESC
            FETCH FIRST 10 ROWS ONLY
        """, {"id": movie_id})
        
        reviews = []
        for review_text, score, date in c.fetchall():
            # ★ LOB 데이터 처리 ★
            text = review_text.read() if hasattr(review_text, 'read') else str(review_text)
            reviews.append({
                'text': text[:200] + ('...' if len(text) > 200 else ''),
                'score': round(float(score or 0), 1),
                'date': date.strftime('%Y-%m-%d') if date else ''
            })
        return jsonify(reviews)
    except Exception as e:
        print(f"Movie reviews error: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/movies/info/<int:movie_id>')
def get_movie_info(movie_id):
    """특정 영화의 기본 정보(제목, 평균점수, 리뷰수) 반환"""
    try:
        c = g.cursor
        c.execute("""
            SELECT m.title, mss.avg_score, mss.review_count
            FROM movies m
            LEFT JOIN movie_sentiment_scores mss ON m.movie_id = mss.movie_id
            WHERE m.movie_id = :id
        """, {"id": movie_id})
        
        result = c.fetchone()
        if result:
            title, avg_score, review_count = result
            return jsonify({
                'title': title,
                'avgScore': round(float(avg_score or 0), 1),
                'reviewCount': review_count or 0
            })
        else:
            return jsonify({'error': '영화를 찾을 수 없습니다'}), 404
            
    except Exception as e:
        print(f"Movie info error: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/api/refresh')
def refresh_data():
    """새로운 리뷰 크롤링·분석 트리거(백그라운드)"""
    return jsonify({'message': '데이터 새로고침이 시작되었습니다.'})

# ─── 앱 실행 ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    init_model()
    app.run(debug=True, port=5000)