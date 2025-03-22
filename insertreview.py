from database import conn, cx_Oracle
from crawling import test_crawling, encode_movie_name

def get_all_movies(conn):
    """
    Oracle DB의 movies 테이블에서 모든 영화 정보를 조회합니다.
    반환값은 각 영화의 (movie_id, title) 튜플 목록입니다.
    """
    cursor = conn.cursor()
    query = "SELECT movie_id, title FROM movies"
    cursor.execute(query)
    movies = cursor.fetchall()  # 예: [(1, '영화제목1'), (2, '영화제목2'), ...]
    cursor.close()
    return movies

def store_movie_reviews_in_db(reviews, movie_id, conn):
    """
    크롤링한 리뷰 리스트를 Oracle DB의 movie_reviews 테이블에 저장합니다.
    
    각 리뷰는 딕셔너리 형태이며, 'review_text'와 'review_time' (작성일자, 예: '2015.03.12. 23:04') 키를 포함한다고 가정합니다.
    
    테이블 구조:
    CREATE TABLE movie_reviews (
        review_id NUMBER GENERATED ALWAYS AS IDENTITY,
        movie_id NUMBER NOT NULL,
        review_text CLOB,
        review_date DATE,
        insert_date DATE DEFAULT SYSDATE,
        PRIMARY KEY(review_id),
        FOREIGN KEY(movie_id) REFERENCES movies(movie_id)
    );
    """
    cursor = conn.cursor()
    insert_sql = """
        INSERT INTO movie_reviews (movie_id, review_text, review_date)
        VALUES (:movie_id, :review_text, TO_DATE(:review_date, 'YYYY.MM.DD. HH24:MI'))
    """
    for review in reviews:
        try:
            cursor.execute(insert_sql, {
                "movie_id": movie_id,
                "review_text": review["review_text"],
                "review_date": review["review_time"]
            })
            conn.commit()
            print(f"영화 ID {movie_id} 리뷰 저장 성공: {review['review_text'][:30]}... / 작성일자: {review['review_time']}")
        except cx_Oracle.DatabaseError as e:
            print(f"영화 ID {movie_id} 리뷰 저장 실패:", e)
    cursor.close()

def main():
    # DB에 저장된 모든 영화 정보를 가져옵니다.
    movies = get_all_movies(conn)
    print(f"총 영화 개수: {len(movies)}")

    failed_movies = []  # 오류가 발생한 영화의 movie_id를 저장할 리스트

    
    for movie_id, title in movies:
        print(f"\n[영화 처리] movie_id: {movie_id}, title: {title}")
        # 영화 제목에 "관람평"을 추가한 후 URL 인코딩
        encoded = encode_movie_name(title + "관람평")
        
        # 크롤링 시도 - 실패할 경우 예외를 잡아 해당 영화(movie_id)를 건너뛰도록 함
        try:
            reviews = test_crawling(encoded)
        except Exception as e:
            print(f"영화 '{title}' (movie_id: {movie_id}) 크롤링 실패: {e}")
            failed_movies.append(movie_id)  # 실패한 영화의 movie_id 저장
            continue  # 오류 발생 시 해당 영화 건너뛰기
        
        if reviews is None or len(reviews) == 0:
            print(f"영화 '{title}' (movie_id: {movie_id})에 대해 크롤링된 리뷰가 없습니다.")
            failed_movies.append(movie_id)  # 리뷰가 없는 영화도 실패로 간주하여 저장
            continue

        print(f"크롤링한 리뷰 개수: {len(reviews)}")
        # 리뷰 저장
        store_movie_reviews_in_db(reviews, movie_id, conn)
    
    conn.close()

    if failed_movies:
        print("\n크롤링에 실패한 영화 IDs:")
        for movie_id in failed_movies:
            print(f"movie_id: {movie_id}")

    print("모든 영화 리뷰가 DB에 저장되었습니다.")

if __name__ == "__main__":
    main()
