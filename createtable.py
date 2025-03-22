# create_tables.py
import cx_Oracle
from database import conn  # database.py에 DB 연결 함수가 있다고 가정

def create_tables():
    cursor = conn.cursor()

    create_movies_sql = """
        CREATE TABLE movies (
            movie_id NUMBER GENERATED ALWAYS AS IDENTITY,
            title VARCHAR2(200),
            insert_date DATE DEFAULT SYSDATE,
            PRIMARY KEY(movie_id)
        )
    """
    
    create_reviews_sql = """
        CREATE TABLE movie_reviews (
            review_id NUMBER GENERATED ALWAYS AS IDENTITY,
            movie_id NUMBER NOT NULL,
            review_text CLOB,
            review_date DATE,
            insert_date DATE DEFAULT SYSDATE,
            PRIMARY KEY(review_id),
            FOREIGN KEY(movie_id) REFERENCES movies(movie_id)
        )
    """
    
    try:
        cursor.execute(create_movies_sql)
        print("Movies 테이블 생성 성공")
    except cx_Oracle.DatabaseError as e:
        print("Movies 테이블 생성 실패:", e)
    
    try:
        cursor.execute(create_reviews_sql)
        print("movie_reviews 테이블 생성 성공")
    except cx_Oracle.DatabaseError as e:
        print("movie_reviews 테이블 생성 실패:", e)
    
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    create_tables()
