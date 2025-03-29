from database import conn, cx_Oracle
from newmovie import crawl_all_movie_titles

def store_movies_in_db(titles, conn):
    """
    크롤링한 영화 제목 리스트를 Oracle DB의 movies 테이블에 저장합니다.
    테이블 구조는 아래와 같이 가정합니다:
    
    CREATE TABLE movies (
        movie_id NUMBER GENERATED ALWAYS AS IDENTITY,
        title VARCHAR2(200),
        insert_date DATE DEFAULT SYSDATE,
        PRIMARY KEY(movie_id)
    );
    """
    cursor = conn.cursor()
    insert_sql = "INSERT INTO movies (title) VALUES (:title)"
    
    for title in titles:
        try:
            cursor.execute(insert_sql, {"title": title})
            conn.commit()
            print(f"영화 '{title}' 저장 성공")
        except cx_Oracle.DatabaseError as e:
            print(f"영화 '{title}' 저장 실패:", e)
    cursor.close()

def remove_duplicate_movies(conn):
    """
    movies 테이블에서 중복된 영화 제목을 제거합니다.
    각 영화 제목별로 가장 작은 movie_id를 가진 레코드를 남기고 나머지를 삭제합니다.
    """
    cursor = conn.cursor()
    delete_sql = """
    DELETE FROM movies
    WHERE movie_id NOT IN (
      SELECT MIN(movie_id)
      FROM movies
      GROUP BY title
    )
    """
    try:
        cursor.execute(delete_sql)
        conn.commit()
        print("중복 영화 제거 완료")
    except cx_Oracle.DatabaseError as e:
        print("중복 영화 제거 실패:", e)
    cursor.close()

def main():
    # newmovie.py의 크롤링 함수를 호출하여 영화 제목 리스트를 얻습니다.
    titles = crawl_all_movie_titles()
    print("총 크롤링한 영화 제목 개수:", len(titles))
    
    # Oracle DB 연결은 미리 설정된 conn을 사용합니다.
    # 크롤링한 영화 제목들을 DB에 저장
    store_movies_in_db(titles, conn)
    
    # 중복된 영화 제목 제거
    remove_duplicate_movies(conn)
    
    conn.close()
    print("모든 영화 정보가 DB에 저장되었으며, 중복 제거 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()
