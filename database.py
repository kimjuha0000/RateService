import cx_Oracle


username = "SYSTEM"
password = "kk1801"
dsn = "127.0.0.1:1521/XE"

try:
    conn = cx_Oracle.connect(username, password, dsn)
    print("Oracle DB 연결 성공")

    
except cx_Oracle.DatabaseError as e:
    print("연결 실패:", e)
