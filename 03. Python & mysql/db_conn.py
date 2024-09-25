import pymysql


from pymysql.constants.CLIENT import MULTI_STATEMENTS

# db_conn.py에서
def open_db(dbname='DS2023'):
    conn = pymysql.connect(
        host='localhost',
        user='datascience',  # 생성한 사용자와 일치해야 합니다.
        passwd='0000',       # 설정한 비밀번호와 일치해야 합니다.
        db=dbname,
        client_flag=MULTI_STATEMENTS,
        charset='utf8mb4'
    )
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    return conn, cursor


def close_db(conn, cur):
    cur.close()
    conn.close()
    
if __name__ == '__main__':
    conn, cur = open_db()
    close_db(conn, cur)