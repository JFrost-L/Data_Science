from db_conn import *
import pandas as pd

file_name = 'score.xlsx'
df = pd.read_excel(file_name)

insert_sql = """insert into score (sno, attendance, homework,
                    discussion, midterm, final, score, grade)
                values(%s, %s,%s, %s,%s, %s,%s, %s);"""
                
conn, cur = open_db('ds2023')

truncate_sql = """truncate table score;"""
cur.execute(truncate_sql)
conn.commit()

for index, r in df.iterrows():
    t = (r['sno'],r['attendance'],r['homework'], r['discussion'],
      r['midterm'],r['final'],r['score'],r['grade'])
    try:
        cur.execute(insert_sql, t)
    except Exception as e:
        print(t)
        print(e)
        break
        
    
conn.commit()

close_db(conn, cur)