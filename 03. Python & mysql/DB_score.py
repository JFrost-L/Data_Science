# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:15:26 2023

@author: L
"""
from db_conn import *
import pandas as pd

file_name = "score.xlsx"
df = pd.read_excel(file_name)

use_db_sql="""use ds2023;"""
drop_table_sql = """drop table if exists score;"""
create_table_sql = """create table score (
    sno int primary key,
    attendance int,
    homework int,
    discussion int,
    midterm int,
    final int,
    score float,
    grade varchar(10)
);"""

insert_sql = """insert into score (sno, attendance, homework,
                                    discussion, midterm, final, score, grade)
                                values(%s, %s, %s, %s, %s, %s, %s, %s)"""

conn, cur = open_db()
conn.commit()
cur.execute(use_db_sql)
conn.commit()
cur.execute(drop_table_sql)
conn.commit()
cur.execute(create_table_sql)
conn.commit()

for index, r in df.iterrows():
    t = (r['sno'], r['attendance'], r['homework'],
         r['discussion'], r['midterm'], r['final'], r['score'], r['grade'] )
    try:
        cur.execute(insert_sql, t)
    except Exception as e:
        print(t)
        print(e)
        break

conn.commit()
close_db(conn, cur)
