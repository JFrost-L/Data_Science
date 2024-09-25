# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:40:29 2023

@author: L
"""

from db_conn import *
import pandas as pd
import re
from pymysql.constants.CLIENT import MULTI_STATEMENTS

def open_db(dbname='DS2023'):
    conn = pymysql.connect(host='localhost',
                           user='root',
                           passwd='0000',
                           db=dbname,
                           client_flag=MULTI_STATEMENTS,
                           charset='utf8mb4')

    cursor = conn.cursor(pymysql.cursors.DictCursor)

    return conn, cursor

def close_db(conn, cur):
    cur.close()
    conn.close()
    
if __name__ == '__main__':
    file_name = "top_movies.csv"
    df = pd.read_csv(file_name).fillna(value = 0)
    use_db_sql="""use ds2023;"""
    drop_table_sql = """drop table if exists top_movies;"""
    
    create_table_sql = """create table top_movies (
                    	id int primary key,
                        movie_name varchar(200),
                        release_year int,
                        watch_time int,
                        movie_rating float,
                        metascore int,
                        gross float,
                        votes int,
                        description varchar(10000),
                        enter_date datetime default now());"""
    insert_sql = """insert into top_movies (id,	movie_name, release_year,
        	watch_time, movie_rating, metascore, gross, votes, description)
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    conn, cur = open_db()
    conn.commit()
    cur.execute(use_db_sql)
    conn.commit()
    cur.execute(drop_table_sql)
    conn.commit()
    cur.execute(create_table_sql)
    conn.commit()
    
    for index, r in df.iterrows():
        r['Votes'] = int(str(r['Votes'].replace(",", "")))
        if len(r['Year of Release'])!=4:
            r['Year of Release']=r['Year of Release'][-4:-1]
        if type(r['Gross']) != float:
            a=[]
            for word in str(r['Gross']):
                if word.isdigit() or word=='.':
                    a.append(word)
            r['Gross']=float(''.join(a))
                    
        t = (r['id'], r['Movie Name'], r['Year of Release'], r['Watch Time'],
             r['Movie Rating'], r['Metascore of movie'],
             r['Gross'], r['Votes'], r['Description'])
        try:
            cur.execute(insert_sql, t)
        except Exception as e:
            print(t)
            print(e)
            break
    select_sql = "select* from top_movies;"
    cur.execute(select_sql)
    conn.commit()
    
    rows = cur.fetchall()
    for row in rows:
        print(row)
        print()
    
    conn.commit()
    close_db(conn, cur)
