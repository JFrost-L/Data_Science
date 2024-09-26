import pandas as pd
from db_conn import *
#내부가 java로 되어 있어서 jvm 형식을 맞춰야함!
from konlpy.tag import *
### pip install konlpy

#from mecab import MeCab
### pip install python-mecab-ko

import pandas as pd
import os

import matplotlib.pyplot as plt


class class_document_tfidf():
    def __init__(self):
        self.conn, self.cur = open_db()
        self.news_article_excel_file = 'combined_article.xlsx'
        self.pos_tagger = Kkma()
        #self.pos_tagger = Mecab()

    #엑셀들을 모두 통합해서 하나의 엑셀에 취합해 하나의 엑셀을 생성
    def combine_excel_file(self):
        directory_path = './'
        #모든 엑셀의 경로를 모으기
        excel_files = [file for file in os.listdir(directory_path) if file.endswith('.xlsx')]
        
        #데이터 프레임 생성
        combined_df = pd.DataFrame()
        #각 엑셀별로 엑셀 내용을 리스트로 내용 취합
        for file in excel_files:
            try:
                file_path = os.path.join(directory_path, file)
                df = pd.read_excel(file_path)
                #엑셀 파일의 내용을 읽어서
                df = df[['url', 'title', 'content']]
                combined_df = combined_df.append(df, ignore_index=True)
                #열별로 내용을 삽입해서 업데이트
            except Exception as e:
                print(file_path)
                print(e)
                continue
            
        #취합한 리스트로 엑셀을 생성
        combined_df.to_excel(self.news_article_excel_file, index=False)        
        
    #테이블 생성!해서 데이터 삽입
    def import_news_article(self):
        #이미 테이블 있으면 삭제
        create_sql = """
            drop table if exists news_article;
        
            CREATE TABLE news_article (
              id int auto_increment primary key,
              url varchar(500),
              title varchar(500),
              content TEXT,
              enter_date datetime default now()
            ) ;
        """
    
        self.cur.execute(create_sql)
        self.conn.commit()
    
        #self.news_article_excel_file가 현재 combined_article.xlsx로 되어있어 반영
        file_name = self.news_article_excel_file
        news_article_data = pd.read_excel(file_name)
        
        rows = []
    
        insert_sql = """insert into news_article(url, title, content)
                        values(%s,%s,%s);"""
    
        for _, t in news_article_data.iterrows():
            #예외처리 구문으로 파일 내에 데이터 하나씩 제공
            t = tuple(t)
            try:
                self.cur.execute(insert_sql, t)
            except:
                continue
            #rows.append(tuple(t))
    
        #self.cur.executemany(insert_sql, rows)
        #한 번에 넣기 하지만 노이즈 데이터 때문에
        #위처럼 예외처리 구문으로 일일이 데이터를 넣었음
        self.conn.commit()

        print("table created and data loaded")     
        
    def extract_nouns(self):
        #명사를 뽑아내는 형태소 분석
        
        #extracted_terms 각 도큐멘트에서 뽑은 term들을 1개의 테이블로 생성
        #term_dict corpus 내에 distinct한 term을 dictionary 1개로 지정한 테이블 생성
        create_sql = """
            drop table if exists extracted_terms;
            drop table if exists term_dict;
            
            create table extracted_terms (
                id int auto_increment primary key,
                doc_id int,
                term varchar(30),
                term_region varchar(10),
                seq_no int,
                enter_date datetime default now()    ,
                index(term)
                );
            
            create table term_dict (
                id int auto_increment primary key,
                term varchar(30),
                enter_date datetime default now(),
                index(term)
                );
        """
        
        self.cur.execute(create_sql)
        self.conn.commit()
        
        
        sql = """select * from news_article;"""
        self.cur.execute(sql)
        r = self.cur.fetchone()
        #기존 테이블에서 하나씩 읽어와서
        
        noun_terms = set()
        
        rows = []
        #rows에 추출된 것을 하나씩 넣기
        
        while r:
            print(f"doc_id={r['id']}")
            i = 0
            title_res = self.pos_tagger.nouns(r['title'])
            #kkma의 메서드로 한 document의 제목 내에서 명사들을 다 뽑아내기
            content_res = self.pos_tagger.nouns(r['content'])
            #kkma의 메서드로 한 기사 본문 내에서 명사들을 다 뽑아내기
            
            title_rows = [ (r['id'], t, 'title', i+1) for i, t in enumerate(title_res) ]
            content_rows = [ (r['id'], c, 'content', i+1) for i, c in enumerate(content_res) ]
            #각 rows를 뽑아서
            rows += title_rows
            rows += content_rows
            #기존 rows에 추가
            
            #print(f"title_res={title_res}")
            #print(f"content_res={content_res}")
            
            noun_terms.update(title_res)
            noun_terms.update(content_res)
            #해당 set에 대해서 update() 하면 distinct하게 모음
            r = self.cur.fetchone()
            #다음 document로 이동
            
        if rows:
            insert_sql = """insert into extracted_terms(doc_id, term, term_region, seq_no)
                            values(%s,%s,%s,%s);"""
            self.cur.executemany(insert_sql, rows)
            #rows에 있는 것들을 insert_sql 한 번에 수행
            self.conn.commit()

        #print(noun_terms)
        print(f"\nnumber of terms = {len(noun_terms)}")
        #위는 distinct한 term의 개수를 표현
        
        insert_sql = """insert into term_dict(term) values (%s);"""
        self.cur.executemany(insert_sql, list(noun_terms))
        #noun_terms 자체는 딕셔너리로 되어 있는데 list로 변경해서
        #마찬가지로 한 번에 데이터 삽입
        self.conn.commit()
        
    def gen_idf(self):
        #idf 테이블 생성 이 때 idf와 df 값 추출!
        #이는 각 term에 대해서 value(df와 idf)가 존재
        #term id 별로 나온 df(해당 term이 존재하는 document 빈도수)와
        #idf(전체 중에 존재 document 수 비율) 구하기
        create_sql = """
            drop table if exists idf;
            
            create table idf (
                term_id int primary key,
                df int,
                idf float,
                enter_date datetime default now()                
                );
        """
        
        self.cur.execute(create_sql)
        self.conn.commit()
        
        #idf를 위한 document 개수를 추출
        sql = "select count(*) as doc_count from news_article;"
        self.cur.execute(sql)
        self.num_of_docs = self.cur.fetchone()['doc_count']
        #document 개수 추출
        
        #양 table에서 동일한 term 에 대해서 조인하고 term_dict의 id에 대해서 
        #distinct document id 값을 도출하면 df 리턴
        #거기에 idf는 전체 document를 distinct doc_id의 개수로 나누기
        idf_sql = f""" insert into idf(term_id, df, idf)
                select ntd.id, count(distinct doc_id) as df, log({self.num_of_docs}/count(distinct doc_id)) as idf
                from extracted_terms ent, term_dict ntd
                where ent.term = ntd.term
                group by ntd.id;
            """
        self.cur.execute(idf_sql)        
        self.conn.commit()


    def show_top_df_terms(self):
       #df 대한 내림차순으로 idf 테이블과 term_dict 테이블에서 각각 동일한 id에 대해서 조인해서 출력
        sql = """select * from idf idf, term_dict td
                where idf.term_id = td.id
                order by df desc
                ;"""
        self.cur.execute(sql)
        
        res = [ (r['term'], r['df'], r['idf']) for r in self.cur.fetchall() ]

        print("\nTop DF terms:\n")
        
        for r in res[:30]:
            print(f"{r[0]}: df={r[1]}, idf={r[2]}")
            
        # DF 히스토그램 
        #but corpus가 적어서 의미가 제대로 반영되지 못한 히스토그램이 나옴
        df_list = [ r[1] for r in res]
        plt.figure(figsize=(10, 5))
        plt.hist(df_list, bins=100, alpha=0.7, color='blue') 
        plt.title('Histogram of DF')
        plt.xlabel('Document Frequency')
        plt.ylabel('Number of Terms')
        plt.grid(axis='y', alpha=0.75)
        
        plt.show()        


    def show_top_idf_terms(self):
       #idf에 대한 내림차순으로 idf 테이블과 term_dict 테이블에서 각각 동일한 id에 대해서 조인해서 출력
        sql = """select * from idf idf, term_dict td
                where idf.term_id = td.id
                order by idf desc
                ;"""
        self.cur.execute(sql)
        
        res = [ (r['term'], r['df'], r['idf']) for r in self.cur.fetchall() ]

        print("\nTop IDF terms:\n")
        
        for r in res[:30]:
            print(f"{r[0]}: df={r[1]}, idf={r[2]}")


        # IDF 히스토그램 
        #but corpus가 적어서 의미가 제대로 반영되지 못한 히스토그램이 나옴
        idf_list = [ r[2] for r in res]
        plt.figure(figsize=(10, 5))
        plt.hist(idf_list, bins=30, alpha=0.7, color='red') 
        plt.title('Histogram of IDF')
        plt.xlabel('IDF')
        plt.ylabel('Number of Terms')
        plt.grid(axis='y', alpha=0.75)
        
        plt.show()  


    def gen_tfidf(self):
        #마지막으로 tfidf에 대한 테이블 생성
        #일련번호 id 넣고 document id와 term id에 대해서 tf와 tfidf 값을 입력
        create_sql = """
            drop table if exists tfidf;
            
            create table tfidf (
                id int auto_increment primary key,
                doc_id int,
                term_id int,
                tf float,
                tfidf float,
                enter_date datetime default now()
                );
        """
        
        self.cur.execute(create_sql)
        self.conn.commit()
        
        #extracted_terms는 ent, term_dict은 ntd, idf는 idf로 앨리어싱 후에 시작
        #count(*)로 tf가 나옴
        #count(*) * idf.idf이 tfidf가 나옴
        #ent.doc_id, ntd.id로 그룹핑!
        
        tfidf_sql = """ insert into tfidf(doc_id, term_id, tf, tfidf )  
                        select ent.doc_id, ntd.id, count(*) as tf, count(*) * idf.idf as tfidf
                        from extracted_terms ent, term_dict ntd, idf idf
                        where ent.term = ntd.term and ntd.id = idf.term_id
                        group by ent.doc_id, ntd.id;
                    """

        #테이블을 보면 corpus에 의해서 tf가 같아도 tf*idf가 다르게 나올 수 있음
        self.cur.execute(tfidf_sql)        
        self.conn.commit()

    #주어진 document의 keyword 구하기
    def get_keywords_of_document(self, doc):
        sql = f""" 
            select *
            from tfidf tfidf, term_dict td
            where tfidf.doc_id = {doc}
            and tfidf.term_id = td.id
            order by tfidf.tfidf desc
            limit 10;
        """
        #tfidf 테이블과 term_dict 테이블로부터 tfidf의 document id와 주어진 document id와 동일한
        #document를 뽑아서 그 내부에서 중요 단어 뽑아내기!
        #tfidf순으로 내림차순하면 중요도로 정렬해서 document의 keyword 리턴
        self.cur.execute(sql)
        
        r = self.cur.fetchone()
        while r:
            print(f"{r['term']}: {r['tfidf']}")
            r = self.cur.fetchone()
        
    #두개의 주어진 document에 대해서 코사인 유사도 계산
    def doc_similarity(self, doc1, doc2):
        def cosine_similarity(vec1, vec2):
            #두 벡터에 대한 코사인 유사도 식
            dict1 = dict(vec1)
            dict2 = dict(vec2)
            
            common_terms = set(dict1.keys()) & set(dict2.keys())
            #두 벡터 중 하나라도 0이면 재끼기(어짜피 곱하면 0이니까)
            dot_product = sum([dict1[term] * dict2[term] for term in common_terms])
            #벡터 내적
            vec1_magnitude = sum([val**2 for val in dict1.values()])**0.5
            vec2_magnitude = sum([val**2 for val in dict2.values()])**0.5
            #벡터의 크기 구하기
            if vec1_magnitude == 0 or vec2_magnitude == 0:
                return 0
            else:
                return dot_product / (vec1_magnitude * vec2_magnitude)        
        
        sql1 = f"""select term_id, tfidf from tfidf where doc_id = {doc1};"""
        self.cur.execute(sql1)
        doc1_vector = [(t['term_id'], t['tfidf']) for t in self.cur.fetchall()]
        #tfidf 테이블에서 term_id와 tfidf 열을 갖고오는데 주어진 document에 해당하는 것만 긁어오기
        sql1 = f"""select term_id, tfidf from tfidf where doc_id = {doc2};"""
        self.cur.execute(sql1)
        doc2_vector = [(t['term_id'], t['tfidf']) for t in self.cur.fetchall()]
        #tfidf 테이블에서 term_id와 tfidf 열을 갖고오는데 주어진 document에 해당하는 것만 긁어오기
    
        return cosine_similarity(doc1_vector, doc2_vector)
        
        
    #주어진 document와 유사도를 기준으로 정렬
    def sort_similar_docs(self, doc):
        sim_vector = []

        for i in range(1,386):
            if i == doc:#본인의 document는 제외하고 
                continue
            sim = cdb.doc_similarity(doc, i)
            #본인의 document와 그 이외의 document들에 대해서 document간의 유사도 측정
            sim_vector.append((i, sim))
            #측정한 유사도와 document id를 sim_vector에 추가
        
        sorted_sim_vector = sorted(sim_vector, key=lambda x: x[1], reverse=True)
        #코사인 유사도에 의해서 나온 벡터들 중 유사도가 높은 순서로 정렬해서 가져오기
        print(sorted_sim_vector)
    
        
if __name__ == '__main__':
    cdb = class_document_tfidf()
    #cdb.combine_excel_file()
    # cdb.import_news_article()
    cdb.extract_nouns()
    cdb.gen_idf()
    cdb.show_top_df_terms()
    cdb.show_top_idf_terms()
    cdb.gen_tfidf()
    cdb.get_keywords_of_document(100)
    cdb.sort_similar_docs(100)

