from db_conn import *
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import statsmodels.api as sm

import time



class class_linear_regression_for_multiple_independent_variable():
    def __init__(self, import_data_flag=True):
        self.conn, self.cur = open_db()
        if import_data_flag:
            self.import_dbscore_data()
        

    def import_dbscore_data(self):
        drop_sql =""" drop table if exists db_score;"""
        self.cur.execute(drop_sql)
        self.conn.commit()
    
        create_sql = """
            CREATE TABLE `db_score` (
              `sno` int NOT NULL,
              `attendance` float DEFAULT NULL,
              `homework` float DEFAULT NULL,
              `discussion` int DEFAULT NULL,
              `midterm` float DEFAULT NULL,
              `final` float DEFAULT NULL,
              `score` float DEFAULT NULL,
              `grade` char(1) DEFAULT NULL,
              enter_date datetime default now(),
              PRIMARY KEY (`sno`)
            ) ;
        """
    
        self.cur.execute(create_sql)
        self.conn.commit()
    
        file_name = 'db_score.xlsx'
        dbscore_data = pd.read_excel(file_name)
    
        rows = []
    
        insert_sql = """insert into db_score(sno, attendance, homework, discussion, midterm, final, score, grade)
                        values(%s,%s,%s,%s,%s,%s,%s,%s);"""
    
        for t in dbscore_data.values:
            rows.append(tuple(t))
    
        self.cur.executemany(insert_sql, rows)
        self.conn.commit()


        print("table created and data loaded")

    def load_data_for_linear_regression(self):
        sql = "select * from db_score;"
        self.cur.execute(sql)
    
        data = self.cur.fetchall()
        
        #print("data=", data)
        
        #중간고사와 기말고사를 독립변수로 설정
        self.X = [ (t['midterm'], t['final']) for t in data ]
        self.X = np.array(self.X)

    
        self.y = [ t['score'] for t in data]
        self.y = np.array(self.y)  
        self.y_label = 'score'
        
        #print("X=",self.X)
        #print("y=", self.y)


    def least_square(self):
        X = sm.add_constant(self.X)
        model = sm.OLS(self.y, X)
        results = model.fit()
        print(results.summary())
        
        
        print(f"params:\n{results.params}")
        """
        중간고사와 기말고사로 분석할 때
        통계값이 유의미한지 결과로 분석해보자
        
        R-squared: 0.964으로 1에 가까울수록 좋음
        
        F-statistic:1194은 귀무가설을 기각할 경계를 의미하며
                  넘어가면 독립 변수와 종속 변수간의 관련이 없다는 것을 의미
                  그러나 사실 그럴 가능성이 거의 없음
        Prob (F-statistic):5.30e-65은 귀무가설을 기각할 확률
        
        const:22.8191는 c값에 대한 값이고 그 때의 
        t분포는 23.786으로 그 때의 확률은 P>|t|이 0으로 무조건 믿어도 됨.
        이 때 [0.025 0.975] 값들은 해당 분포에서의 t값을 의미
        
        x1:1.1370는 m1인 중간고사에 대한 값이고 그 때의
        t분포는 24.709으로 그 때의 확률은 P>|t|이 0으로 무조건 믿어도 됨.
        이 때 [0.025 0.975] 값들은 해당 분포에서의 t값을 의미
        
        x2:1.0203는 m2인 기말고사에 대한 값이고
        그 때의 t분포는 21.741으로 그 때의 확률은 P>|t|이 0으로 무조건 믿어도 됨.
        이 때 [0.025 0.975] 값들은 해당 분포에서의 t값을 의미
        """
        self.c, self.m = results.params[0], results.params[1:]
        
        print(f"\nm={self.m}, final c={self.c} from least square")
        

    def gradient_descent(self):
      
        self.start_time = time.time()
        
        epochs = 100000
        min_grad = 0.000001
        learning_rate_m = 0.001
        learning_rate_c = 0.001
        
        num_params = self.X.shape[1]
        
        m = [0.0]*num_params
        c = 0.0
        print(f"m : {m}")
        n = len(self.y)
    
        #동일한 데이터에 대해서 epoch을 반복해서 학습할수록
        #fitting이 잘 된다! 너무 많이 하면 과적합 문제 발생
        for epoch in range(epochs):
    
            c_partial = 0.0        
            m_partial = [0.0]*num_params
            #epoch마다 초기화!
            
            #현재 epoch 상태에서 
            for i in range(n):
                y_pred = c
                
                #시그마 처리
                for j in range(num_params):
                    y_pred += m[j] * self.X[i][j]
                
                #오차값 계산
                c_partial += (y_pred-self.y[i])
                for j in range(num_params):
                    m_partial[j] += (y_pred-self.y[i])*self.X[i][j]
            
            c_partial *= 2/n
            for j in range(num_params):
                m_partial[j] *= 2/n
            
            #각 편미분에 대해서 조정할 값들인 delta값 설정
            delta_c = -learning_rate_c * c_partial
            delta_m = [0.0]*num_params
            
            for j in range(num_params):
                delta_m[j] = -learning_rate_m * m_partial[j]
            
            #기준에 대해서 stop 유도
            break_condition = True
            if abs(delta_c) > min_grad:
                break_condition = False
            for j in range(num_params):
                if abs(delta_m[j]) > min_grad:
                    break_condition = False
            
            if break_condition:
                break
            
            #delta값에 따라서 데이터 조정
            c = c + delta_c
            for j in range(num_params):
                m[j] = m[j] + delta_m[j]
            
            #특정 에폭마다 print()
            if ( epoch % 1000 == 0 ):
                print(f"epoch {epoch}: delta_c={delta_c}, delta_m={delta_m}, c={c}, m={m}")
            
        #학습이 끝나면 값 보존
        print(f"c={c}, m={m} from gradient descent")
        self.c, self.m = c, m
        
        self.end_time = time.time()
        print('response time=%f seconds' %(self.end_time - self.start_time) )
 

        

      



if __name__ == "__main__":
    lr = class_linear_regression_for_multiple_independent_variable(import_data_flag=False)
    lr.load_data_for_linear_regression()
    lr.least_square()
    lr.gradient_descent()
    #ls와 gradient_descent의 결과의 차이가 그리 크지 않아서 좋다.
    #범용성은 경사하강법이 좋긴한데 속도가 조금 느려서 벡터화 연산을 적용해보자

    
    