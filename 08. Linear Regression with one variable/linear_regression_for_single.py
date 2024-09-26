from db_conn import *
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import statsmodels.api as sm
#통계 분석 모듈


class class_linear_regression_for_single_independent_variable():
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
    
        self.X = [ t['midterm'] for t in data ]
        self.X = np.array(self.X)
    
    
        self.y = [ t['score'] for t in data]
        self.y = np.array(self.y)    
        
        #print("X=",self.X)
        #print("y=", self.y)
    

    def plot_data(self):
        plt.scatter(self.X, self.y)  # X와 y의 매칭된 값을 2차원 평면 상에 점으로 찍음
        plt.xlabel('midterm')  # x축 레이블
        plt.ylabel('score')  # y축 레이블
        plt.title('midterm vs score')  # 그래프 제목
        plt.grid(True)  # 그리드 라인 표시
        plt.show()  # 그래프 표시        
        

    def least_square(self):
        X = sm.add_constant(self.X)
        #상수 항을 추가!
        model = sm.OLS(self.y, X)
        #OLS는 ordinary least square 함수!로 y와 x로 모델 생성
        results = model.fit()#학습
        print(results.summary())
        """
        중간고사로 분석할 때
        통계값이 유의미한지 결과로 분석해보자
        
        R-squared:0.773으로 1에 가까울수록 좋음
        
        F-statistic:306.8은 귀무가설을 기각할 경계를 의미하며
                  넘어가면 독립 변수와 종속 변수간의 관련이 없다는 것을 의미
                  
        Prob (F-statistic):9.65e-31은 귀무가설을 기각할 확률
        
        const:24.3479는 c값에 대한 것이고 P>|t|이 0으로 무조건 믿어도 됨.
        x1:1.6848는 m값에 대한 것이고 P>|t|이 0으로 무조건 믿어도 됨.
        """
        
        print(f"params:\n{results.params}")
        
        self.c, self.m = results.params
        
        print(f"\nm={self.m}, final c={self.c} from least square")
        
        
    def plot_graph_after_regression(self, interactive_flag=False):
        #scatter에 regression 선을 추가!
        if interactive_flag:
            plt.ion()  # 대화식 모드

        plt.scatter(self.X, self.y, label='Data Points')
        

        y_pred = self.m * self.X + self.c
        plt.plot(self.X, y_pred, color='red', label='Regression Line')
        
        plt.xlabel('midterm')
        plt.ylabel('score')
        plt.title('Regression Line on Scatter Plot')
        plt.legend()  
        plt.grid(True)  
        plt.ylim(-5, max(self.y) + 5)
        
        if interactive_flag:
            plt.draw()  
            plt.pause(0.1)  
            plt.clf() 
        else:
            plt.show() 

    def gradient_descent(self):
        epochs = 100000 #반복횟수
        min_grad = 0.00001 #어느정도 threshold 허용값
        learning_rate = 0.001#이것이 너무 크면 pingpong 너무 작으면 너무 오래걸림
        
        #m과 c의 초기값 설정
        self.m = 0.0
        self.c = 0.0
        
        self.plot_graph_after_regression(interactive_flag=True) 
        
        n = len(self.y)
        
        for epoch in range(epochs):
            #epochs를 반복
            
            m_partial = 0.0
            c_partial = 0.0
            
            for i in range(n):
                #각 편미분값을 구해서 시그마를 루프로 부분 m과 c를 구해서 더하기
                y_pred = self.m * self.X[i] + self.c
                m_partial += (y_pred-self.y[i])*self.X[i]
                c_partial += (y_pred-self.y[i])
            
            #루프로 시그마 처리 후 나머지 연산 처리
            m_partial *= 2/n
            c_partial *= 2/n
            
            #delta_m과 c를 구하자
            delta_m = -learning_rate * m_partial
            delta_c = -learning_rate * c_partial
            
            #delta값들을 초기 기준값과 비교
            if ( abs(delta_m) < min_grad and abs(delta_c) < min_grad ):
                break
            
            #경사하강 진행!
            self.m += delta_m
            self.c += delta_c
            
            if ( epoch % 1000 == 0 ):
                #1000번마다 중간 과정 출력
                print("epoch %d: delta_m=%f, delta_c=%f, m=%f, c=%f" %(epoch, delta_m, delta_c, self.m, self.c) )
                self.plot_graph_after_regression(interactive_flag=True) 
                
        print(f"\nm={self.m}, final c={self.c} from gradient descent")        
        self.plot_graph_after_regression()
        #계속된 직선 보정
        


        


if __name__ == "__main__":
    lr = class_linear_regression_for_single_independent_variable(import_data_flag=False )
    lr.load_data_for_linear_regression()
    lr.plot_data()
    lr.least_square()
    lr.plot_graph_after_regression()
    
    lr.gradient_descent()
    #ls와 큰차이 없이 느리지 않게 결과 도출 가능
    lr.plot_graph_after_regression()