from db_conn import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import cross_val_score, KFold, cross_validate

#붓꽃 분류 클래스
class class_iris_classification():
    def __init__(self, import_data_flag=True):
        #this역할을 하는 self
        self.conn, self.cur = open_db()
        if import_data_flag:#data를 DB로 생성할 때 사용 DB가 있는지 없는지 여부!
        #true면 생성해달라고 요청
            self.import_iris_data()
        

    def import_iris_data(self):
        #새 테이블부터 시작하겠다는 의미
        drop_sql ="""drop table if exists iris;"""
        self.cur.execute(drop_sql)
        self.conn.commit()
    
        #table 생성 primary key는 자동 지정
        create_sql = """
            create table iris (
                id int auto_increment primary key,
                sepal_length float,
                sepal_width float,
                petal_length float,
                petal_width float,
                species varchar(10), 
                enter_date datetime default now(),
                update_date datetime on update now()
                ); 
        """
    #enter_date는 튜플이 만들어진 날짜를 찍어주기 위함
    #update_date는 어떤 attribute가 update된 날짜를 찍어주기 위함
        self.cur.execute(create_sql)
        self.conn.commit()#db에 실행 보장
    
        #pandas로 파일 읽어서 dateFrame 생성
        file_name = 'iris.csv'
        iris_data = pd.read_csv(file_name)
    
        rows = []
        #삽입
        insert_sql = """insert into iris(sepal_length, sepal_width, petal_length, petal_width, species)
                        values(%s,%s,%s,%s,%s);"""
        #tuple 하나씩 append
        for t in iris_data.values:
            print(t)
            rows.append(tuple(t))
    
        self.cur.executemany(insert_sql, rows)
        self.conn.commit()
        print("data import completed")


    def load_data_for_binary_classification(self, species):
        sql = "select * from iris;"
        self.cur.execute(sql)
    
        data = self.cur.fetchall()
        
        #print("data=", data)
    
        #self.X = [ (t['sepal_length'], t['sepal_width'], t['petal_length'], t['petal_width'] ) for t in data ]
        #self.X = [ (t['sepal_length'], t['sepal_width'] ) for t in data ]
        self.X = [ (t['petal_length'], t['petal_width'] ) for t in data ]
        #list comprehension으로 x 데이터 뽑기

        self.X = np.array(self.X)
        #벡터화 연산을 위해 numpy 배열에 저장
    
    
        self.y = [ 1 if (t['species'] == species) else 0 for t in data]
        #list comprehension으로 y를 뽑고 여기에 label 설정
        self.y = np.array(self.y)   
        #벡터화 연산을 위해 numpy 배열에 저장 
        
        #print(f"X={self.X}")
        #print(f"y={self.y}")
    
    
    
    def data_split_train_test(self):
    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=42, test_size=0.3)
        #train data와 test data를 나누고 이에 맞게 x, y도 나눔
        #random state를 고정해서 주는 이유는 그냥 재현을 위함
        #비율은 7:3으로 나누기
        #그 결과 4개의 array를 반환
        print("X_train=", self.X_train)
        print("y_train=", self.y_train)
        print("X_test=", self.X_test)
        print("y_test=", self.y_test)
        
        
        
    #결정트리로 학습하기!
    def train_and_test_dtree_model(self):
        dtree = tree.DecisionTreeClassifier()#결정트리 객체
        dtree_model = dtree.fit(self.X_train, self.y_train)#결정트리로 학습해서 모델 생성
        self.y_predict = dtree_model.predict(self.X_test)#학습된 모델을 이용해서 예측!
        #y_predict가 test 결과로 예측한 것
        
        print(f"self.y_predict[10]={self.y_predict[:10]}")#학습 결과
        print(f"self.y_test[10]={self.y_test[:10]}")#원래 정답 비교하기!


    #성능 평가 메서드
    def classification_performance_eval_binary(self, y_test, y_predict):
        tp, tn, fp, fn = 0,0,0,0
        #tp : 예측한 것을 예측했는데 맞음, tn : 예측하지 않은 것을 예측했는데 맞음
        #fp : 예측한 것을 예측했는데 틀림, fn : 예측하지 않은 것을 예측했는데 틀림
        
        #성능 평가 수치 지정
        for y, yp in zip(y_test, y_predict):#zip 함수로 순회가능한 객체들의 쌍을 튜플로 제공
        #y는 예측한 것, #yp는 정답
            if y == 1 and yp == 1:
                tp += 1
            elif y == 1 and yp == 0:
                fn += 1
            elif y == 0 and yp == 1:
                fp += 1
            else:
                tn += 1
                
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        precision = (tp)/(tp+fp)
        recall = (tp)/(tp+fn)
        f1_score = 2*precision*recall / (precision+recall)
        
        print("accuracy=%f" %accuracy)
        print("precision=%f" %precision)
        print("recall=%f" %recall)
        print("f1 score=%f" %f1_score)


    #K-fold 교차 검증
    #K-fold는 미리 데이터를 나누고 학습을 하고 하는 게 아니라 x, y데이터를 준비하는 로드까지는 동일하고
    #바로 K-fold에서 데이터를 나누고 학습하는 것을 K번 학습을 진행하고 모든 것들을 평균해서 측정
    def binary_dtree_KFold_performance(self):
        accuracy = []
        precision = []
        recall = []
        f1_score = []
    
        #kfold = KFold(n_splits=5, random_state=42, shuffle=True)
        #데이터를 먼저 random_state를 이용해서 섞고 K개로 쪼개는 객체 생성
        #우선 이 객체는 multiClassification에서 사용되고 binaryClassification은 cross_validate 메서드만으로도 충분
        
        dtree = tree.DecisionTreeClassifier()
        
        cv_results = cross_validate(dtree, self.X, self.y, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'])
        #cross_validate라는 메서드를 통해서 k번의 교차검증을 수행해서 이 메서드로는 KFold 객체도 필요 없음
        #딕셔너리를 반환
        
        #cv_results = cross_val_score(model, X, y, cv=5, scoring='accuracy’)
        #cross_val_score이 메서드는 scoring을 오직 accuracy같이 한 개의 성능 지표만 얻을 수 있음
        #사실 cross_validate가 있는데 굳이 사용할 필요는 없음.
        
        print(cv_results)
        
        for metric, scores in cv_results.items():
            #cv_results라는 딕셔너리에서 각 아이템을 순회하면서 이용
            if metric.startswith('test_'):
                print(f'\n{metric[5:]}: {scores.mean():.2f}')    


       
        
#이진 분류하는데 내가 test와 train을 나누고 결정트리 이용하고 성능 평가하는 함수
def binary_dtree_train_test_performance():
    clf = class_iris_classification(import_data_flag=True)#clf는 classification을 의미
    clf.load_data_for_binary_classification(species='versicolor')
    clf.data_split_train_test()#train data와 test data 분류
    clf.train_and_test_dtree_model()#모델 생성 및 학습 진행
    clf.classification_performance_eval_binary(clf.y_test, clf.y_predict)#성능 평가
    
#K-fold 교차검증을 이용해서 data를 나눠서 결정트리로 모델을 학습하고 그에 맞는 성능 평가하는 함수
#K-Fold 교차검증은 상대적으로 안정적인 학습 결과를 제공함
#이는 성능 평가함수들로 알 수 있고 여러 개의 결과를 평균을 내기 때문에 분산이 줄어들어 상대적으로 안정적인 결과를 도출함
def binary_dtree_KFold_performance():
    clf = class_iris_classification(import_data_flag=False)
    clf.load_data_for_binary_classification(species='virginica')
    clf.binary_dtree_KFold_performance()



if __name__ == "__main__":
    #binary_dtree_train_test_performance()
    binary_dtree_KFold_performance()
