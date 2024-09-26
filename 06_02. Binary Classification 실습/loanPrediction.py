from db_conn import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

class class_loan_classification():
    def __init__(self, import_data_flag=True):
        self.conn, self.cur = open_db()
        if import_data_flag:
        #data를 DB로 생성할 때 사용 DB가 있는지 없는지 여부!
        #true면 생성해달라고 요청
            self.import_loan_data()
        

    def import_loan_data(self):
        #새 테이블부터 시작하겠다는 의미
        drop_sql ="""drop table if exists loan;"""
        self.cur.execute(drop_sql)
        self.conn.commit()
    
        #table 생성 primary key는 자동 지정
        create_sql = """
            create table loan (
                id int auto_increment primary key,
                gender int,
                married int,
                dependents int,
                education int,
                self_Employed int,
                applicantIncome int, 
                coapplicantIncome int, 
                loanAmount int, 
                loan_Amount_Term int, 
                credit_History varchar(10), 
                property_Area int, 
                loan_Status char(1),
                enter_date datetime default now(),
                update_date datetime on update now()
                ); 
        """
        #enter_date는 튜플이 만들어진 날짜를 찍어주기 위함
        #update_date는 어떤 attribute가 update된 날짜를 찍어주기 위함
        self.cur.execute(create_sql)
        self.conn.commit()
    
        #pandas로 파일 읽어서 dateFrame 생성
        file_name = 'loan_train.csv'
        loan_data = pd.read_csv(file_name).fillna(value = 'Nan')
    
        rows = []
        #삽입
        insert_sql = """insert into loan(gender, married, dependents, education, self_Employed, applicantIncome,
                        coapplicantIncome, loanAmount, loan_Amount_Term, credit_History, property_Area, loan_Status)
                        values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);"""
        
        count=0
        for index, r in loan_data.iterrows():
            if r['Gender'] == 'Nan' or r['Married'] == 'Nan' or r['Dependents'] == 'Nan'\
            or r['Education'] == 'Nan' or r['Self_Employed']=='Nan' or r['ApplicantIncome']=='Nan'\
            or r['CoapplicantIncome']=='Nan' or r['LoanAmount']=='Nan' or r['Loan_Amount_Term']=='Nan'\
            or r['Property_Area']=='Nan':
                count+=1
                continue
            
            #성별 필터
            if r['Gender'] == 'Male':
                r['Gender'] = 0#남자면 0
            else:
                r['Gender'] = 1#여자면 1
            
            #결혼 여부 필터
            if r['Married'] == 'Yes':
                r['Married'] = 0#결혼했으면 0
            else:
                r['Married'] = 1#안했으면 1
                
            #부양 가족수 숫자만 필터
            r['Dependents'] = r['Dependents'][0]
            
            #졸업 여부 필터
            if r['Education'] == 'Graduate':
                r['Education'] = 0#졸업했으면 0
            else:
                r['Education'] = 1#안했으면 1
            
            #자영업자 여부 필터
            if r['Self_Employed'] == 'Yes':
                r['Self_Employed'] = 0#자영업자면 0
            else:
                r['Self_Employed'] = 1#아니면 1
                
            #부동산 필터
            if r['Property_Area'] == 'Urban':
                r['Property_Area'] = 0#Urban 0
            elif r['Property_Area']== 'Semiurban':
                r['Property_Area'] = 1#Semiurban 1
            else:
                r['Property_Area'] = 2#Rural면 2
                
            #신용도 필터
            if r['Credit_History'] == 1:
                r['Credit_History'] = "긍정적"
            elif r['Credit_History'] == 0:
                r['Credit_History'] = "부정적"
            else:
                r['Credit_History'] = "알수없음"
                
            t = (r['Gender'], r['Married'], r['Dependents'], r['Education'],
                 r['Self_Employed'], r['ApplicantIncome'],
                 r['CoapplicantIncome'], r['LoanAmount'], r['Loan_Amount_Term'],
                 r['Credit_History'], r['Property_Area'], r['Loan_Status'])
            
            try:
                self.cur.execute(insert_sql, t)
                self.conn.commit()
            except Exception as e:
                print(t)
                print(e)
                break
        

    def load_data_for_binary_classification(self, status):
        sql = "select * from loan;"
        self.cur.execute(sql)
    
        data = self.cur.fetchall()
        
        self.X = [ (t['gender'], t['married'], t['dependents'], t['education'], t['self_Employed']\
                    , t['applicantIncome'], t['coapplicantIncome'], t['loanAmount'], t['loan_Amount_Term']\
                    ,1 if t['credit_History']=="긍정적" else 0 if t['credit_History']=="부정적" else 2 ,t['property_Area'] ) for t in data ]
        
        self.X = np.array(self.X)
        #벡터화 연산을 위해 numpy 배열에 저장
    
        self.y = [ 1 if (t['loan_Status'] == status) else 0 for t in data]
        #list comprehension으로 y를 뽑고 여기에 label 설정
        self.y = np.array(self.y)   
        #벡터화 연산을 위해 numpy 배열에 저장 
        
            
    def data_split_train_test(self):
    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=42, test_size=0.3)
        #train data와 test data를 나누고 이에 맞게 x, y도 나눔
        #random state를 고정해서 주는 이유는 그냥 재현을 위함
        #비율은 7:3으로 나누기
        #그 결과 4개의 array를 반환
        
    def train_and_test_model(self, number=0):
        if number==0:
            print("LogisticRegression")
            log_reg = LogisticRegression(max_iter=10000)  # 로지스틱 회귀 객체
            model = log_reg.fit(self.X_train, self.y_train)  # 로지스틱 회귀로 학습해서 모델 생성
        elif number==1:
            print("KNeighborsClassifier")
            knn = KNeighborsClassifier(n_neighbors=5)  # K-NN 객체, n_neighbors는 이웃의 수를 설정
            model = knn.fit(self.X_train, self.y_train)  # K-NN으로 학습해서 모델 생성
        else:
            print("GradientBoostingClassifier")
            gboost = GradientBoostingClassifier()  # gboost 객체
            model = gboost.fit(self.X_train, self.y_train)  # gboost 학습해서 모델 생성
            
        self.y_predict = model.predict(self.X_test)  # 학습된 모델을 이용해서 예측
        # y_predict가 test 결과로 예측한 것
        
    #성능 평가 메서드
    def classification_performance_eval_binary(self, y_test, y_predict):
        tp, tn, fp, fn = 0,0,0,0
        
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
        
        print("accuracy=%.2f" %accuracy)
        print("precision=%.2f" %precision)
        print("recall=%.2f" %recall)
        print("f1 score=%.2f" %f1_score)
        print()


    #K-fold 교차 검증
    #K-fold는 미리 데이터를 나누고 학습을 하고 하는 게 아니라 x, y데이터를 준비하는 로드까지는 동일하고
    #바로 K-fold에서 데이터를 나누고 학습하는 것을 K번 학습을 진행하고 모든 것들을 평균해서 측정
    def binary_KFold_performance(self, number=0):
        accuracy = []
        precision = []
        recall = []
        f1_score = []
    
        if number==0:
            print("LogisticRegression")
            model = LogisticRegression(max_iter=10000)
        elif number==1:
            print("KNeighborsClassifier")
            model = KNeighborsClassifier(n_neighbors=5)
        else:
            print("GradientBoostingClassifier")
            model = GradientBoostingClassifier()
                
        cv_results = cross_validate(model, self.X, self.y, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'])
        #cross_validate라는 메서드를 통해서 k번의 교차검증을 수행해서 이 메서드로는 KFold 객체도 필요 없음
        #딕셔너리를 반환
        
        for metric, scores in cv_results.items():
            #cv_results라는 딕셔너리에서 각 아이템을 순회하면서 이용
            if metric.startswith('test_'):
                print(f'{metric[5:]}: {scores.mean():.2f}')   
        print()
        
#이진 분류하는데 내가 test와 train을 나누고 성능 평가하는 함수
def binary_train_test_performance(number):
    clf = class_loan_classification(import_data_flag=True)#clf는 classification을 의미
    clf.load_data_for_binary_classification(status='Y')
    clf.data_split_train_test()#train data와 test data 분류
    clf.train_and_test_model(number)#모델 생성 및 학습 진행
    clf.classification_performance_eval_binary(clf.y_test, clf.y_predict)#성능 평가
    
#K-fold 교차검증을 이용해서 data를 나눠서 모델을 학습하고 그에 맞는 성능 평가하는 함수
#K-Fold 교차검증은 상대적으로 안정적인 학습 결과를 제공함
#이는 성능 평가함수들로 알 수 있고 여러 개의 결과를 평균을 내기 때문에 분산이 줄어들어 상대적으로 안정적인 결과를 도출함
def binary_KFold_performance(number):
    clf = class_loan_classification(import_data_flag=False)
    clf.load_data_for_binary_classification(status='Y')
    clf.binary_KFold_performance(number)

if __name__ == "__main__":
    #number가 0이면 로지스틱 회귀, 1이면 KNN, 2이면 Gradient Boosting
    for i in range(3):
        print("Random Sampling")
        binary_train_test_performance(i)
        print("K-fold Cross Validation")
        binary_KFold_performance(i)
        print()
