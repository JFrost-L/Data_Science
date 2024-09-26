from db_conn import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix

class class_iris_classification():
    def __init__(self, import_data_flag=True):
        self.conn, self.cur = open_db()
        if import_data_flag:
            self.import_iris_data()
        

    def import_iris_data(self):
        drop_sql =""" drop table if exists iris;"""
        self.cur.execute(drop_sql)
        self.conn.commit()
    
        create_sql = """
            create table iris (
                id int auto_increment primary key,
                sepal_length float,
                sepal_width float,
                petal_length float,
                petal_width float,
                species varchar(10), 
                enter_date datetime default now() 
                ); 
        """
    
        self.cur.execute(create_sql)
        self.conn.commit()
    
        file_name = 'iris.csv'
        iris_data = pd.read_csv(file_name)
    
        rows = []
    
        insert_sql = """insert into iris(sepal_length, sepal_width, petal_length, petal_width, species)
                        values(%s,%s,%s,%s,%s);"""
    
        for t in iris_data.values:
            rows.append(tuple(t))
    
        self.cur.executemany(insert_sql, rows)
        self.conn.commit()

  
    
    def load_data_for_multiclass_classification(self):
        sql = "select * from iris;"
        self.cur.execute(sql)
    
        data = self.cur.fetchall()
        
        #print("data=", data)
    
        #self.X = [ (t['sepal_length'], t['sepal_width'], t['petal_length'], t['petal_width'] ) for t in data ]
        #self.X = [ (t['sepal_length'], t['sepal_width'] ) for t in data ]
        self.X = [ (t['sepal_length'], t['petal_length'] ) for t in data ]

        self.X = np.array(self.X)
        #여기까지는 동일
        
        #여기서부터 labeling을 0, 1, 2로 모든 클래스에 대해서 분기해서 대입
        self.y =  [0 if t['species'] == 'setosa' else 1 if t['species'] == 'versicolor' else 2 for t in data]
        self.y = np.array(self.y)    
        
        #print("X=",self.X)
        #print("y=", self.y)
        
        
    #로딩한 데이터에 대해서 split후에 학습 진행은 binary_Classficaion과 동일
    def data_split_train_test(self):
    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        
        '''
        print("X_train=", self.X_train)
        print("X_test=", self.X_test)
        print("y_train=", self.y_train)
        print("y_test=", self.y_test)
        '''
        
        

    #학습한 모델에 대해서 성능 평가는 binary_Classficaion과 다름
    def classification_performance_eval_multiclass(self, y_test, y_predict, output_dict=False):
        
        target_names=['setosa', 'versicolor', 'virginica' ]
        #출력 리포트에 표시할 레이블 이름을 제공할 리스트
        labels = [0,1,2]
        #labels 매개변수는 리포트에 포함할 클래스 레이블을 지정
        #혹은 confusion_matrix에 포함할 레이블을 지정
        
        #출력 리포트!
        self.classification_report = classification_report(y_test, y_predict, target_names=target_names, labels=labels, output_dict=output_dict)
        #y_test와 y_predict만으로 결과 제공 output은 딕셔너리(computation 용도) 혹은 String(사람이 보는 용도)
        #분류 모델의 성능을 평가하여 리포트하는 함수
        #이 함수는 주어진 분류 모델의 정밀도(Precision), 재현율(Recall), F1 점수(F1-Score), 그리고 지원도(Support)를 계산
        
        #confusion_matrix
        self.confusion_matrix = confusion_matrix(y_test, y_predict, labels=labels)
        #confusion_matrix도 y_test와 y_predict만으로 결과 제공
        #분류 모델의 성능을 평가하여 confusion matrix 를 생성하는 함수
        
        print(f"[classification_report]\n{self.classification_report}")
        #결과에서 macro avg는 각 클래스에서 대한 accuracy
        #결과에서 weighted avg가 더 정확한 것으로 
        
        print(f"[confusion_matrix]\n{self.confusion_matrix}")
        #y_test * y_predict의 행렬로서 계산
        #결과에서 3행에서 문제 존재 즉 2개가 틀린 결과를 도출한 것

    def train_and_test_dtree_model(self):
        dtree = tree.DecisionTreeClassifier()    
        dtree_model = dtree.fit(self.X_train, self.y_train)
        self.y_predict = dtree_model.predict(self.X_test)




    def multiclass_dtree_KFold_performance(self):
        accuracy = []
        precision = []
        recall = []
        f1_score = []

        kfold_reports = []
    
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        #K-Fold 객체 생성하는데 k를 5로 random_state는 42로 섞어서 생성
    
        for train_index, test_index in kf.split(self.X):
            #kf를 X에 대해서 split하면 순서적으로 k만큼 루핑하는 iterable data
            #거기서 각 test와 train에 대한 인덱스에 해당하는 것들을 읽어옴
            #이 때 각 test와 train에 대한 index는 array임 이는 slicing 매커니즘으로 진행
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            
            dtree = tree.DecisionTreeClassifier()
            dtree_model = dtree.fit(X_train, y_train)#학습
            y_predict = dtree_model.predict(X_test)#예측
            self.classification_performance_eval_multiclass(y_test, y_predict, output_dict=True)
            #성능 평가 지표 생성
            kfold_reports.append(pd.DataFrame(self.classification_report).transpose())
        #암튼 교차 검증을 k회 돌려서 평균을 제공하는데 위는 cross_validate 메서드의 구현 내용임
            
        for s in kfold_reports:
            print('\n', s)
            
        mean_report = pd.concat(kfold_reports).groupby(level=0).mean()
        print('\n\n', mean_report)
        
        
def multiclass_dtree_train_test_performance():
    clf = class_iris_classification(import_data_flag=False)
    clf.load_data_for_multiclass_classification()
    #multiclass_classification은 데이터 로딩부터 다름
    clf.data_split_train_test()
    clf.train_and_test_dtree_model()
    clf.classification_performance_eval_multiclass(clf.y_test, clf.y_predict)




def multiclass_dtree_KFold_performance():
    clf = class_iris_classification(import_data_flag=False)
    clf.load_data_for_multiclass_classification()
    clf.multiclass_dtree_KFold_performance()


if __name__ == "__main__":
    # multiclass_dtree_train_test_performance()
    multiclass_dtree_KFold_performance()
