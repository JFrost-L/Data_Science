import pandas as pd
import numpy as np

def adjust_data_frame(df):
    #DataFrame의 데이터를 오직 숫자로 float 형식으로만 보고 싶어서 조작하는 함수
    #years를 빈칸으로 replace
    df['Males'] = df['Males'].str.replace(' years', '').astype('float')
    df['Females'] = df['Females'].str.replace(' years', '').astype('float')

def mean_median_mode(df, attr_list):
    #DataFrame의 평균, 중앙값, 최빈값을 column별로 출력하는 함수
    mean = {}
    median = {}
    mode = {}
    
    for c in attr_list:
        mean[c] = df[c].mean()
        median[c] = df[c].median()
        mode[c] = dict(df[c].mode())
        
    print(f"mean = {mean}\n")
    print(f"median = {median}\n")
    print(f"mode = {mode}\n")
    
def std_var(df, attr_list):
    #DataFrame의 분산과 표준편차를 column별로 계산해서 출력하는 함수
    std = {}
    var = {}

    
    for c in attr_list:
        std[c] = df[c].std()
        var[c] = df[c].var()

        
    print(f"std = {std}\n")
    print(f"var = {var}\n")



import matplotlib.pyplot as plt 

def percentile(df, attr_list):
    #DataFrame의 percentile을 column별로 출력하는 함수
    p = [x for x in range(0, 101, 10)]

    for c in attr_list:
        percentile = np.percentile(df[c], p)#column과 percentile 기준 값을 리스트로 제공
        plt.plot(p, percentile, 'o-')
        plt.xlabel('percentile')
        plt.ylabel(c)
        plt.xticks(p)
        plt.yticks(np.arange(0, max(percentile)+1, max(percentile)/10.0))
        plt.show()


def boxplot(df, attr_list):
    #DataFrame의 boxplot을 column별로 출력하는 함수
    boxplot = df[attr_list].boxplot()
    #인자 주의!!
    plt.show()  

def histogram(df, attr_list):
    #DataFrame의 histogram을 column별로 출력하는 함수
    for c in [attr_list]:
        plt.hist(df[c], facecolor='blue', bins=5)
        #bin은 표현할 막대의 개수 즉, 5등분!
        plt.xlabel(c)
        plt.show()
        
def scatter_plot(df, attr_list):
    #DataFrame의 scatter_plot을 출력하는 함수
    #다른 것들과의 차이는 2개의 column간의 연관성을 비교
    for c1 in attr_list: 
        for c2 in attr_list:
            if c1 < c2:
                continue
            plt.scatter(df[c1], df[c2])
            plt.xlabel(c1)
            plt.ylabel(c2)
            plt.show()
            
def pairplot(df, attr_list):
    #DataFrame의 scatter_plot과 histogram을 쌍으로 볼 수 있게 해주는 함수
    import seaborn as sns
    sns.pairplot(df[attr_list])
    #인자 주의할 것!!

if __name__ == '__main__':
    csv_file1 = 'Life expectancy.csv'
    csv_file2 = '2019 happiness index.csv'
    
    df1 = pd.read_csv(csv_file1)
    adjust_data_frame(df1)
    df2 = pd.read_csv(csv_file2)
    
    attr_list1 = ['Males', 'Females', 'Birth rate', 'Death rate']
    attr_list2 = ['Score', 'GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices',
              'Generosity', 'Perceptions of corruption']
    
    df = pd.merge(df1, df2, on = 'Country', how = 'inner')
    attr_list = attr_list1 + attr_list2
    
    df.to_csv('merged Dataframe.csv')
    
    
    #mean_median_mode(df)
    #std_var(df)
    #percentile(df)
    #boxplot(df)
    #histogram(df)
    scatter_plot(df, attr_list)
    #pairplot(df)
    
    
    