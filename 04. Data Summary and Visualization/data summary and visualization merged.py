import pandas as pd
import numpy as np


def adjust_data_frame(df):
    df['Males'] = df['Males'].str.replace(' years', '').astype(float)
    df['Females'] = df['Females'].str.replace(' years', '').astype(float)
    

def mean_median_mode(df, attr_list):
    mean = {}
    median = {}
    mode = {}
    for m in attr_list:
        mean[m] = df[m].mean()
        median[m] = df[m].median()
        mode[m] = dict(df[m].mode())
        
    print(f"mean={mean}")
    print(f"median={median}")
    print(f"mode={mode}")
    
    
def std_var(df, attr_list):

    std = {}
    var = {}
    for m in attr_list:
        std[m] = df[m].std()
        var[m] = df[m].var()
        
    print("std=", std)
    print("var=", var)   


import matplotlib.pyplot as plt
    
def percentile(df, attr_list):    
    p = [x for x in range(0,101,10)]
    for col in attr_list:
        percentile = np.percentile(df[col], p)
        plt.plot(p, percentile, 'o-')
        plt.xlabel('percentile')
        plt.ylabel(col)
        plt.xticks(p)
        plt.yticks(np.arange(0, max(percentile)+1, max(percentile)/10.0))
        plt.show()
        
def boxplot(df, attr_list):
    boxplot = df[attr_list].boxplot()
    plt.show()    

def histogram(df, attr_list):
    for col in attr_list:
        plt.hist(df[col], facecolor='blue', bins=20)
        plt.xlabel(col)
        plt.show()        

def scatter_plot(df, attr_list1, attr_list2):
    for col1 in attr_list1:  
        for col2 in attr_list2:  
            if col1 == col2:
                continue
            plt.scatter(df[col1], df[col2])
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.show()   
            
def pairplot(df, attr_list):        
    import seaborn as sns        
    sns.pairplot(df[attr_list])




if __name__ == '__main__':
    csv_file = 'Life expectancy.csv'
    attr_list1 = ['Males', 'Females', 'Birth rate', 'Death rate']
    df1 = pd.read_csv(csv_file)    
    adjust_data_frame(df1)

    #mean_median_mode(df1, attr_list1)
    #std_var(df1, attr_list1)
    #percentile(df1, attr_list1)
    #boxplot(df1, attr_list1)
    #histogram(df1, attr_list1)
    #scatter_plot(df1, attr_list1)
    #pairplot(df1, attr_list1)


    
    csv_file = '2019 happiness index.csv'
    attr_list2 = ['Score', 'GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
    df2 = pd.read_csv(csv_file)    
  


    #mean_median_mode(df2, attr_list2)
    #std_var(df2, attr_list2)
    #percentile(df2, attr_list2)
    #boxplot(df2, attr_list2)
    #histogram(df2, attr_list2)
    #scatter_plot(df2, attr_list2)
    #pairplot(df2, attr_list2)


    df = pd.merge(df1, df2, on='Country', how='inner')    
    attr_list = attr_list1 + attr_list2

    #df.to_csv('merged.csv', index=False)
    
    #mean_median_mode(df, attr_list)
    #std_var(df, attr_list)
    #percentile(df, attr_list)
    #boxplot(df, attr_list)
    #histogram(df, attr_list)
    #scatter_plot(df, attr_list1, attr_list2)
    pairplot(df, attr_list)
