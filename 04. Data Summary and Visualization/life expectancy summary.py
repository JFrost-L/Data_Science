import pandas as pd
import numpy as np

def adjust_data_frame(df):
    df['Males'] = df['Males'].str.replace(' years', '').astype('float')
    df['Females'] = df['Females'].str.replace(' years', '').astype('float')


def mean_median_mode(df):
    mean = {}
    median = {}
    mode = {}
    
    for c in ['Males', 'Females', 'Birth rate', 'Death rate']:
        mean[c] = df[c].mean()
        median[c] = df[c].median()
        mode[c] = dict(df[c].mode())
        
    print(f"mean = {mean}\n")
    print(f"median = {median}\n")
    print(f"mode = {mode}\n")
    
def std_var(df):
    std = {}
    var = {}

    
    for c in ['Males', 'Females', 'Birth rate', 'Death rate']:
        std[c] = df[c].std()
        var[c] = df[c].var()

        
    print(f"std = {std}\n")
    print(f"var = {var}\n")



import matplotlib.pyplot as plt 

def percentile(df):

    p = [x for x in range(0, 101, 10)]

    for c in ['Males', 'Females', 'Birth rate', 'Death rate']:
        percentile = np.percentile(df[c], p)
        plt.plot(p, percentile, 'o-')
        plt.xlabel('percentile')
        plt.ylabel(c)
        plt.xticks(p)
        plt.yticks(np.arange(0, max(percentile)+1, max(percentile)/10.0))
        plt.show()


def boxplot(df):
    boxplot = df[['Males', 'Females', 'Birth rate', 'Death rate']].boxplot()
    plt.show()  

def histogram(df):
    for c in ['Males', 'Females', 'Birth rate', 'Death rate']:
        plt.hist(df[c], facecolor='blue', bins=5)
        plt.xlabel(c)
        plt.show()
        
def scatter_plot(df):
    for c1 in ['Males', 'Females', 'Birth rate', 'Death rate']: 
        for c2 in ['Males', 'Females', 'Birth rate', 'Death rate']:
            if c1 == c2:
                continue
            plt.scatter(df[c1], df[c2])
            plt.xlabel(c1)
            plt.ylabel(c2)
            plt.show()


if __name__ == '__main__':
    csv_file = 'Life expectancy.csv'
    
    df = pd.read_csv(csv_file)
    adjust_data_frame(df)
    
    #mean_median_mode(df)
    #std_var(df)
    #percentile(df)
    #boxplot(df)
    #histogram(df)
    scatter_plot(df)
    
    
    
    