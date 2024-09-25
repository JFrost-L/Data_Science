# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 16:20:25 2023

@author: L
"""

'''
1. Pandas & DataFrame
• Pandas: Python Data Analysis Library
• DataFrame: Pandas 가 제공하는 테이블 형태의(row, col) 데이터 분석을 위한 자료구조
-> 속도면에서는 sql이 더 빠름
• ndarray, dict, list 등과 데이터 호환 및 변환 가능
• Excel/csv 포맷의 데이터파일을 쉽게 import 할 수 있다는 장점이 있음
-> 매우 강력한 기능
--> read_excel(), read_csv()

import pandas as pd
import numpy as np
xls_file = "score.xlsx"
df = pd.read_excel(xls_file)
#엑셀을 데이터 프레임으로 변경
for index, row in df.iterrows():
    #index : 행번호
    row : 그 행의 정보
    print(row)
    print(f"sno = {row['sno']}, score = {row['score']}, grade = {row['grade']}")
'''

'''
2. kaggle을 이용한 실습
'''

import pandas as pd
import numpy as np
csv_file = "top_movies.csv"
df = pd.read_csv(csv_file)
#엑셀을 데이터 프레임으로 변경
for index, row in df.iterrows():
    print(f"Movie Name = {row['Movie Name']}")
