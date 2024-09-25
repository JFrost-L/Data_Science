'''
List 를 입력받아 소팅하는 Bubble sort 함수
bubble_sort(a) 를 정의할 것.
• list 변수 a 에 임의의 1과 1000 사이의 정수
10개를 입력하고, 출력할 것.
• bubble_sort()를 호출하여, 소팅이된 list를
출력할 것.
'''
import random
def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i]>arr[j]:
                arr[i], arr[j]=arr[j], arr[i]
    return arr

def make_randomNumber(arr):
    for i in range(10):
        arr.append(random.randint(1,1000))
    printArr(arr)

def printArr(a):
    for i in a:
        print(i, end=" ")
    print()
    
list=[]

make_randomNumber(list)
sort_list = bubble_sort(list)
printArr(sort_list)
