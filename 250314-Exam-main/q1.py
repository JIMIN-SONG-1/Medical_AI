import numpy as np

x=[[1,2,3], [4,5,6], [7,8,9]]

def solution(data):
    #리스트 x를 Numpy 배열로 출력합니다.
    array1 = None

    #모든 원소가 0인 3×3의 배열을 출력합니다.
    array2 = None

    #모든 원소가 1인 2×5의 배열을 출력합니다.
    array3 = None

    # 0 이상 1 미만의 랜덤 값을 갖는 5×3의 배열을 출력합니다.
    array4 = None    

    # 0부터 9까지의 값을 원소로 하는 1×10 배열을 출력합니다.
    array5 = None
    
    return array1, array2, array3, array4, array5

def print_answer(**kwargs):
    for key in kwargs.keys():
        print(key,"\n", kwargs[key], "\n")

array1, array2, array3, array4, array5 = solution(x)

print_answer(array1=array1, array2=array2, array3=array3, array4=array4, array5=array5)
