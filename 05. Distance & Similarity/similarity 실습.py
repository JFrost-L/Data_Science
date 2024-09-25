
'''
1. minkowski distance for r=1, 2, ...
2. Cosine similarity and dot product
3. most similar k

'''

import numpy as np
#벡터는 numpy로 표현

def euclidean_distance(vector1, vector2):
    return np.linalg.norm(np.array(vector1) - np.array(vector2))
    #norm()으로 두 벡터간의 거리 구함



def minkowski_distance(vector1, vector2, p):#p는 차원을 결정
    if p == np.inf:#p가 maximum인 경우
        return np.max(np.abs(np.array(vector1) - np.array(vector2)))
        #두 벡터의 각 차원의 차이 중 가장 큰 값!
    else:
        return np.sum(np.abs(np.array(vector1) - np.array(vector2))**p)**(1/p)
        #공식 계산


def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    #내적!
    vector1_len = np.linalg.norm(vector1)
    vector2_len = np.linalg.norm(vector2)
    
    similarity = dot_product / (vector1_len * vector2_len)
    
    return similarity



def dot_product(vector1, vector2):#두 벡터의 내적
#내적 연산시 두 벡터의 유닛 벡터로 해야 시간이 절약된다.
    return np.dot(vector1, vector2)   



def topk_vectors(one_vector, vectors, k):

    similarities = [cosine_similarity(one_vector, v) for v in vectors]

    topk_indices = np.argsort(similarities)[-k:][::-1] #뒤에서 k개를 reverse해서 slicing하기
    #즉, 유사도가 가장 높은 인덱스의 k개 가져오기
    #argsort로 정렬 : sort를 직접하는 게 아닌 내부의 값을 보고 정렬한 인덱스를 리턴해줌
    
    topk_indices = np.argsort(similarities)[::-1][:k]
    #[start:stop:step]

    topk_vectors = vectors[topk_indices]
    #해당 인덱스로 값 찾기
    
    print(topk_vectors)
    


if __name__ == '__main__':
    #이미지나 텍스트를 벡터 데이터로 변환시키고 유사도를 계산

    dim = 10#차원 수

    vector1 = np.random.randint(0, 100, dim)
    vector2 = np.random.randint(0, 100, dim)
    
    print(f"""minkowski_distance
          norm1(v1, v2) = {minkowski_distance(vector1, vector2, 1)}
          norm2(v1, v2) = {minkowski_distance(vector1, vector2, 2)}
          norm_max(v1, v2) = {minkowski_distance(vector1, vector2, np.inf)}
          """)
    
    num_vectors = 1000#1000개의 vector 생성!
    vectors = np.random.randint(0, 101, (num_vectors, dim))
    
    print("Cosine Similarity")
    topk_vectors = topk_vectors(vector1, vectors, k=3)
    #vector1과 1000개의 벡터 중에서(vectors) 유사도가 가장 가까운 3개를 뽑아내기
       