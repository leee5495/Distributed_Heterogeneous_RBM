# Distributed Heterogeneous RBM 실험 결과

### 실험 데이터 설명
- 인터넷 서점(알라딘)에서 크롤링한 사용자 리뷰 데이터 사용
- 12,000여 권의 도서에 대한 리뷰를 크롤링한 후 2개 미만의 리뷰를 가지는 사용자와 도서 필터링
- 최종적으로 총 6,000여 개의 도서와 4,500여 명의 사용자에 대한 리뷰 데이터 사용

  사용자 수 | 아이템 수 | Rating 수 | 평가   밀도(density)
  -- | -- | -- | --
  4542 | 5992 | 55019 | 0.20%

##
### 1. Distributed RBM의 출력 통합 방법의 비교 (compare_output_method.py)
![image](https://user-images.githubusercontent.com/39192405/94337317-84193800-0024-11eb-906c-543bdb534364.png)
- `M-Selection`: 입력 사용자가 속한 클러스터에 대한 RBM의 출력만을 사용
- `M-Weighted`: 입력 사용자에 대한 은닉벡터와 K-means centroid와의 거리를 가중치로 사용해 각 RBM에 대한 가중 평균을 사용
- `M-Ensemble`: MLP를 사용해 RBM의 은닉벡터들을 모아 각 아이템에 대한 점수를 예측하는 앙상블 네트워크를 만들어 사용

   &nbsp; | M-Selection | M-Weighted | M-Ensemble 
  -- | -- | -- | -- 
  HR@10 | 0.05947 | 0.06278 | 0.07489 
  HR@25 | 0.1200 | 0.1167 | 0.1233 
  ARHR | 0.003786 | 0.003541 | 0.004342 
  Time(sec) | 1.0300 | 1.0040 | 0.5230

  **실험 결과 M-Ensemble 방식이 다른 두 개의 방법보다 더 높은 성능을 보이는 것을 확인할 수 있었다**

##
### 2. Baseline 추천 시스템과의 성능 비교 (compare_baselines.py)
![image](https://user-images.githubusercontent.com/39192405/93019262-d1eb7480-f610-11ea-8473-92b9616b0ee5.png)
- `M-Ensemble`: 제안 모델
- `Itempop`: 제일 인기가 많은 아이템 추천
- `Itempop-Cluster`: 사용자 군집에서 제일 인기가 많은 아이템 추천
- `SVD`: 특이값 분해 Matrix Factorization을 이용한 협업 필터링
- `NMF`: NMF Matrix Factorization을 이용한 협업 필터링

    | M-Ensemble | ItemPop | ItemPop-Cluster | SVD | NMF
  -- | -- | -- | -- | -- | --
  HR@10 | 0.07449 | 0.01872 | 0.02093 | 0.05617 | 0.01432
  HR@25 | 0.1233 | 0.03524 | 0.04295 | 0.1090 | 0.03414
  ARHR | 0.004342 | 0.0007325 | 0.001186 | 0.003223 | 0.0008671
  Time(sec) | 0.6060 | 0.08100 | 0.8650 | 7.176 | 11.43

  **실험 결과 상대적으로 적은 시간 안에 제안한 모델의 성능이 더 높은 것을 볼 수 있었다**

##
### 3. 단일 RBM과의 성능 비교 (compare_control.py)
![image](https://user-images.githubusercontent.com/39192405/93019413-d2d0d600-f611-11ea-91a8-cc54bbd56b00.png)
- `M-Ensemble`: 제안 모델
- `Control-256`: 은닉벡터의 크기가 256인 RBM
- `Control-512`: 은닉벡터의 크기가 512인 RBM
- `Control-1024`: 은닉벡터의 크기가 1024인 RBM

    | M-Ensemble | Control-256 | Control-512 | Control-1024
  -- | -- | -- | -- | --
  HR@10 | 0.07449 | 0.06278 | 0.05507 | 0.04956
  HR@25 | 0.1233 | 0.1178 | 0.1167 | 0.1123
  ARHR | 0.004342 | 0.003378 | 0.004011 | 0.004243
  Time(sec) | 0.6060 | 0.2290 | 0.3210 | 0.4900

  **실험 결과 다중RBM의 성능이 단일RBM의 성능보다 높은 것을 관찰하였다**
