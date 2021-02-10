- 기계 학습 분야에서는 일반적으로 분석적으로 식을 풀어서 답을 얻는 것이 아니라, 조금씩 조금씩 추정치를 바꾸는 작업을 여러 번 반복하여 문제를 해결한다.
- 대다수의 작업은 선형 수식을 풀거나 최소, 최대값을 찾는 최적화 과정인데, 실수(real number)가 포함되는 계산의 경우 디지털 컴퓨터로 정확히 처리하지 못할 수도 있다.

## 4.1 Overflow and Underflow
- 디지털 컴퓨터로 숫자를 처리하는 이상 rounding error가 발생할 수 밖에 없다.
  - underflow: 0에 가까운 숫자가 0으로 처리되어 발생하는 문제
    - 예) 의도치않게 분모에 0이 입력되어 계산 오류가 발생 할 수 있다.
  - overflow: 큰 숫자가 $\pm\infty$로 처리되어 발생하는 문제
    - 예) 안정화되지 않은 값의 원소를 가지는 벡터 x의 softmax 함수 값 계산 (Eq 4.1)
      - x의 원소가 모두 c일 경우: 1/n
      - 하지만 c가 매우 작을 경우: underflow가 발생해 값이 정의되지 않는다.
      - c가 매우 클 경우: overflow가 발생해 값이 정의되지 않는다.
      - 해결 예시) softmax(z) where z = x - max,i(xi)로 만들어 큰 element를 없애 문제를 해결한다.

## 4.2 Poor conditioning
- Conditioning: 입력 변수의 작은 변화에 대해 함수 값이 얼마나 빨리 변하는지
- $A \in \mathbb{R}^{nn}$이 eigenvalue decomposition을 가질 때, condition number = (Eq 4.2)
  - 가장 크고 작은 eigenvalue 사이의 비율
  - 이 숫자가 크면 역행렬을 계산하는 등의 작업이 input의 에러에 대해 큰 차이를 만들 수 있음
  - 역행렬을 계산할때의 rounding error 때문이 아니라, 행렬 고유의 성질임
  - Poorly conditioned 행렬은 입력 신호 단계에서 발생한 에러를 증폭 시킴

## 4.3 Gradient-Based optimization
- 대부분의 딥러닝 알고리즘은 f(x)의 최대값이나 최소값에 대한 x를 구하는 최적화 과정을 포함한다.
- f(x): objective function 혹은 criterion, 최소화해야 할때는 cost function, loss function, error function
- f(x)를 최적화해야 할 때 *로 표시 -> x ∗ = arg min f ( x )
- Gradient descent: x 값을 약간 바꾸었을 때 f(x) 값이 감소한다면, 해당 방향으로 x 값을 수정하는 과정을 반복하여 f(x)의 최소값을 만드는 x를 찾는 방법
- (fig 4.1)
- f’(x) = 0: critical point, stationary point (local minimum or maximum of saddle point)
- (fig 4.2)
- Global minimum: 전체 정의역에 대해 가장 작은 함수값
  - 여러개의 local minima를 갖거나 평평한 saddle point가 많은 경우에는 optimization이 어렵다.
- (fig 4.3)
- f: Rn->R의 경우, 편미분을 사용하여 gradient를 계산: $\nabla_{x}f(x)$
  - critical point: 모든 element가 0
- directional derivative in direction u (a unit vector)
  - 함수 f의 u  방향으로의 기울기
  - 함수 f를 최소화하기 위해 f를 가장 빠르게 감소시키는 방향을 찾고자 함
  - (Eq 4.3, 4.4)
  - u에 무관한 텀을 무시하면, min u cos theta로 간소화되고, u가 gradient와 반대 방향일 때 최소가 됨
  - 위의 방법을 method of steepest descent 혹은 gradient descent라고 함
- 위에서 구한 방향으로 새로운 x값을 찾음
  - (eq 4.5)
  - $\epsilon$: learning rate
  - Learning rate는 보통 상수를 사용하나, line search라는 방법에서는 몇 가지 learning rate 값을 동시에 테스트해서 f를 최소화 시키는 값을 고름

### 4.3.1 Beyond the Gradient: Jacobian and Hessian Matrices
- Jacobian matrix: f: Rm -> Rn에서 모든 편미분에 대한 행렬
- Second derivative: 미분에 대한 미분으로, curvature를 측정한다고 볼 수 있음
  - (fig 4.4)
  - -: 아래로 볼록, cost function이 입실론 보다 많이 감소
  - 0: curvature가 없음, cost function의 기울기가 1일 때, 입실론 만큼 감소
  - +: 위로 볼록, cost function이 입실론 보다 적게 감소
- Hessian matrix
  - (Eq 4.6)
  - 다변수를 입력 받을 때의 모든 조합을 고려하는 second derivative와 같은 개념
  - Gradient의 Jacobian이라 할 수 있음
  - 미분의 순서가 바뀌어도 값은 유지되므로, 2차 미분이 continuous하다면 symmetric matrix임
  - Real, symmetric하므로 real eigenvalues와 orthogonal basis eigenvector로 분해 가능
  - $d^{T}Hd$: unit vector $d$방향으로의 2차 미분
  - 2차 미분과 Hessian matrix를 이용하여 적절한 learning rate를 예상 할 수 있음
    - (eq 4.8)
    - (eq 4.9)
    - (eq 4.10)
    - 최악의 경우: g가 가장 큰 eigenvalue lamda_max와 상응하는 H의 eigenvector의 방향과 일치 -> step size = 1/lamda_max가 됨
- Hessian matrix, H를 이용하면 critical point의 성질을 알 수 있음
  - H가 positive definite (모든 eigenvalue가 양수): local minimum
  - H가 negative definite (모든 eigenvalue가 음수): local maximum
  - 양/음인 eigenvalue가 모두 있음: 방향에 따라 min/max 여부가 다름
  - (fig 4.5)
- Hessian의 condition number가 클 때, gradient descent는 효과가 좋지 않은 예시
  - 한 방향으로는 급격하게 변하고 (예-$\lamda_{max}$에 대한 eigenvector 방향), 다른 방향으로는 조금 변함 (예-$\lamda_{min}$에 대한 eigenvector 방향)
  - (fig 4.6)
- Newton’s method로 이와 같은 문제를 해결 가능
  - second-order Taylor series expansion을 이용해 x(0) 근처의 f(x)를 근사함
  - (Eq 4.11, 4.12)
  - 함수 f가 positive deﬁnite quadratic 이면 Newton’s method는 위 식을 이용해서 한번에 minimum으로 갈 수 있음
  - 하지만 현실적으로 f는 국소적으로 positive deﬁnite quadratic이고 전체적으로는 아니므로, 여러번 반복해야 함
  - 이와 같이 gradient descent보다 더 빠르게 critical point로 도달할 수 있지만, local minimum 근처 한정이며 saddle point에서는 오히려 안 좋을 수 있음
- First-order optimization algorithms의 예시 - gradient 만을 이용하는 gradient descent
- Second-order optimization예시 - Hessian matrix를 이용하는 Newton’s method
- 함수에 제약을 걸어 성능을 보장하기도 함: 예) Lipschitz continuous 혹은 Lipschitz continuous derivatives를 가지는 함수
  - 변화율이 Lipschitz constant L에 의해 제한되는 함수 f
  - (Eq 4.13)
  - 입력의 변화가 작을 때, 출력의 변화가 작을 것이라 보장함
- Convex optimization: 강한 제약을 이용해 좋은 성능을 보장함
  - 모든 지점에서 Hessian이 positive semidefinite (eigenvalue가 모두 0 이상)인 confex function에만 적용 가능함
  - 이러한 함수는 saddle point가 없고, local minima가 global minima라 최적화가 용이함

## 4.4 Constrained optimization
- 함수를 최적화 할 때, 모든 x에 대해서 최적화 하는 것이 아니라, 특정 집합 S에 속하는 x에 대해서 최적화 하는 방법
  - S: Feasible points
- Constrained optimization의 예시: Karush-Kuhn-Tucker (KKT)
  - 아래 그림 참고
![_config.yml]({{ site.baseurl }}/assets/Ch4_KKT.png)

## 4.5 Example: Linear Least Squares
- KKT와 linear least squares 계산 과정에 대해서는 아래 블로그 참조
  - https://leejunhyun.github.io/deep%20learning/2018/09/27/DLB-04/



## 2.1 Scalars, Vectors, Metrices, and Tensors

- scalar  $i$: 수, italics로 표기
- vector $\pmb{x}$: 여러 수를 특정 순서로 나열. 굵은 영문 소문자로 표기, 벡터의 각 항은 $x_i$처럼  subscript로 표기
- matrix $A$: 수를 2차원으로 배열한 것. 굵은 영어 대문자로 표현
- tensor **A**: 3차원 이상.
- Transpose(전치): 행렬의 행과 열을 바꾼 것.

### 2.2 행렬과 벡터의 곱셈

- $(AB)^T = B^{T}A^{T}$

### 2.3 단위행렬, 역행렬

- Identity matrix: $\forall x \in \mathbb{R}^{n}, I_{n}x = x$인 $I$
- inverse matrix: $A^{-1}A = I_n$인 $A^{-1}$

### 2.4 Linear dependence and span(생성공간)

- $A^{-1}$이 존재하기 위해서는 모든 b에 대해 $Ax=b$를 만족하는 해가 하나 있어야 한다.
- linear combination:  $Ax = \sum{x_iA_{:, i}}$
- 벡터집합 $\{ v^{(1)}, v^{(2)},... v^{(n)}\}$의 일차결합은 각 벡터에 스칼라 계수를 곱한 것.   
                                              

$\sum_i{c_iv^{(i)}}$
- span: 주어진 벡터 집합의 일차결합으로 얻을 수 잇는 모든 점의 집합
- 일차연립방정식 $Ax=b$의 해가 잇는지는 b가 $A$의 열들의 생성곤간에 속하는지로 판단 가능
- 일차종속: 어떤 벡터집합의 구성 요소의 조합으로 그 벡터집합의 어떤 요소를 만들 수 있는 경우
- 일차독립: 없는 경우

### 2.5 Norms (번역서엔 노름이라 되어있는데 건전하게 놂이라 읽자)

벡터의 크기 구하기. 

- n-th norm:

$$L^p = \parallel{x}\parallel_{p} =   (\sum_{i}|x_i|^p)^{1/p} 
$$

- L2 norm: Euclidean norm, $x^Tx$
- cost func에 많이 쓰는데, 큰, 작은 오차를 얼마나 강조하느냐에 따라 L1, L2 등등을 씀
- max norm: $\max_i|x_i|$ → 이것도 기계학습에서 씀
- 행렬 크기를 구해야 할 때 Frobenius norm(프로베니우스 놂)을 주로 쓴다. L2 norm이랑 비슷

$$||A||_F =\sqrt{\sum_{i,j}{A_{i, j}^{2}}} $$

### 2.6 Special kinds of Matrices and vectors

- Diagonal matrix(대각행렬): $A_{i, j} = 0 \text{ if } i\neq j$
- vector v로 square + diagonal matrix를 만들면 diag(v)로 표기
    - $diag(v)x$는 $v$의 각 성분 $v_i$에 $x_i$배를 한 것.
- symmetric matrix: $A = A^T$
- unit vector: L2 norm의 크기가 1인 vector
- 서로 orthogonal한 두 벡터: $x^Ty=0$
- orthogonal matrix(직교행렬):  각 행들이 서로 정규직교이도 열들도 서로 정규직교인 square matrix
- orthogonal matrix에서는 $A^TA = I$ 따라서 $A^{-1} = A^T$

### 2.7 Eigendecomposition(EVD)

수학적 대상 중에는 그것을 구성요소들로 분해하여 표현 방식과 무관하게 보편적인 어떤 성질을 찾아내면 더 잘 이해할 수 있는 것들이 많다. 

가장 널리 쓰이는 행렬 분해 방법: eigendecomposition(고윳값 분해)

square matrix A의 eigenvector v, eigen value $\lambda$는 아래를 만족한다. 

$$Av = \lambda v$$

$A$의 eigenvector의 집합, 혹은 직교행렬을 V$=\{v^{(1)},...v^{(n)}\}$라고 할 때 A의 Eigendecomposition은 아래와 같이 정의된다. 아래 식은 위 eigenvector들을 한꺼번에 표기한 것이라 보면 된다. 

$$A = Vdiag(\lambda)V^{-1}$$

![_config.yml]({{ site.baseurl }}/assets/ch2/Untitled.png)

- 모든 eigen value가 양수인 행렬을 positive definite matrix라고 한다.
- 모든 eigen value가 0 이상인 행렬을 positive semidefinite matrix라고 한다.

### 2.8 Singular value Decomposition(SVD)

- 또다른 방식의 matrix decomposition
- 행렬을 singular vector와 singular value로 분해한다.
- 좀 더 일반적인 행렬들에 적용 가능. 모든 실수 행렬에 적용 가능. square matrix일 필요도 읎음.
- 아래와 같이 어떤 행렬 A가 세 행렬의 곱으로 표현됨

$$A = UDV^T$$

- $A$가 m*n 행렬이면 ***U***는 m*m, ***D***는 m*n, ***V***는 n*n 행렬.
- ***U, V***는 둘 다 orthogonal matrix, ***D***는  diagonal matrix
- ***D***의 main diagonal 성분을 ***A***의 singular value라고 부른다.
- 특이값분해의 기하학적 의미
    - 행렬을 좌표공간에서의 선형변환으로 봤을때,
        - orthogonal matrix의 의미는 회전변환.
        - diagonal matrix의 의미는 작 좌표성분의 스케일 변환
    - $A = UDV^T$ 에서 U, V는 직교행렬, D는 대각행렬이므로 Ax는 x를 먼저 $V^T$에 의해 회전시킨 후 D로 스케일을 변화시키고 다시 U로 회전시키는 것임을 알 수 있다.

![_config.yml]({{ site.baseurl }}/assets/ch2/Untitled%201.png)

- Thin SVD, compact SVD, Truncated SVD 등도 있음. → 이미지 압축에 사용가능

![_config.yml]({{ site.baseurl }}/assets/ch2/Untitled%202.png)

Thin SVD

![_config.yml]({{ site.baseurl }}/assets/ch2/Untitled%203.png)

Truncated SVD

![_config.yml]({{ site.baseurl }}/assets/ch2/Untitled%204.png)

50개의 singular value로 근사한 이미지. 

### 2.9 무어-펜로즈 유사역행렬

- square matrix가 아닌 행렬은 역행렬이 없음.
- 역행렬 비스무리한 연산이 필요할 때 사용 가능.
- $A$의 유사역행렬 $A^+$는 아래와 같이 정의된다.

$$A^+ = lim_{\alpha\rightarrow0}(A^TA + \alpha I)^{-1}A^T$$

- 근데 보통 실제로 유사역행렬을 구할 때에는 고윳값 분해를 사용.

$$A^+=VD^+U^T$$

- U, D, V는 A의 고윳값 분해 결과임.

### 2.10 대각합 연산자(trace operator)

- 대각합 연산자 Tr는 행렬의 모든  main diagonal의 합을 계산한다.

$$Tr(A) = \sum_i{A_{i,i*}}$$

- 여러 수식을 더 단순하게 만듦. 예를 들어 프로베니우스 노름을 아래와 같이 표현 가능

$$||A||_F = \sqrt{Tr(AA^T)}$$

### 2.11 행렬식

- 행렬을 실수 스칼라로 사상하는 함수. Det(A).
- 행렬의 모든 eigen value를 곱한 값과 같다.
- 주어진 행렬을 곱했을 때 공간이 얼마나 확장 또는 축소되는지를 나타내는 측도.
- 행렬식이 0이면 공간은 적어도 하나의 차원에서 완전히 축소됨. (즉 해당 공간에서 부피 0)
- 행렬식이 1이면 공간의 부피가 안 변함.

### 2.12 예시: PCA

