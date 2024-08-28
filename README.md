### Introduction

pytorch 실습 이후 실제로 사용해보고자 간단한 이미지 분류 모델을 구현했다.

모델 학습에 사용한 데이터는 MNIST, 모델은 pytorch에서 제공하고 있는 resnet50을 활용하였다.

### Files
- train.py
    - train data 다운로드 및 모델 학습
- test.py
    - test data를 이용한 모델의 정확도 계산

### Run
```
pip install -r requirements.txt
python train.py
python test.py
```
### Detail
- Hyperparameter
    - learning rate: (0.01, 0.005, 0.0025, 0.002, 0.001)
    - batch size: 32
    - epochs: (20, 30, 40)
- Optimization
    - Adam
- Cost computation
    - Cross entropy loss