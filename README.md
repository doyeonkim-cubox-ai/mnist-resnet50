# mnist-resnet50
### Introduction

pytorch 실습 이후 실제로 사용해보고자 간단한 이미지 분류 모델을 구현했다.

모델 학습에 사용한 데이터는 MNIST dataset, 모델은 pytorch에서 제공하고 있는 resnet50을 활용하였다.

### Files
- model.py
  - Mymodel 정의
- train.py
  - train data 다운로드 및 모델 학습
- test.py
  - test data를 이용한 모델의 정확도 계산

### Run
```
pip install -r requirements.txt
python -m mnist_resnet50.train
python -m mnist_resnet50.test
```

### Inference.py
- change line18 to use your own data
```
image = Image.open("{PATH_TO_IMAGE}/{IMAGE_NAME}.{png/jpg/..}")
```
- run
```
python inference.py
```
해당 이미지에 대한 분류 결과가 출력될 것이다.