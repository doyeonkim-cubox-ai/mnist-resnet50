# mnist-resnet50
간단한 이미지(숫자) 분류 모델입니다.

### Introduction

pytorch 스터디 이후 실제로 사용해보고자 간단한 이미지 분류 모델을 구현했습니다.
모델은 pytorch에서 제공하고 있는 resnet50, 학습 데이터셋은 mnist를 사용했습니다.

### Requirements
Run this line below:
```
pip install -r requirements.txt
```

### Usage

##### 1. Clone the repo
```
git clone https://github.com/doyeonkim-cubox-ai/mnist-resnet50.git
```
##### 2. train & test model
```
mkdir model
python -m mnist_resnet50.train
python -m mnist_resnet50.test
```
##### 3.  inference
```
# change line18 to use your own data
image = Image.open("{PATH_TO_IMAGE}/{IMAGE_NAME}.{png/jpg/..}")
# run inference.py
python inference.py
```
해당 이미지에 대한 분류 결과가 출력될 것입니다.


