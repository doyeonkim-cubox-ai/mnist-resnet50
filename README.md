# mnist-resnet50
간단한 이미지(숫자) 분류 모델입니다.

### Introduction

pytorch 스터디 이후 실제로 사용해보고자 간단한 이미지 분류 모델을 구현했습니다.
모델은 pytorch에서 제공하고 있는 resnet50, 학습 데이터셋은 mnist를 사용했습니다.
pytorch lightning으로 구현한 버전은 파일명 앞에 L을 추가했습니다.

### Requirements
After cloning the repo, run this line below:
```
pip install -r requirements.txt
```

### Usage

##### 1. train & test model
```
mkdir model
python -m mnist_resnet50.Ltrain
python -m mnist_resnet50.Ltest
```
##### 2. inference
```
python -m mnist_resnet50.inference --img ${IMG_PATH}
```
해당 이미지에 대한 분류 결과가 출력될 것입니다.

##### 3. custom dataset
```
from mnist_resnet50 import dataset
```
