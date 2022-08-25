# 학습시킨 모델을 이용하여 test data 예측 후 결과 txt로 저장

# local 찌:2 바위:1 보:0
# spec 0(주먹) 1(가위) 2(보)

from keras.models import load_model
import tensorflow as tf

from skimage.io import imread
from skimage.transform import resize

import sys
import os

# 모델 load
model = load_model('/Users/nam/Downloads/rps_60160223.h5')

# image directory 경로 받아 오기
test_dir = str(sys.argv[1])
# test image 파일 이름 받아오기
test_img = os.listdir(test_dir)
# 해당 이미지 폴더에 있는 파일명 출력
# print(test_img)

# output.txt를 저장할 위치 지정
f = open('/Users/nam/Downloads/rps_train_test/test/output.txt', 'w')

f.write("output:\n\n")

for i in test_img:
  try:
    img = imread(f'{test_dir}/{i}') #이미지 파일 open
    img = resize(img, output_shape = (150, 150, ))
    img = img[:,:,:3]

    # 이미지 shpae이 150 150 4인 것들이 있어서 다시 150 150 3으로 바꾸는 작업
    img = tf.reshape(img, shape = [1, 150, 150, 3])
    target = list(model.predict(img)[0])
    target = target.index(max(target))

    if target == 0:
      ans = 1
    elif target == 1:
      ans = 2
    else:
      ans = 0

    f.write(f'{i} {ans}\n')   
  except:
    print(i)

f.close()