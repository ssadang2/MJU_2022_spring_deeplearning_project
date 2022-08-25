# 써야 하는 모듈을 import하고 이미지를 보여주는 함수(show)와 이미지를 비교하는 함수(compare) 정의

from skimage.io import imread
from skimage.transform import resize
from PIL import Image
from PIL import ImageFile 
import imageio

import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt

from keras.utils import to_categorical, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import os
import splitfolders

def show(image : np.ndarray, title = "Image", cmap_type = "gray", axis = False):
  plt.imshow(image, cmap = cmap_type)
  plt.title(title)

  if not axis:
    plt.axis("off")

  plt.margins(0, 0)
  plt.show()

def compare(
original,
filtered,
title_filtered="Filtered",
cmap_type="gray",
axis = False,
title_original = "Original"
) :
    fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (10, 8), sharex = True, sharey = True)

    ax1.imshow(original, cmap  = cmap_type)
    ax1.set_title(title_original)

    ax2.imshow(original, cmap  = cmap_type)
    ax2.set_title(title_original)

    if not axis:
      ax1.axis("off")
      ax2.axis("off")

    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0.01)
    plt.margins(0, 0)
    plt.show()

# train test data split

# splitfolders 라이브러리가 train, test, val으로 분할을 기본적으로 
# 염두에 두고 있기 때문에 val = 0.0을 줘도 val 파일이 생기기 때문에 train test 디렉토리에 가서 val 파일 삭제해줘야 됨
# train test를 8:2로 나눔
splitfolders.ratio('/Users/nam/Downloads/rps', output="/Users/nam/Downloads/rps_train_test", seed=1337, ratio=(.8, .0, .2))

# imagedatagenerator에 필요한 dir 경로 표시 및 저장

train_dir = os.path.abspath("/Users/nam/Downloads/rps_train_test/train")
test_dir = os.path.abspath("/Users/nam/Downloads/rps_train_test/test")

# test output 출력 시 필요한 dir 정보
paper_dir = os.path.abspath("/Users/nam/Downloads/rps_train_test/test/paper_summary")
rock_dir = os.path.abspath("/Users/nam/Downloads/rps_train_test/test/rock_summary")
scissors_dir = os.path.abspath("/Users/nam/Downloads/rps_train_test/test/scissors_summary")

paper_img = os.listdir(paper_dir)
rock_img = os.listdir(rock_dir)
scissors_img = os.listdir(scissors_dir)

print(type(paper_img))
print(len(paper_img))
print(paper_img[0])

# sample datas show

plt.figure(figsize=(20, 4))
for i, img_path in enumerate(paper_img[:5]):
    sp = plt.subplot(1, 5, i+1)
    img = imread(os.path.join(paper_dir, img_path))
    plt.imshow(img)
plt.show()

# 모델 최적화를 위한 케라스 콜백 함수들 정의 => 오히려 성능 떨어져서 사용 X

# reduce = ReduceLROnPlateau(monitor='val_accuracy',
#                           patience=2,
#                           verbose=1,
#                           factor=0.5,
#                           min_lr=0.000003
# )

# earlystop = EarlyStopping(
#     monitor='val_loss',
#     patience=10,
#     verbose=1
# )

# callbacks = [reduce, earlystop]

# split된 디렉토리를 기준으로 ImageDataGenerator 모듈을 사용하여 train data, test data generate
# imagedatagenerator 모듈에 여러가지 Data augmentation 기능을 지원하나 안 쓰는 게 generalization error가 더 낮음

train_datagen = ImageDataGenerator(rescale = 1./255                           
                                  #  rotation_range=50,
                                  #  width_shift_range=0.3,
                                  #  height_shift_range=0.3,
                                  #  brightness_range=(0.0, 0.9),
                                  #  shear_range=0.5,
                                  #  zoom_range=0.5,
                                  #  horizontal_flip=True,
                                  #  vertical_flip=True,
          
                                  #  horizontal_flip=True,
                                  #  height_shift_range=.2,
                                  #  vertical_flip = True,
                                  #  validation_split = 0.2
                                   )

test_datagen = ImageDataGenerator(rescale=1./255
                                  )

BATCH_SIZE = 32
TARGET_SIZE = 150
EPOCHS = 20

train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size=(TARGET_SIZE, TARGET_SIZE),
                                                    batch_size=BATCH_SIZE, 
                                                    shuffle=True,
                                                    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                       target_size=(TARGET_SIZE, TARGET_SIZE), 
                                                       batch_size=BATCH_SIZE, 
                                                       shuffle=True,
                                                       class_mode='categorical')

# CNN 모델 학습

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 모델 컴파일

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 summary

model.summary()

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    verbose=1
    # callbacks = callbacks
    )

# 결과 그래프 화

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig('/Users/nam/Downloads/rps_train_test/Result.png')
plt.show()

# 모델 저장

model.save("//Users/nam/Downloads/rps_train_test/rps.h5")