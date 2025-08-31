import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint

# 재현성을 위한 시드 설정
seed = 8985
np.random.seed(seed)
tf.random.set_seed(seed)

# Keras에서 MNIST 데이터셋 직접 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 이진 분류를 위한 레이블 변경
# 숫자 7은 1, 그 외 숫자는 0으로 변환
y_train_binary = (y_train == 5).astype(np.float32) # True/False를 1.0/0.0으로 변환
y_test_binary = (y_test == 5).astype(np.float32)

"""
원본 레이블: y_train = [5, 7, 0, 7, 8]
불리언 변환: (y_train == 5) -> [True, False, False, False, False]
최종 이진 레이블: y_train_binary = [1.0, 0.0, 0.0, 0.0, 0.0]
"""

# 데이터 전처리
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# CNN 모델에 맞는 형태로 reshape (높이, 너비, 채널)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("훈련 데이터 형태:", x_train.shape)
print("이진 훈련 레이블 형태:", y_train_binary.shape)

# Functional API
input_tensor = Input(shape=(28, 28, 1))

x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
x = BatchNormalization()(x)
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# 최종 출력 레이어: 이진 분류를 위해 노드 1개, 활성화 함수 'sigmoid'
output_tensor = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_tensor, outputs=output_tensor)

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

# 콜백 함수 설정
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=30,
    restore_best_weights=True,
)

# 가장 좋은 모델을 저장하기 위한 ModelCheckpoint 콜백
checkpoint_path = './mnist_example/models/best_binary_discriminator.keras'
model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',       # 모니터링할 지표 (검증 손실)
    save_best_only=True,      # 가장 좋은 성능의 모델만 저장
    mode='min',               # val_loss가 최소일 때 저장
    verbose=1                 # 저장될 때 메시지 출력
)

# 모델 학습
batch_size = 64
epochs = 1000

history = model.fit(x_train, y_train_binary,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test_binary), # 그냥 val을 test로 쓰기로 함
                    verbose=1,
                    callbacks=[early_stopping, model_checkpoint]) # 조기 종료 콜백 추가

## 학습 결과 시각화
plt.figure(figsize=(13, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.grid()
plt.show()

plt.figure(figsize=(13, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train','Validation'])
plt.grid()
plt.show()

# print("모델 저장 중...")
# model.save('./mnist_example/models/discriminator.keras') # callback을 사용하지 않으므로 최적의 모델이 아닌, 종료된 시점에 model 객체가 갖고 있는 가중치와 구조가 저장됨

# loaded_model = tf.keras.models.load_model('./mnist_example/models/discriminator.keras')

# # 불러온 모델의 가중치와 구조에 접근 가능
# loaded_model.summary()