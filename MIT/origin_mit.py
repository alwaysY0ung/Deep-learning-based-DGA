# MIT model with a stacked CNN and LSTM layer
# originally implemented in TensorFlow/Keras

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.models import Model

# 모델 구조 정의
def build_model():
    # Listing 5: MIT model with a stacked CNN and LSTM layer
    main_input = Input(shape=(75,), dtype='int32', name='main_input')
    embedding = Embedding(input_dim=128, output_dim=128, input_length=75)(main_input)
    conv = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu', strides=1)(embedding)
    max_pool = MaxPooling1D(pool_size=2, padding='same')(conv)
    encode = LSTM(64, return_sequences=False)(max_pool)
    output = Dense(1, activation='sigmoid')(encode)
    
    model = Model(inputs=main_input, outputs=output)
    
    return model

if __name__ == '__main__':
    # 모델 빌드
    model = build_model()
    
    # 모델 컴파일
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # 모델 구조 요약
    print("모델 구조 요약:")
    model.summary()
    
    # # 더미 데이터 생성 (훈련 및 검증을 위한 예시)
    # # 2000개의 샘플, 각 샘플은 75개의 정수로 구성
    # num_samples = 2000
    # sequence_length = 75
    # vocab_size = 128
    
    # # 입력 데이터 (0부터 127 사이의 랜덤 정수)
    # X_train = np.random.randint(0, vocab_size, (num_samples, sequence_length))
    # # 레이블 데이터 (0 또는 1)
    # y_train = np.random.randint(0, 2, num_samples)
    
    # print("\n훈련 데이터 형태:")
    # print("X_train.shape:", X_train.shape)
    # print("y_train.shape:", y_train.shape)
    
    # # 모델 훈련 (더미 데이터로 5 에포크 학습)
    # print("\n모델 훈련 시작...")
    # model.fit(
    #     X_train, 
    #     y_train, 
    #     epochs=5, 
    #     batch_size=32, 
    #     validation_split=0.2  # 훈련 데이터의 20%를 검증에 사용
    # )
    
    # # 모델 평가 (성능 확인)
    # loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
    # print(f"\n모델 최종 성능: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")