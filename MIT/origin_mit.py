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
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(f"모델 구조 요약: {model.summary()}")