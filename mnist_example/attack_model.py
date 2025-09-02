import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
import datetime

def generator(input_dim):
    input_layer = Input(shape=(input_dim,))

    x = Dense(7 * 7 * 128)(input_layer)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((7, 7, 128))(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    output_layer = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='sigmoid')(x)
    
    return Model(inputs=input_layer, outputs=output_layer, name='generator')

def make_attack_model(generator_input_dim=100, discriminator_path='./mnist_example/models/best_binary_discriminator.keras'):
    # 1. Discriminator (Classifier) 로드 및 가중치 고정
    classifier = load_model(discriminator_path)
    classifier.trainable = False

    # 2. Generator 생성
    gen = generator(input_dim=generator_input_dim)
    
    # 3. Adversarial Model (Generator + Discriminator) 생성
    gen_input = Input(shape=(generator_input_dim,), name='adversarial_input')
    gen_output = gen(gen_input)
    discriminator_output = classifier(gen_output)
    
    # 최종 공격 모델 정의 (두 개의 출력을 가짐)
    # 첫 번째 출력은 이진 분류 결과, 두 번째 출력은 생성된 이미지 자체
    adversarial_model = Model(gen_input, [discriminator_output, gen_output], name='adversarial_model')
    
    # 4. Adversarial Model 컴파일
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    
    # BCE와 MSE 손실 함수를 결합. MSE에 가중치(0.01)를 부여
    adversarial_model.compile(optimizer=optimizer, loss=[BinaryCrossentropy(), MeanSquaredError()], loss_weights=[1.0,1.5])

    return adversarial_model, gen, classifier

class ImageGeneratorCallback(Callback):
    def __init__(self, generator_model, classifier_model, num_images=10):
        super(ImageGeneratorCallback, self).__init__()
        self.generator = generator_model
        self.classifier = classifier_model
        self.num_images = num_images
        self.test_noise = np.random.normal(0, 1, (self.num_images, 100))
        
        # 1. 현재 실행 시간을 가져와 '월일시분' 형식의 문자열로 만듭니다. (8월 31일 22시 08분 = 08312208)
        current_time = datetime.datetime.now().strftime("%m%d%H%M")
        
        # 2. 이 시간을 기반으로 출력 디렉터리 경로를 생성합니다.
        self.base_output_dir = './generated_images'
        self.output_dir = os.path.join(self.base_output_dir, current_time)
        
        # 3. 디렉터리가 없으면 생성합니다.
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"이미지 저장 디렉터리 생성: {self.output_dir}")

    def on_epoch_end(self, epoch, logs=None):
        # 1. 생성자 모델을 사용하여 이미지 생성
        generated_images = self.generator.predict(self.test_noise, verbose=0)
        
        # 2. 생성된 이미지를 classifier로 예측
        predictions = self.classifier.predict(generated_images, verbose=0)
        
        current_loss = logs.get('loss')
        print(f"\nEpoch {epoch+1} - Loss: {current_loss:.4f}")
        
        # 3. 예측값 출력
        for i in range(self.num_images):
            prediction_val = predictions[i][0]
            print(f"  Image {i+1}: Pred={prediction_val:.4f}")

        # 4. 10 에포크마다 이미지 시각화 및 저장
        if (epoch + 1) % 10 == 0:
            plt.figure(figsize=(15, 2))
            for i in range(self.num_images):
                plt.subplot(1, self.num_images, i + 1)
                plt.imshow(generated_images[i, :, :, 0], cmap='gray')
                
                prediction_val = predictions[i][0]
                title = f'Pred: {prediction_val:.4f}'
                plt.title(title, fontsize=10)
                plt.axis('off')
            
            plt.suptitle(f"Epoch {epoch + 1} Generated Images", fontsize=16)
            
            plt.savefig(os.path.join(self.output_dir, f"epoch_{epoch + 1}.png"))
            plt.show()
            plt.close()

# 적대적 모델을 생성하고 훈련시키는 함수
def train_adversarial_model(epochs=1000, batch_size=64, generator_input_dim=100, classifier_path='./mnist_example/models/best_binary_discriminator.keras'):
    # make_attack_model 함수를 호출하여 모델, Generator, Classifier를 가져옴
    adversarial_model, gen, classifier = make_attack_model(generator_input_dim, classifier_path)
    
    adversarial_model.summary(expand_nested=True)

    print("\n생성자(Generator) 훈련을 시작합니다...")
    
    # MNIST 데이터셋을 사용하여 평균 '7' 이미지 생성
    average_image = np.load('./cache/average_image_8.npy')

    noise = np.random.normal(0, 1, (epochs * 100, generator_input_dim))
    target_labels_bce = np.ones((epochs * 100, 1))
    target_labels_mse = np.tile(average_image, (epochs * 100, 1, 1, 1))
    target_labels = [target_labels_bce, target_labels_mse]
    
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=30,
        restore_best_weights=True,
    )
    image_callback = ImageGeneratorCallback(gen, classifier)

    history = adversarial_model.fit(
        noise,
        target_labels,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, image_callback]
    )

    print("\n훈련이 완료되었습니다.")
    
if __name__ == '__main__':
    train_adversarial_model()