import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, Callback

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
    
    return Model(input_layer, output_layer)

def make_attack_model(generator_input_dim=100, discriminator_path='./mnist_example/models/best_binary_discriminator.keras'):
    
    # 2. Discriminator (Classifier) 로드 및 가중치 고정
    # 이전에 학습시킨 이진 분류 모델을 로드
    classifier = load_model(discriminator_path)
    # 판별자는 훈련시키지 않고 가중치를 고정
    classifier.trainable = False

    # 3. Adversarial Model (Generator + Discriminator) 생성
    gen = generator(input_dim=generator_input_dim)
    generator_input = Input(shape=(generator_input_dim,))
    # Generator 이미지를 생성
    generated_image = gen(generator_input)
    # 생성된 이미지를 Discriminator에 입력하여 판별 결과를 얻음
    classifier_output = classifier(generated_image)

    # 전체 Adversarial 모델을 정의
    # 입력: 난수 벡터, 출력: Discriminator의 결과
    attack_model = Model(generator_input, classifier_output)

    # 4. Adversarial Model 컴파일
    # 이 모델은 Generator를 훈련시키는 데 사용
    # Generator의 목표는 Discriminator가 생성된 이미지를 '5' (레이블 1)로 인식하게 하는 것
    # 따라서 이진 분류 손실 함수와 함께 실제 레이블을 1로 가정하여 훈련
    optimizer = Adam(learning_rate=0.0001, beta_1=0.5) # GAN 학습에 최적화된 하이퍼파라미터
    attack_model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return attack_model, gen, classifier

# 50 에포크마다 이미지를 그리고 예측값을 출력하는 커스텀 콜백
class ImageGeneratorCallback(Callback):
    def __init__(self, generator_model, classifier_model, num_images=10):
        super(ImageGeneratorCallback, self).__init__()
        self.generator = generator_model
        self.classifier = classifier_model
        self.num_images = num_images
        self.test_noise = np.random.normal(0, 1, (self.num_images, 100))
        self.output_dir = './generated_images'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.generator.predict(self.test_noise)
        # 생성된 이미지를 classifier로 예측
        predictions = self.classifier.predict(generated_images, verbose=0) # verbose=0으로 설정하여 callback 함수 내 예측 시 출력은 억제하여, 터미널에 진행바가 fit 함수와 겹쳐서 출력되는 것을 방지
        current_loss = logs.get('loss')
        print(f"\nEpoch {epoch+1} - Loss: {current_loss:.4f}") # on_epoch_end는 하나의 에포크가 완전히 끝난 후에 호출
                                                                # 이 시점에 logs 딕셔너리에 포함된 loss 값은 그 에포크의 전체 평균 손실을 나타냄
                                                                # 따라서 이 값은 한 에포크 동안의 모든 배치의 손실을 평균 낸 최종적인 값
        for i in range(self.num_images):
            prediction_val = predictions[i][0]
            print(f"  Image {i+1}: Pred={prediction_val:.4f}")

        if (epoch + 1) % 50 == 0:
            plt.figure(figsize=(15, 2))
            for i in range(self.num_images):
                plt.subplot(1, self.num_images, i + 1)
                plt.imshow(generated_images[i, :, :, 0], cmap='gray')
                # 예측값 출력: 소수점 4자리까지 표시
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
    loaded_classifier = load_model(classifier_path)
    classifier = Model(
        inputs=loaded_classifier.input,
        outputs=loaded_classifier.output,
        name='unique_classifier'  # Give it a unique name
    )
    classifier.set_weights(loaded_classifier.get_weights())
    classifier.trainable = False

    gen = generator(input_dim=generator_input_dim)
    
    gen_input = Input(shape=(generator_input_dim,), name='adversarial_input')
    gen_output = gen(gen_input)
    discriminator_output = classifier(gen_output)
    
    adversarial_model = Model(gen_input, discriminator_output, name='adversarial_model')
    
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    adversarial_model.compile(loss='binary_crossentropy', optimizer=optimizer)

    adversarial_model.summary(expand_nested=True)

    print("\n생성자(Generator) 훈련을 시작합니다...")
    noise = np.random.normal(0, 1, (epochs * 100, generator_input_dim))
    target_labels = np.ones((epochs * 100, 1))
    
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=10,
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

