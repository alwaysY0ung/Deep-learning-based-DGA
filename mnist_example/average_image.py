import numpy as np
from keras.datasets import mnist

def create_average_image(target_digit=5, img_shape=(28, 28)): # target_digit (int): 평균을 낼 목표 숫자 (0-9)
    """Returns:
        np.ndarray: 평균 이미지 배열."""
    print("target image digit: ", target_digit)
    (x_train, y_train), _ = mnist.load_data()
    target_images = x_train[y_train == target_digit]
    target_images = target_images.astype('float32') / 255.0
    average_image = np.mean(target_images, axis=0)
    average_image = np.expand_dims(average_image, axis=-1) # 차원 추가하여 (28, 28, 1) 형태로 변환
    
    return average_image

if __name__ == "__main__":
    avg_image = create_average_image(target_digit=6)
    print("Average image shape:", avg_image.shape)
    # print("Average image array:\n", avg_image)
    np.save("./cache/average_image_6.npy", avg_image) # Stored in cache directory ^.^