from torchvision.datasets import ImageFolder
from matplotlib import pyplot as plt
import cv2


def read_image(filepath):
    # print(filepath)
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def show_image(image):
    # plt.figure(figsize=(10, 10))
    # plt.axis('off')
    # plt.imshow(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Image', image)
    # 等待按下任意键
    cv2.waitKey(0)
    # 关闭窗口
    cv2.destroyAllWindows()
    # plt.imshow(image)


def show_image_by_tensor(tensor_image):
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    # print(tensor_image.shape)
    to_pil = transforms.ToPILImage()
    tensor_image = to_pil(tensor_image)
    # print(tensor_image.shape)
    # image = Image.fromarray(tensor_image)
    # 展示图片
    # tensor_image.show()

    image = np.array(tensor_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Image', image)
    # 等待按下任意键
    cv2.waitKey(0)


if __name__ == '__main__':
    filepath = r'F:/dog.jpg'
    img = read_image(filepath)
    show_image(img)
