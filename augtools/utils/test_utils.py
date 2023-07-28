import cv2
import copy
import numpy as np

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


def show_bbox_keypoint_image(image, bbox, keypoint):
    image = copy.deepcopy(image)
    for item in bbox:
        x_min, y_min, x_max, y_max = int(item[0]), int(item[1]), int(item[2]), int(item[3])
        # image[x_min - 1:x_min + 1, y_min:y_max, 0] = 0
        # image[x_min - 1:x_min + 1, y_min:y_max, 1] = 255
        # image[x_min - 1:x_min + 1, y_min:y_max, 2] = 0
        #
        # image[x_max - 1:x_max + 1, y_min:y_max, 0] = 0
        # image[x_max - 1:x_max + 1, y_min:y_max, 1] = 255
        # image[x_max - 1:x_max + 1, y_min:y_max, 2] = 0
        #
        # image[x_min:x_max, y_min - 1: y_min + 1, 0] = 0
        # image[x_min:x_max, y_min - 1: y_min + 1, 1] = 255
        # image[x_min:x_max, y_min - 1: y_min + 1, 2] = 0
        #
        # image[x_min:x_max, y_max - 1: y_max + 1, 0] = 0
        # image[x_min:x_max, y_max - 1: y_max + 1, 1] = 255
        # image[x_min:x_max, y_max - 1: y_max + 1, 2] = 0
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0))

    for item in keypoint:
        x, y = item[0], item[1]
        cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_bbox_keypoint_image_float(image, bbox=None, keypoint=None):
    image = copy.deepcopy(image)
    image = np.ascontiguousarray(image) # Transpose 不连续
    h, w, _ = image.shape

    if bbox is not None:
        for item in bbox:
            x_min, y_min, x_max, y_max = int(item[0]*w), int(item[1]*h), int(item[2]*w), int(item[3]*h)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0))

    if keypoint is not None:
        for item in keypoint:
            x, y = int(item[0]), int(item[1])
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    filepath = r'F:/dog.jpg'
    img = read_image(filepath)
    show_image(img)
