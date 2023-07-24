import os
import io
from pathlib import Path
import requests

import PIL
from PIL import Image

import numpy as np

def is_dir(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            return True
    return False

def is_file(path):
    if os.path.exists(path):
        return True
    return False

def get_dirs_file(path, express='*/*'):
    dir_path = Path(path)
    generator = dir_path.glob(express)
    result = []
    for g in generator:
        result.append(str(g))
    return result

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        

def print_generator(g):
    for i in g:
        print(i, type(i))
        
def load_image(fname):
    with open(fname, mode='rb') as f:
      s = f.read()

    image = Image.open(io.BytesIO(s))
    image.load()
    return image
        
        

def resize_fg(fg, bg, fg_target_size):
    fg_copy = fg.copy()
    fg_area = fg.height * fg.width
    bg_area = bg.width * bg.height
    fg_area_ratio = fg_area / bg_area
    resize_factor = np.sqrt(fg_target_size / fg_area_ratio)
    fg_copy = fg_copy.resize(
        (int(fg.width * resize_factor), int(fg.height * resize_factor)),
        PIL.Image.BILINEAR)
    return fg_copy


def paste_fg_on_bg(fg, bg, x_coord, y_coord):
    bg_copy = bg.copy()
    bg_copy.paste(fg, box=(x_coord, y_coord), mask=fg)
    return bg_copy


def resize_bg(bg, tgt_width, tgt_height):
    return bg.resize((tgt_width, tgt_height), PIL.Image.BILINEAR)


def crop_image_to_square(img):
    side_length = min(img.height, img.width)
    return img.crop((0, 0, side_length, side_length))


def calc_top_left_coordinates(fg, bg, x_coord, y_coord):
    x_coord = int(x_coord * bg.width)
    y_coord = int(y_coord * bg.height)
    # x_coord, y_coord should be at the centre of the object.
    x_coord_start = int(x_coord - fg.width*0.5)
    y_coord_start = int(y_coord - fg.height*0.5)

    return x_coord_start, y_coord_start


def calc_pct_inside_image(fg, bg, x_coord_start, y_coord_start):
    x_coord_end = x_coord_start + fg.width
    y_coord_end = y_coord_start + fg.height

    x_obj_start = max(x_coord_start, 0)
    x_obj_end = min(x_coord_end, bg.width)
    y_obj_start = max(y_coord_start, 0)
    y_obj_end = min(y_coord_end, bg.height)

    object_area = fg.width * fg.height
    area_inside_image = (x_obj_end - x_obj_start) * (y_obj_end - y_obj_start)
    pct_inside_image = area_inside_image / object_area
    return pct_inside_image


def generate_instance_tuples(fgs_dict):
    num_class = 0
    class_and_instance_indices = []
    for key, val in fgs_dict.items():
        num_instances = len(val)
        class_and_instance_indices.extend([(num_class, j) for j in range(num_instances)])
        num_class = num_class + 1

    return class_and_instance_indices


def rotate_image(img, rotation_angle):
    return img.rotate(
        rotation_angle, resample=PIL.Image.BICUBIC, expand=True)


def write_backgrounds_csv(new_dataset_dir, backgrounds_dir):
    bg_filenames = get_dirs_file(backgrounds_dir, '*')
    bg_filenames = [fname.split('/')[-1] for fname in bg_filenames]
    csv_filepath = os.path.join(new_dataset_dir, 'backgrounds.csv')
    with open(csv_filepath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['int', 'label'])

        for i, fname in enumerate(bg_filenames):
            writer.writerow([i, fname])
 
def show_img(img):
    if isinstance(img, str):
        img = Image.open(img)
    # 展示图像
    img.show() 
    
def is_blank_pixels_row(data, row, cols):
    for i in range(cols):
        r, g, b, a = data[i, row]
        if a != 0:
            return True
    return False

def is_blank_pixels_col(data, col, rows):
    for i in range(rows):
        r, g, b, a = data[col, i]
        if a != 0:
            return True
    return False
  
  
  
def set_blank_pixels_transparent(image, mask, mask_id=0):
    image_copy = image.copy()
    image_copy = image_copy.convert("RGBA")

    # 获取图像的像素数据
    pixel_data = image_copy.load()

    # 设置空白像素为透明
    top = 0
    bottom = image_copy.height - 1
    left = 0
    right = image_copy.width - 1
    for y in range(image_copy.height):
        for x in range(image_copy.width):
            r, g, b, a = pixel_data[x, y]
            if mask[y][x] != mask_id:
                pixel_data[x, y] = (r, g, b, 0)
    height = image_copy.height
    width = image_copy.width
    for i in range(height):
        if is_blank_pixels_row(pixel_data, i, width):
            left = i
            break
    
    for i in range(height-1, 0, -1):
        if is_blank_pixels_row(pixel_data, i, width):
            right = i + 1
            break
    
    for i in range(width):
        if is_blank_pixels_col(pixel_data, i, height):
            top = i
            break
    
    for i in range(width-1, 0, -1):
        if is_blank_pixels_col(pixel_data, i, height):
            bottom = i + 1
            break
    
    
    # print(left, top, right, bottom)
    return image_copy.crop((top, left, bottom, right))


def segment_img(img, model='facebook/mask2former-swin-base-coco-panoptic'):
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    import torch
    processor = AutoImageProcessor.from_pretrained(model)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model)
    inputs = processor(img, return_tensors='pt')
    with torch.no_grad():
        output = model(**inputs)
    prediction = processor.post_process_panoptic_segmentation(output, target_sizes=[img.size[::-1]])[0]
    segment = prediction['segments_info'][0]
    # print(segment)
    label = model.config.id2label[segment['label_id']]
    # print(prediction.keys())
    return prediction['segmentation'], segment['id'], label
      
      
if __name__ == '__main__':
    # fgs_dict = {
    #     'row0': ['1' , '2', '3'],
    #     'row1': ['1' , '2', '3', '4'],
    #     'row2': ['1' , '2'],
    #     'row3': ['1'],
    #     'row4': ['1' , '2', '3'],
    # }
    # index = generate_instance_tuples(fgs_dict)
    # print(index)

 
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    mask, mask_id, label = segment_img(image)
    img_copy = set_blank_pixels_transparent(image, mask, mask_id)
    show_img(img_copy)
    show_img(image)
    print(label)