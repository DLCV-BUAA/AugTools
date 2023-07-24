import os
import io
from pathlib import Path

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
  """Resizes foregrounds to `fg_target_size`% of the background area."""
  # Resize foreground to have area = fg_size**2 * background_area.
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
  """Pastes foreground on background at offset (x_coord, y_coord).

  x_coord, y_coord are floats in range [0, 1].

  Args:
    fg: foreground image of type PIL image. Examples of PIL image types include
      PIL.PngImagePlugin.PngImageFile and PIL.JpegImagePlugin.JpegImageFile.
    bg: background image of type PIL image.
    x_coord: float in range [0, 1]. x-coord offset from top left, for pasting
    foreground.
    y_coord: float in range [0, 1]. y-coord offset from top left, for pasting
    foreground.

  Returns:
    Background: PIL image (e.g. type PIL.JpegImagePlugin.JpegImageFile).
  """
  bg_copy = bg.copy()
  bg_copy.paste(fg, box=(x_coord, y_coord), mask=fg)
  return bg_copy


def resize_bg(bg, tgt_width, tgt_height):
  """Resizes bg to width = tgt_width, height = tgt_height."""
  return bg.resize((tgt_width, tgt_height), PIL.Image.BILINEAR)


def crop_image_to_square(img):
  """Crops image to the largest square that fits inside img.

  Crops from the top left corner.

  Args:
    img: image of type PIL image, e.g. PIL.JpegImagePlugin.JpegImageFile.

  Returns:
    Square image of same type as input image.
  """
  side_length = min(img.height, img.width)
  return img.crop((0, 0, side_length, side_length))


def calc_top_left_coordinates(fg, bg, x_coord, y_coord):
  """Returns coordinates of top left corner of object.

  Input coordinates are coordinates of centre of object scaled in the range
  [0, 1].

  Args:
    fg: PIL image. Foreground image.
    bg: PIL image. Background image.
    x_coord: central x-coordinate of foreground object scaled between 0 and 1.
      0 = leftmost coordinate of image, 1 = rightmost coordinate of image.
    y_coord: central y-coordinate of foreground object scaled between 0 and 1.
      0 = topmost coordinate of image, 1 = bottommost coordinate of image.
  """
  x_coord = int(x_coord * bg.width)
  y_coord = int(y_coord * bg.height)
  # x_coord, y_coord should be at the centre of the object.
  x_coord_start = int(x_coord - fg.width*0.5)
  y_coord_start = int(y_coord - fg.height*0.5)

  return x_coord_start, y_coord_start


def calc_pct_inside_image(fg, bg, x_coord_start, y_coord_start):
  """Calculate the percentage of the object that is inside the image.

  This calculation is based on the bounding box of the object
  as opposed to object pixels.

  Args:
    fg: PIL image. Foreground image.
    bg: PIL image. Background image.
    x_coord_start: leftmost x-coordinate of foreground object.
    y_coord_start: topmost y-coordinate of foreground object.

  Returns:
    Float between 0.0 and 1.0 inclusive, indicating the percentage of the
    object that is in the image.
  """
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
  """Generate list of tuples [(class_index, instance_index)...]."""
  num_class = 0
  class_and_instance_indices = []
  for key, val in fgs_dict.items():
      num_instances = len(val)
      class_and_instance_indices.extend([(num_class, j) for j in range(num_instances)])
      num_class = num_class + 1

  return class_and_instance_indices


def rotate_image(img, rotation_angle):
  """Rotate image by rotation_angle counterclockwise."""
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
      
      
if __name__ == '__main__':
    fgs_dict = {
        'row0': ['1' , '2', '3'],
        'row1': ['1' , '2', '3', '4'],
        'row2': ['1' , '2'],
        'row3': ['1'],
        'row4': ['1' , '2', '3'],
    }
    index = generate_instance_tuples(fgs_dict)
    print(index)