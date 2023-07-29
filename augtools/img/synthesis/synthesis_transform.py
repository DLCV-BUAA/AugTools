import os
from multiprocessing import pool
import itertools
import functools
import csv
import io
import operator

import PIL
from PIL import Image
import numpy as np
import random

from augtools.img.transform import ImageTransform
from augtools.img.synthesis.utils import *
from augtools.utils.file_utils import LibraryUtil

DEFAULT_FG_IMGS = os.path.join(LibraryUtil.get_res_dir(), 'img', 'foreground')
DEFAULT_BG_IMGS = os.path.join(LibraryUtil.get_res_dir(), 'img', 'background')
DEFAULT_RS_IMGS = os.path.join(LibraryUtil.get_res_dir(), 'img', 'result')


class SynthesisTransform(ImageTransform):
    def __init__(self,
                 bg_img=DEFAULT_BG_IMGS,
                 coords=(0.5, 0.5),
                 area=0.5,
                 rotation=0,
                 bg_sizes=(500, 500),
                 ):
        super().__init__(always_apply=True, p=1)

        self.bg_img = bg_img
        self.bg_sizes = bg_sizes
        self.x_coord, self.y_coord = coords[0], coords[1]
        self.area = area
        self.rotation = rotation
        self._load_bg_img()

    def _compute_x_function(self, x, rs=None):
        # load fg_img
        self._load_fg_imgs_from_model(x)
        # load bg_img
        return self._generate_img()

    def _load_fg_imgs_from_model(self, fg_img):
        if isinstance(fg_img, np.ndarray):
            fg_img = Image.fromarray(fg_img)
        mask, mask_id, label = segment_img(fg_img)
        self.fg_img = set_blank_pixels_transparent(fg_img, mask, mask_id)

    def _load_bg_img(self):
        if is_dir(self.bg_img):
            if not os.path.exists(self.bg_img):
                raise ValueError(
                    f'Backgrounds directory {self.bg_img} does not exist.')
            bg_fnames = get_dirs_file(self.bg_img, '*')
            bg_fname = random.choice(bg_fnames)
            bg_img = load_image(bg_fname)
            self.bg_img = self._preprocess_background(bg_img)
        elif is_file(self.bg_img):
            bg_img = load_image(self.bg_img)
            self.bg_img = self._preprocess_background(bg_img)
        else:
            self.bg_img = self._preprocess_background(self.bg_img)

    def _preprocess_background(self, bg):
        bg = crop_image_to_square(bg)
        return bg

    def _generate_img(self):
        fg = self.fg_img
        bg = self.bg_img

        bg = resize_bg(bg, self.bg_sizes[0], self.bg_sizes[1])
        fg = resize_fg(fg, bg, self.area)
        fg = rotate_image(fg, self.rotation)
        x_coord_start, y_coord_start = calc_top_left_coordinates(
            fg, bg, self.x_coord, self.y_coord)
        image = paste_fg_on_bg(fg, bg, x_coord_start, y_coord_start)
        if isinstance(image, Image.Image):
            image = np.array(image)
        return image


if __name__ == '__main__':
    from augtools.utils.test_utils import *
    synthesis = SynthesisTransform(rotation=0)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image = synthesis(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Image', image)
    # 等待按下任意键
    cv2.waitKey(0)
    # 关闭窗口
    cv2.destroyAllWindows()
