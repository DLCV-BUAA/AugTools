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

from augtools.img.transform import ImageTransform
from augtools.img.synthesis.utils import *
from augtools.utils.file_utils import LibraryUtil

DEFAULT_FG_IMGS = os.path.join(LibraryUtil.get_res_dir(), 'img', 'foreground')
DEFAULT_BG_IMGS = os.path.join(LibraryUtil.get_res_dir(), 'img', 'background')
DEFAULT_RS_IMGS = os.path.join(LibraryUtil.get_res_dir(), 'img', 'result')


class SynthesisTransform(ImageTransform):
    def __init__(self,
            config=None,
            new_dataset_dir=DEFAULT_RS_IMGS,
            num_bgs_per_fg_instance=2,
            min_pct_inside_image=0.95,
            num_thread=100,
            batch_size=128,
    ):
        super().__init__(always_apply=True, p=1)
        
        self.config = config
        self.new_dataset_dir = new_dataset_dir
        self._thread_pool = pool.ThreadPool(num_thread)
        
        
        self.config = self._validate_config(config)
        self.new_dataset_dir = new_dataset_dir
        self.num_bgs_per_fg_instance = num_bgs_per_fg_instance
        self.min_pct_inside_image = min_pct_inside_image
        self.bg_sizes = self.config['bg_resolution']  # width, height
        self.multiple_background_resolutions = False
        if len(self.bg_sizes) > 1:
            self.multiple_background_resolutions = True
        self.metadata_filepath = os.path.join(new_dataset_dir, 'metadata.csv')
        self.metadata_header = [
            'image_id', 'x_coord', 'y_coord', 'area', 'rotation',
            'foreground_class', 'foreground_instance', 'backgro und',
            'bg_resolution_width', 'bg_resolution_height', 'pct_inside_image'
        ]
        self._thread_pool = pool.ThreadPool(num_thread)
        self.batch_size = batch_size

        # write_backgrounds_csv(self.new_dataset_dir, backgrounds_dir)
        
        
    def _validate_config(self, config=None):
        if config is None:
            config = {
                'coords':[(0.5, 0.5)],
                'area':[0.5],
                'rotation':[0],
                'bg_resolution':[(1000,1000)]
            }
        if 'area' not in config.keys():
            config['area'] = [0.5]
        if 'coords' not in config.keys():
            config['coords'] = [0.5]
        if 'rotation' not in config.keys():
            config['rotation'] = [0]
        if 'bg_resolution' not in config.keys():
            config['bg_resolution'] = [(1000, 1000)]
        bg_res = config['bg_resolution']
        if not isinstance(bg_res, list) or not isinstance(bg_res[0], tuple) or len(
            bg_res[0]) != 2:
            config['bg_resolution'] = [(1000, 1000)]
        return config
    
    def _make_root_class_dirs(self):
        if not os.path.exists(self.new_dataset_dir):
            os.makedirs(self.new_dataset_dir, exist_ok=True)

        for class_name in self.fg_classes:
            class_dir = os.path.join(self.new_dataset_dir, class_name, '')
            if not os.path.exists(class_dir):
                os.makedirs(class_dir, exist_ok=True)
    
    def _write_metadata_header(self):
        with open(self.metadata_filepath, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.metadata_header)
    
    def __call__(self, fg_img=DEFAULT_FG_IMGS, bg_img=DEFAULT_BG_IMGS, mask=None, label=None):
        """
            1. validate config
                config:{
                    'coords':[0.5, 0.5],
                    'area':[0.5],
                    'rotation':[0],
                    'bg_resolution:(1000,1000)
                }
            2. load fore img
                2.1 load from img path and given label
                2.2 load from img dirs and get lable from dirs
                2.3 load from img and deep model split foreimg and give label
            3. load back img
                3.1 load from img path
                3.2 load from dirs
                3.3 load from img
            
            generate fore&back pair
            generate result list and metadata according config     
            
            4. process foreimg according metadata
            5. process backimg according metadata
            6. synthesis
            7. save
        """
        self.fg_img = fg_img
        self.bg_img = bg_img
        self.label = label
        self.mask = mask
        self._load_fg_imgs(self.fg_img) # 生成self.fgs and self.fgs_dict
        self._load_bg_imgs(self.bg_img) # 生成self.bgs
        if self.new_dataset_dir:
            self._make_root_class_dirs()
            self._write_metadata_header()
        fg_bgs = self._generate_fg_bg() # 生成self.fg_bgs => ((1, 2), 0) 第1类的第二张图作为前景， 第0张图作为背景
        rs_img_metas = self._generate_rs_metas(fg_bgs) # 根据配置生成一个个结果图像的元数据
        self._generate_img_and_save(rs_img_metas) # 根据元数据生成结果图像
        return 
    
    def _load_fg_imgs(self, fg_img):
        if isinstance(fg_img, str):
            if is_dir(fg_img):
                self._load_fg_imgs_from_dir(fg_img)
            elif is_file(fg_img) and self.mask and self.label:
                self._load_fg_imgs_from_file(fg_img)
            elif is_file(fg_img):
                self._load_fg_imgs_from_model(fg_img)
        else:
            if self.mask and self.label:
                self._load_fg_imgs_from_file(fg_img)
            else:
                self._load_fg_imgs_from_model(fg_img)
    
    def _load_fg_imgs_from_dir(self, fg_dir):
        if not os.path.exists(fg_dir):
            raise ValueError(
                f'Foregrounds directory {fg_dir} does not exist.')
        fg_fnames = get_dirs_file(fg_dir, '*/*')
        fg_labels = [x.split('/')[-2] for x in fg_fnames]  # e.g. 'car', 'cow'
        self.fg_classes = sorted(list(set(fg_labels)))
        self.fgs = self._thread_pool.map(load_image, fg_fnames)
        self.fgs_dict = {fg_class: [] for fg_class in self.fg_classes}
        for i, label in enumerate(fg_labels):
            self.fgs_dict[label].append(self.fgs[i])
        print('Foregrounds loaded.')
        
    def _load_fg_imgs_from_file(self, fg_img):
        if isinstance(fg_img, str):
            if not os.path.exists(fg_img):
                raise ValueError(
                    f'Foregrounds directory {fg_img} does not exist.')
            fg_img = load_image(fg_img)
            
        self.fg_classes = [self.label]
        self.fgs = [set_blank_pixels_transparent(fg_img, self.mask[0], self.mask[1])]
        self.fgs_dict = {self.label: [self.fgs[0]]}
        print('Foregrounds loaded.')
    
    def _load_fg_imgs_from_model(self, fg_img):
        if isinstance(fg_img, str):
            if not os.path.exists(fg_img):
                raise ValueError(
                    f'Foregrounds directory {fg_img} does not exist.')
            fg_img = load_image(fg_img)
        
        mask, mask_id, label = segment_img(fg_img)
        self.fg_classes = [label]
        self.fgs = [set_blank_pixels_transparent(fg_img, mask, mask_id)]
        self.fgs_dict = {label: [self.fgs[0]]}
        print('Foregrounds loaded.')
    
    def _load_bg_imgs(self, bg_img):
        if isinstance(bg_img, str):
            if is_dir(bg_img):
                self._load_bg_imgs_from_dir(bg_img)
            elif is_file(bg_img):
                self._load_bg_img_from_file(bg_img)
        else:
            self._load_bg_img_from_file(bg_img)
        
    def _load_bg_imgs_from_dir(self, bg_dir):
        if not os.path.exists(bg_dir):
            raise ValueError(
                f'Backgrounds directory {bg_dir} does not exist.')
        bg_fnames = get_dirs_file(bg_dir, '*')
        self.bgs = self._thread_pool.map(load_image, bg_fnames)
        self.bgs = self._thread_pool.map(self._preprocess_background, self.bgs)
        self.num_bgs = len(self.bgs)
        print('Backgrounds loaded.')
        
    def _load_bg_img_from_file(bg_img):
        self.bgs = []
        if is_file(bg_img):
            bg_img = load_image(bg_img)
        bg_img = self._preprocess_background(bg_img)
        self.num_bgs = 1
        self.bgs.append(bg_img)
        print('Backgrounds loaded.')
        
    def _preprocess_background(self, bg):
        bg = crop_image_to_square(bg)
        # If only one bg size is given [(width, height)]
        if not self.multiple_background_resolutions:
            if bg.width != self.bg_sizes[0][0] or bg.height != self.bg_sizes[0][1]:
                bg = bg.resize((self.bg_sizes[0][0], self.bg_sizes[0][1]),
                            PIL.Image.BILINEAR)
        return bg
    
    def _generate_rs_metas(self, fgs_bgs):

        config_lists = [self.config['coords'],
                        self.config['area'],
                        self.config['rotation'],
                        fgs_bgs,
                        self.config['bg_resolution']
                    ]

        image_metadata = itertools.product(*config_lists)

        self.num_images = functools.reduce(operator.mul, map(len, config_lists), 1)
        rs_img_metas = []
        for i, row in enumerate(image_metadata):
            temp_dict = {
                'id': i,
                'x_coord': row[0][0],
                'y_coord': row[0][1],
                'area': row[1],
                'rotation': row[2],
                'fg_class': row[3][0][0],
                'fg_instance': row[3][0][1],
                'bg_instance': row[3][1],
                'bg_resolution': row[4]
            }
            rs_img_metas.append(temp_dict)
        return rs_img_metas
            
    def _generate_fg_bg(self):
        fg_tuples = generate_instance_tuples(self.fgs_dict)
        fg_bg_tuples = []
        for fg_tuple in fg_tuples:
            bgs = np.random.choice(
                self.num_bgs, self.num_bgs_per_fg_instance, replace=False)
            fg_bg_tuples.extend([(fg_tuple, bg) for bg in bgs])
        return fg_bg_tuples
    
    def _generate_img_and_save(self, rs_img_metas):
        metadata_batch = []
        for i, single_image_metadata in enumerate(rs_img_metas):
            metadata_batch.append(single_image_metadata)

        if (i + 1) % self.batch_size == 0:
            self._generate_img_and_save_batch(metadata_batch)
            metadata_batch = []
        self._generate_img_and_save_batch(metadata_batch)
    
    def _generate_img_and_save_batch(self, metadata_batch):
        metadata_batch = self._thread_pool.map(self._generate_img_and_save_single,
                                           metadata_batch)
        self._write_batch_metadata(metadata_batch)
        
    def _generate_img_and_save_single(self, image_metadata):
        fg_class, fg_instance = image_metadata['fg_class'], image_metadata[
        'fg_instance']
        bg_num = image_metadata['bg_instance']
        x_coord, y_coord = image_metadata['x_coord'], image_metadata['y_coord']
        fg_tgt_area = image_metadata['area']
        rotation_angle = image_metadata['rotation']
        bg_resolution = image_metadata['bg_resolution']

        fg = self.fgs_dict[self.fg_classes[fg_class]][fg_instance]
        bg = self.bgs[bg_num]

        if self.multiple_background_resolutions:
            bg = resize_bg(bg, bg_resolution[0], bg_resolution[1])
        fg = resize_fg(fg, bg, fg_tgt_area)  # fg_target_size, uses background size
        fg = rotate_image(fg, rotation_angle)
        x_coord_start, y_coord_start = calc_top_left_coordinates(
            fg, bg, x_coord, y_coord)
        pct_inside_image = calc_pct_inside_image(fg, bg, x_coord_start,
                                                y_coord_start)
        if pct_inside_image < self.min_pct_inside_image:
            return None
        pct_inside_image = round(pct_inside_image, 4)
        image = paste_fg_on_bg(fg, bg, x_coord_start, y_coord_start)

        image_metadata.update(
            {'pct_inside_image': pct_inside_image})
        image_id = image_metadata['id']
        label = self.fg_classes[fg_class]
        file_path = '{}/{}/{}.jpg'.format(self.new_dataset_dir, label, image_id)
        with open(file_path, 'wb') as f:
            try:
                image.save(f)
            except TypeError:
                print('Failed to generate image num {}: fg: {} {}, bg: {}'.format(
                    image_id, fg_class, fg_instance, bg_num))
                print('Problem is likely that one of the foreground or background '
                    'images has an incompatible format. Loading and saving them as '
                    'JPG images using PIL may solve this problem.')

        return image_metadata
    
    def _write_batch_metadata(self, batch_metadata):
        with open(self.metadata_filepath, 'a') as f:
            writer = csv.writer(f)
            for row in batch_metadata:
                if row is not None:
                    csv_row = [row['id']]  # image ID, int
                    csv_row.extend([row['x_coord'], row['y_coord']])
                    csv_row.extend([row['area'], row['rotation']])  # area, rotation
                    csv_row.extend(
                        [row['fg_class'], row['fg_instance'], row['bg_instance']])
                    bg_resolution = row['bg_resolution']
                    csv_row.extend([bg_resolution[0], bg_resolution[1]])
                    csv_row.extend([row['pct_inside_image']])
                    writer.writerow(csv_row)
                    
    def _write_foreground_classes_csv(self):
        csv_filepath = path.join(self.new_dataset_dir,
                                'foreground_classes_metadata_indices.csv')
        with open(csv_filepath, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['int', 'label'])

            for i, label in enumerate(self.fg_classes):
                writer.writerow([i, label])
                
                
if __name__ == '__main__':
    synthesis = SynthesisTransform()
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    synthesis(fg_img=image)