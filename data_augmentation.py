import numpy as np
import os
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


def argument(image_path, nb_new_images=10, save_to_dir='preview',
             save_prefix='img', save_format='JPG'):
    """图像增强"""
    if not os.path.exists(save_to_dir):
        os.makedirs(save_to_dir)
    # print('argument %s ...' % image_path)
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    img = load_img(image_path)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for _ in datagen.flow(x, batch_size=1, save_to_dir=save_to_dir,
                          save_prefix=save_prefix, save_format=save_format):
        i += 1
        if i > nb_new_images:
            break


if __name__ == '__main__':
    root_dir = './data'
    file_filter = lambda x: False if x.startswith('.') else True
    for dir_name in os.listdir(root_dir):
        if not file_filter(dir_name):
            continue
        dir_path = os.path.join(root_dir, dir_name)
        files = list(filter(file_filter, os.listdir(dir_path)))
        np.random.seed(0)
        chosen_files = np.random.choice(files, 4)
        for img_name in tqdm(chosen_files):
            img_path = os.path.join(dir_path, img_name)
            argument(img_path, nb_new_images=3,
                     save_to_dir=os.path.join('argumentation', dir_name))
