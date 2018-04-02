import numpy as np
import os
import time
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from multiprocessing import Pool


def argument(image_path, nb_new_images=10, save_to_dir='preview',
             save_prefix='img', save_format='JPG'):
    """图像增强"""
    os.makedirs(save_to_dir, exist_ok=True)
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
        if i >= nb_new_images:
            break


if __name__ == '__main__':
    t0 = time.time()
    root_dir = './data'
    file_filter = lambda x: False if x.startswith('.') else True
    process_pool = Pool()
    for dir_name in os.listdir(root_dir):
        if not file_filter(dir_name):
            continue
        dir_path = os.path.join(root_dir, dir_name)
        files = list(filter(file_filter, os.listdir(dir_path)))
        np.random.seed(None)
        chosen_files = np.random.choice(files, 6)
        for idx, img_name in enumerate(chosen_files):
            img_path = os.path.join(dir_path, img_name)
            process_pool.apply_async(argument, (img_path,),
                                     kwds={'nb_new_images': 6,
                                           'save_to_dir': os.path.join('augmentation', dir_name),
                                           'save_prefix': 'img_%d' % idx},
                                     error_callback=lambda x: print(x))
            # argument(img_path, nb_new_images=1, save_to_dir=os.path.join('augmentation', dir_name))
    process_pool.close()
    process_pool.join()  # 阻塞等待
    print('耗时:%ds' % (time.time() - t0))
