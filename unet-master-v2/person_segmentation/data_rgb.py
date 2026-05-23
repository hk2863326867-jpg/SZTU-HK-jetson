
import sys
import os
import numpy as np
import skimage.io as io
import skimage.transform as trans
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def adjustData(img, mask, flag_multi_class=False, num_class=2):
    if flag_multi_class:
        img = img / 255
        mask = mask[:, :, :, 0] if len(mask.shape) == 4 else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i, i] = 1
        new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2], new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))
        mask = new_mask
    elif np.max(img) > 1:
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img, mask)


def trainGenerator_rgb(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="rgb",
                       mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                       flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256, 256), seed=1):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)


def testGenerator_rgb(test_path, num_image=30, target_size=(256, 256), as_gray=False):
    for i in range(num_image):
        img_path = os.path.join(test_path, "%d.png" % i)
        if not os.path.exists(img_path):
            img_path = os.path.join(test_path, "%d.jpg" % i)
        
        if as_gray:
            img = io.imread(img_path, as_gray=True)
        else:
            img = io.imread(img_path)
        
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, (1,) + img.shape)
        yield img


def saveResult_rgb(save_path, npyfile, flag_multi_class=False, num_class=2):
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        img = (img * 255).astype(np.uint8)
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)
