from ACE_cupy import ACE_cpColor
import cupy as cp
import time
import os
import cv2
import gc
from multiprocessing import Pool
import random


st = time.time()
train_path = 'data/UnderwaterDetection/train_val/images/train/'
val_path = 'data/UnderwaterDetection/train_val/images/val/'
test_path = 'data/UnderwaterDetection/test-A-image/'

train_enhance_path = 'data/UnderwaterDetection/train_val_enhance/images/train/'
val_enhance_path = 'data/UnderwaterDetection/train_val_enhance/images/val/'
test_enhance_path = 'data/UnderwaterDetection/test-A-image-enhance/'

if not os.path.exists(train_enhance_path):
    os.makedirs(train_enhance_path)
if not os.path.exists(val_enhance_path):
    os.makedirs(val_enhance_path)
if not os.path.exists(test_enhance_path):
    os.makedirs(test_enhance_path)


def preprocess_train_img(img_path, gpu_id):
    with cp.cuda.Device(gpu_id):
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        imgn = img_path.split('/')[-1]
        img = cv2.imread(img_path)
        img = ACE_cpColor(img, gpu_id=gpu_id)
        cv2.imwrite(os.path.join(train_enhance_path, imgn), img)
        print(f"preprocess_train_img:{imgn}")
        del img, img_path, imgn
        gc.collect()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        # print(mempool.used_bytes())              # 0
        # print(mempool.total_bytes())             # 0
        # print(pinned_mempool.n_free_blocks())    # 0


def preprocess_val_img(img_path, gpu_id):
    with cp.cuda.Device(gpu_id):
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        imgn = img_path.split('/')[-1]
        img = cv2.imread(img_path)
        img = ACE_cpColor(img, gpu_id=gpu_id)
        cv2.imwrite(os.path.join(val_enhance_path, imgn), img)
        print(f"preprocess_val_img:{imgn}")
        del img, img_path, imgn
        gc.collect()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        print("used_bytes:", mempool.used_bytes())              # 0
        print("total_bytes:", mempool.total_bytes())             # 0
        print("pinned_mempool:", pinned_mempool.n_free_blocks())    # 0


def preprocess_test_img(img_path, gpu_id):
    with cp.cuda.Device(gpu_id):
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        imgn = img_path.split('/')[-1]
        img = cv2.imread(img_path)
        img = ACE_cpColor(img, gpu_id=gpu_id)
        cv2.imwrite(os.path.join(test_enhance_path, imgn), img)
        print(f"preprocess_test_img:{imgn}")
        del img, img_path, imgn
        gc.collect()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        print("used_bytes:", mempool.used_bytes())              # 0
        print("total_bytes:", mempool.total_bytes())             # 0
        print("pinned_mempool:", pinned_mempool.n_free_blocks())    # 0


if __name__ == '__main__':
    random.seed(42)
    train_imgns = os.listdir(train_path)
    train_ok_imgns = os.listdir(train_enhance_path)
    val_imgns = os.listdir(val_path)
    val_ok_imgns = os.listdir(val_enhance_path)
    test_imgns = os.listdir(test_path)
    test_ok_imgns = os.listdir(test_enhance_path)

    train_imgs_path = [os.path.join(train_path, i)
                       for i in train_imgns if i not in train_ok_imgns]
    # train_imgs_path = [os.path.join(train_path, i) for i in train_imgns]
    train_len = len(train_imgs_path)
    train_gpu_index = [0, 1] * (train_len // 2)
    if train_len > len(train_gpu_index):
        train_gpu_index.append(0)

    val_imgs_path = [os.path.join(val_path, i)
                     for i in val_imgns if i not in val_ok_imgns]
    # val_imgs_path = [os.path.join(val_path, i) for i in val_imgns]
    val_len = len(val_imgs_path)
    val_gpu_index = [0, 1] * (val_len // 2)
    if val_len > len(val_gpu_index):
        val_gpu_index.append(0)

    test_imgs_path = [os.path.join(test_path, i)
                      for i in test_imgns if i not in test_ok_imgns]
    # test_imgs_path = [os.path.join(test_path, i) for i in test_imgns]
    test_len = len(test_imgs_path)
    test_gpu_index = [0, 1] * (test_len // 2)
    if test_len > len(test_gpu_index):
        test_gpu_index.append(0)

    pool_size = 8
    pool = Pool(pool_size)
    pool.starmap(preprocess_train_img, zip(train_imgs_path, train_gpu_index))
    pool.close()
    pool.join()

    pool2 = Pool(pool_size)
    pool2.starmap(preprocess_val_img, zip(val_imgs_path, val_gpu_index))
    pool2.close()
    pool2.join()

    pool3 = Pool(pool_size)
    pool3.starmap(preprocess_test_img, zip(test_imgs_path, test_gpu_index))
    pool3.close()
    pool3.join()
