import cupyx
import cupyx.scipy.ndimage
import cupy as cp
import numpy as np
import math
import gc


# 线性拉伸，去掉最大最小0.5%的像素值，然后线性拉伸至[0,1]
def stretchImage(data, s=0.005, bins=2000, gpu_id=0):
    with cp.cuda.Device(gpu_id):
        ht = cp.histogram(data, bins)
        d = cp.cumsum(ht[0])/float(data.size)
        lmin = 0
        lmax = bins-1
        while lmin < bins:
            if d[lmin] >= s:
                break
            lmin += 1
        while lmax >= 0:
            if d[lmax] <= 1-s:
                break
            lmax -= 1
        return cp.clip((data-ht[1][lmin])/(ht[1][lmax]-ht[1][lmin]), 0, 1)


g_para = {}


def getPara(radius=5, gpu_id=0):  # 根据半径计算权重参数矩阵
    with cp.cuda.Device(gpu_id):
        global g_para
        m = g_para.get(radius, None)
        if m is not None:
            return m
        size = radius*2+1
        m = cp.zeros((size, size))
        for h in range(-radius, radius+1):
            for w in range(-radius, radius+1):
                if h == 0 and w == 0:
                    continue
                m[radius+h, radius+w] = 1.0/math.sqrt(h**2+w**2)
        m /= m.sum()
        g_para[radius] = m
        return m


def ACE_cp(img, ratio=4, radius=300, gpu_id=0):  # 常规的ACE实现
    with cp.cuda.Device(gpu_id):
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        para = getPara(radius, gpu_id=gpu_id)
        # print("para.device:", para.device)
        # print("img.device:", img.device)
        height, width = img.shape
        size = 2 * radius + 1
        # zh,zw = [0]*radius + list(range(height)) + [height-1]*radius, [0]*radius + list(range(width))  + [width -1]*radius
        # Z = img[cp.ix_(zh, zw)]
        Z = cp.zeros((height+2*radius, width+2*radius))
        Z[radius:-radius, radius:-radius] = img
        res = cp.zeros(img.shape)
        para = cp.asarray(para)
        for h in range(size):
            for w in range(size):
                if para[h][w] == 0:
                    continue
                res += (para[h][w] *
                        cp.clip((img - Z[h:h + height, w:w + width]) * ratio, -1, 1))
        del Z, para
        gc.collect()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        return res


def ACE_cpFast(img, ratio, radius, gpu_id=0):  # 单通道ACE快速增强实现
    with cp.cuda.Device(gpu_id):
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        height, width = img.shape[:2]
        if min(height, width) <= 2:
            return cp.ones(img.shape)*0.5
        # Rs = cv2.resize(img, ((width+1)//2, (height+1)//2))
        # Rf = ACE_cpFast(Rs, ratio, radius)
        # Rf = cv2.resize(Rf, (width, height))
        # Rs = cv2.resize(Rs, (width, height))
        Rs = cupyx.scipy.ndimage.zoom(img, 0.5, mode='opencv')
        Rf = ACE_cpFast(Rs, ratio, radius, gpu_id=gpu_id)  # 递归调用
        factor = (height/Rs.shape[0], width/Rs.shape[1])
        Rf = cupyx.scipy.ndimage.zoom(Rf, factor, mode='opencv')
        Rs = cupyx.scipy.ndimage.zoom(Rs, factor, mode='opencv')
        ace_img = ACE_cp(img, ratio, radius, gpu_id=gpu_id)
        ace_rs = ACE_cp(Rs, ratio, radius, gpu_id=gpu_id)
        res = Rf + ace_img - ace_rs
        del img, Rs, ace_img, ace_rs, Rf
        gc.collect()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        return res


def ACE_cpColor(img, ratio=4, radius=3, gpu_id=0):
    # rgb三通道分别增强，ratio是对比度增强因子，radius是卷积模板半径
    with cp.cuda.Device(gpu_id):
        img = cp.array(img)/255.
        # res = cp.zeros(img.shape)
        for k in range(3):
            img[:, :, k] = stretchImage(ACE_cpFast(
                img[:, :, k], ratio, radius, gpu_id=gpu_id), gpu_id=gpu_id)
        return cp.asnumpy(img*255.).astype(np.uint8)