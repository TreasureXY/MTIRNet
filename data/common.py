import random
import PIL.Image as pil_image
import numpy as np
import torch


# 读取一张图片
def load_img(path):
    return pil_image.open(path).convert('RGB')


# 图片到numpy
def img2np(img):
    return np.array(img)


# numpy到tensor
def np2tensor(x):
    return torch.from_numpy(x).permute(2, 0, 1).float()  # C * H * W


# tensor到图片
def tensor2img(x):
    return pil_image.fromarray(x.byte().cpu().numpy())


# 归一化
def normalize(x, max_value):
    return x * (max_value / 255.0)


# 随机旋转、对称patch
def augment_patches(patches, hflip=True, vflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    rot90 = rot and random.random() < 0.5
    ret = []
    for p in patches:
        if hflip:
            p = np.fliplr(p).copy()
        if vflip:
            p = np.flipud(p).copy()
        if rot90:
            p = np.rot90(p, axes=(0, 1)).copy()
        ret.append(p)

    return ret


# 随机裁剪图片
def get_patch(l, h, patch_size, augment_patch=False):
    H, W = l.shape[:2]
    # 获得裁剪左上角的坐标
    x, y = random.randrange(0, W - patch_size + 1), random.randrange(0, H - patch_size + 1)
    # 进行裁剪
    l = l[y:y + patch_size, x:x + patch_size]
    h = h[y:y + patch_size, x:x + patch_size]
    if augment_patch:
        l, h = augment_patches([l, h])
    return l, h

# 裁剪
def quantize(x, quantize_range):
    return x.clamp(quantize_range[0], quantize_range[1])

def add_noise(img_np, sigma, train):
    temp_image = np.float32(np.copy(img_np))

    H, W, C = temp_image.shape
    if not train:
        np.random.seed(0)
        noise = np.random.randn(H, W, C) * float(sigma)
    else:
        noise = np.random.randn(H, W, C) * float(sigma)

    return temp_image + noise