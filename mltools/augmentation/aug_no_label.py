"""
Descripttion:
version:
Author: xiaoshuyui
email: guchengxi1994@qq.com
Date: 2022-07-09 08:43:17
LastEditors: xiaoshuyui
LastEditTime: 2022-07-09 09:15:33
"""

import copy
import random
from multiprocessing import Pool
from typing import List

import numpy as np
from skimage import io
from tqdm import tqdm

from mltools import __CPUS__, split_file_name
from mltools.augmentation import AugmentationTypes
from mltools.augmentation.base import BaseAugmentation
from mltools.augmentation.nolabel import *
from mltools.augmentation.nolabel.optional.resize import img_resize_with_shape
from mltools.decorators.incomplete_feature import IncompleteFeature
from mltools.log.logger import logger


@IncompleteFeature(message="multi processing development is incomplete")
class NoLabelAugmentation(BaseAugmentation):
    def from_config(self, c: dict):
        return super().from_config(c)

    def append(self, imgPath):
        return super().append(imgPath)

    def __init__(
        self,
        imgs: List[str],
        parallel: bool = False,
        savedPath: str = "",
        augNumber: int = 1,
        augMethods: List[str] = ["noise", "rotation", "trans", "flip", "zoom"],
        optionalMethods: List[str] = [],
    ) -> None:
        super().__init__(
            imgs, parallel, savedPath, augNumber, augMethods, optionalMethods
        )
        self.augType = AugmentationTypes.NoLabel

    def onlyFlip(self, flipList=[1, 0, -1]):
        if self.parallel:
            pool = Pool(__CPUS__ - 1)
        else:
            pool = None
        _imgCount = 0
        for i in self.imgs:
            _imgCount += 1
            _img = io.imread(i)
            _imgList = img_flip(_img, flipList=flipList)
            _flip_count = 0
            for j in _imgList:
                _flip_count += 1
                io.imsave(
                    self.savedPath + "flip-{}-{}.jpg".format(_imgCount, _flip_count), j
                )

    def onlyNoise(self, noise_types: list = []):
        if self.parallel:
            pool = Pool(__CPUS__ - 1)
        else:
            pool = None
        _imgCount = 0
        for i in self.imgs:
            _imgCount += 1
            _img = io.imread(i)
            for j in range(0, self.augNumber):
                _noisedImg = img_noise(_img, noise_types)
                io.imsave(
                    self.savedPath + "noise-{}-{}.jpg".format(_imgCount, j), _noisedImg
                )

    def onlyRotation(self):
        if self.parallel:
            pool = Pool(__CPUS__ - 1)
        else:
            pool = None
        _imgCount = 0
        for i in self.imgs:
            _imgCount += 1
            _img = io.imread(i)
            for j in range(0, self.augNumber):
                _angle = random.randint(-180, 180)
                _rotatedImg = img_rotation(_img, angle=_angle)
                io.imsave(
                    self.savedPath
                    + "rotation-{}-{}-{}-{}.jpg".format(
                        split_file_name(i), _imgCount, j, _angle
                    ),
                    _rotatedImg,
                )

    def onlyTranslation(self):
        if self.parallel:
            pool = Pool(__CPUS__ - 1)
        else:
            pool = None
        _imgCount = 0
        for i in self.imgs:
            _imgCount += 1
            _img = io.imread(i)
            for j in range(0, self.augNumber):
                _transImg = img_translation(_img)
                io.imsave(
                    self.savedPath
                    + "trans-{}-{}-{}.jpg".format(split_file_name(i), _imgCount, j),
                    _transImg,
                )

    def onlyZoom(self):
        if self.parallel:
            pool = Pool(__CPUS__ - 1)
        else:
            pool = None
        _imgCount = 0
        for i in self.imgs:
            _imgCount += 1
            _img = io.imread(i)
            for j in range(0, self.augNumber):
                _factor = round(random.uniform(0.5, 1.5), 2)
                _zoomImg = img_zoom(_img, _factor)
                io.imsave(
                    self.savedPath
                    + "zoom-{}-{}-{}-{}.jpg".format(
                        split_file_name(i), _imgCount, j, _factor
                    ),
                    _zoomImg,
                )

    def onlyCrop(self):
        if self.parallel:
            pool = Pool(__CPUS__ - 1)
        else:
            pool = None
        _methodsList = [
            multi_polygon_crop,
            multi_rectangle_crop,
            polygon_crop,
            rectangle_crop,
        ]
        _imgCount = 0
        for i in self.imgs:
            _imgCount += 1
            _img = io.imread(i)
            for j in range(0, self.augNumber):
                _augMethod = random.sample(_methodsList, k=1)[0]
                _augImg = _augMethod(_img)
                io.imsave(
                    self.savedPath
                    + "crop-{}-{}-{}-{}.jpg".format(
                        split_file_name(i), _imgCount, j, _augMethod.__name__
                    ),
                    _augImg,
                )

    def onlyCutmix(self):
        if len(self.imgs) < 2:
            return
        for i in range(0, self.augNumber):
            _imgPair = random.sample(self.imgs, k=2)
            img1 = io.imread(_imgPair[0])
            img2 = io.imread(_imgPair[1])
            _factor = random.uniform(0.3, 0.7)
            result = cutmix(img1, img2, _factor)
            io.imsave(
                self.savedPath
                + "cutmix-{}-{}-{}-{}.jpg".format(
                    split_file_name(_imgPair[0]),
                    split_file_name(_imgPair[1]),
                    i,
                    _factor,
                ),
                result,
            )

    def onlyDistort(self):
        if self.parallel:
            pool = Pool(__CPUS__ - 1)
        else:
            pool = None

        _imgCount = 0
        for i in self.imgs:
            _imgCount += 1
            _img = io.imread(i)
            for j in range(0, self.augNumber):
                _distortImg = img_distort(_img)
                io.imsave(
                    self.savedPath
                    + "distort-{}-{}-{}.jpg".format(split_file_name(i), _imgCount, j),
                    _distortImg,
                )

    def onlyInpaint(self, reshape: bool = True):
        if self.parallel:
            pool = Pool(__CPUS__ - 1)
        else:
            pool = None

        _imgCount = 0
        for i in self.imgs:
            _imgCount += 1
            _img = io.imread(i)
            if reshape:
                _img = img_resize_with_shape(_img, 128, 128)
            _methodsList = [polygon_inpaint, rectangle_inpaint]
            for j in range(0, self.augNumber):
                _method = random.sample(_methodsList, k=1)[0]
                result = _method(_img)
                io.imsave(
                    self.savedPath
                    + "inpaint-{}-{}-{}-{}.jpg".format(
                        split_file_name(i),
                        _imgCount,
                        j,
                        _method.__name__,
                    ),
                    result,
                )

    def onlyMixup(self):
        if len(self.imgs) < 2:
            return
        for i in range(0, self.augNumber):
            _imgPair = random.sample(self.imgs, k=2)
            img1 = io.imread(_imgPair[0])
            img2 = io.imread(_imgPair[1])
            _factor = random.uniform(0.3, 0.7)
            result = mixup(img1, img2, _factor)
            io.imsave(
                self.savedPath
                + "mixup-{}-{}-{}-{}.jpg".format(
                    split_file_name(_imgPair[0]),
                    split_file_name(_imgPair[1]),
                    i,
                    _factor,
                ),
                result,
            )

    def onlyMosaic(self):
        if len(self.imgs) == 0:
            return

        for i in range(0, self.augNumber):
            i1 = random.sample(self.imgs, k=1)[0]
            i2 = random.sample(self.imgs, k=1)[0]
            i3 = random.sample(self.imgs, k=1)[0]
            i4 = random.sample(self.imgs, k=1)[0]
            img1 = io.imread(i1)
            img2 = io.imread(i2)
            img3 = io.imread(i3)
            img4 = io.imread(i4)

            result, _, f1, f2 = mosaic_img_no_reshape(
                [img1, img2, img3, img4],
                widthFactor=random.uniform(0.3, 0.7),
                heightFactor=random.uniform(0.3, 0.7),
            )
            io.imsave(
                self.savedPath
                + "mosaic-{}-{}-{}-{}-{}-{}-{}.jpg".format(
                    split_file_name(i1),
                    split_file_name(i2),
                    split_file_name(i3),
                    split_file_name(i4),
                    i,
                    f1,
                    f2,
                ),
                result,
            )

    def onlyResize(self):
        if self.parallel:
            pool = Pool(__CPUS__ - 1)
        else:
            pool = None

        _imgCount = 0
        for i in self.imgs:
            _imgCount += 1
            _img = io.imread(i)
            for j in range(0, self.augNumber):
                f1 = random.uniform(0.1, 0.9)
                f2 = random.uniform(0.1, 0.9)
                _resizedImg = img_resize(_img, f1, f2)
                io.imsave(
                    self.savedPath
                    + "resize-{}-{}-{}-{}-{}.jpg".format(
                        split_file_name(i), _imgCount, j, f1, f2
                    ),
                    _resizedImg,
                )

    def go(self):
        """本来想命名为 `process` 的，但是因为和很多函数名冲突，就改为`go`了"""
        if self.parallel:
            pool = Pool(__CPUS__ - 1)
        else:
            pool = None
        if self.config == {}:
            _augs = copy.deepcopy(self.augMethods)
            if len(self.optionalMethods) > 0:
                _augs.extend(self.optionalMethods)
            if pool is None:
                for i in range(0, len(self.imgs)):
                    _img = io.imread(self.imgs[i])
                    random_aug(_img, i, _augs, self.augNumber, self.savedPath)
            else:
                poolList = []
                for i in range(0, len(self.imgs)):
                    _img = io.imread(self.imgs[i])
                    resultPool = pool.apply_async(
                        random_aug, (_img, i, _augs, self.augNumber, self.savedPath)
                    )
                    poolList.append(resultPool)

                logger.info("successfully create {} tasks".format(len(poolList)))

                for pr in tqdm(poolList):
                    _ = pr.get()

                logger.info("Done!")


def random_aug(img: np.ndarray, index=0, augMethods=[], augNumber=1, savedPath=""):
    """这个方法中所有的增广方式都是随机的,
    @ index: 图片编号
    """
    _augs = augMethods
    # 不可能每种方式都增广一遍，采用随机的方式，随到1的时候增广
    l = np.random.randint(2, size=len(_augs))
    if np.sum(l) == 0:
        l[0] = 1

    l = l.tolist()
    p = list(zip(_augs, l))

    for r in range(0, augNumber):
        for i in p:
            if i[1] == 1:
                if i[0] == "noise":
                    img = img_noise(img)

                elif i[0] == "rotation":
                    img = img_rotation(img)

                elif i[0] == "trans":
                    img = img_translation(img)

                elif i[0] == "zoom":
                    zoomFactor = random.uniform(0.8, 1.8)
                    img = img_zoom(img, zoomFactor)

                elif i[0] == "flip":
                    _imgList = img_flip(img)
                    img = random.sample(_imgList, k=1)[0]

                elif i[0] == "crop":
                    _methodsList = [
                        multi_polygon_crop,
                        multi_rectangle_crop,
                        polygon_crop,
                        rectangle_crop,
                    ]
                    _method = random.sample(_methodsList, k=1)[0]
                    img = _method(img)

                elif i[0] == "distort":
                    img = img_distort(img)

                elif i[0] == "inpaint":
                    _methodsList = [polygon_inpaint, rectangle_inpaint]
                    _method = random.sample(_methodsList, k=1)[0]
                    img = _method(img)

                elif i[0] == "mosaic":
                    img, _, _, _ = mosaic_img_no_reshape([img, img, img, img])

                elif i[0] == "resize":
                    img = img_resize(img)

                # img = np.array(img, dtype=np.uint8)
        io.imsave(savedPath + "round{}-{}.jpg".format(r, index), img)


def random_aug_stream(
    img: np.ndarray,
    augMethods=[
        "noise",
        "rotation",
        "trans",
        "zoom",
        "flip",
        "crop",
        "distort",
        "inpaint",
        "mosaic",
        "resize",
    ],
    augNumber=1,
):
    """
    随机增广，流式返回增广后的图片
    :param img: 输入图片，np.ndarray
    :param index: 图片编号
    :param augMethods: 可用的增广方法列表
    :param augNumber: 总共生成多少组增广图片
    :yield: 每次返回一张增广后的图片，类型为 np.ndarray
    """
    _augs = augMethods
    l = np.random.randint(2, size=len(_augs))
    if np.sum(l) == 0:
        l[0] = 1

    l = l.tolist()
    p = list(zip(_augs, l))

    for r in range(augNumber):
        aug_img = img.copy()

        for i in p:
            if i[1] == 1:
                if i[0] == "noise":
                    aug_img = img_noise(aug_img)

                elif i[0] == "rotation":
                    aug_img = img_rotation(aug_img)

                elif i[0] == "trans":
                    aug_img = img_translation(aug_img)

                elif i[0] == "zoom":
                    zoomFactor = random.uniform(0.8, 1.8)
                    aug_img = img_zoom(aug_img, zoomFactor)

                elif i[0] == "flip":
                    _imgList = img_flip(aug_img)
                    aug_img = random.sample(_imgList, k=1)[0]

                elif i[0] == "crop":
                    _methodsList = [
                        multi_polygon_crop,
                        multi_rectangle_crop,
                        polygon_crop,
                        rectangle_crop,
                    ]
                    _method = random.sample(_methodsList, k=1)[0]
                    aug_img = _method(aug_img)

                elif i[0] == "distort":
                    aug_img = img_distort(aug_img)

                elif i[0] == "inpaint":
                    _methodsList = [polygon_inpaint, rectangle_inpaint]
                    _method = random.sample(_methodsList, k=1)[0]
                    aug_img = _method(aug_img)

                elif i[0] == "mosaic":
                    aug_img, _, _, _ = mosaic_img_no_reshape(
                        [aug_img, aug_img, aug_img, aug_img]
                    )

                elif i[0] == "resize":
                    aug_img = img_resize(aug_img)

        yield aug_img
