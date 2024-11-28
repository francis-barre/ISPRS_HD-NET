from osgeo import gdal
import scipy.io as io
from imgaug import augmenters as iaa
import torchvision.transforms.functional as transF
from PIL import Image
import logging
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

mean_std_dict = {
    "WHU": [
        "WHU",
        [0.43526826, 0.44523221, 0.41307611],
        [0.20436029, 0.19237618, 0.20128716],
        ".tif",
    ],
    "Mass": [
        "Mass",
        [0.32208377, 0.32742606, 0.2946236],
        [0.18352227, 0.17701593, 0.18039343],
        ".tif",
    ],
    "Inria": [
        "Inria",
        [0.42314604, 0.43858219, 0.40343547],
        [0.18447358, 0.16981384, 0.1629876],
        ".tif",
    ],
    "NOCI": [
        "NOCI",
        [0.4260, 0.4260, 0.3984],
        [0.1718, 0.1570, 0.1388],
        ".tif",
    ],
    "NOCI_BW_poor": [
        "NOCI_BW_poor",
        [0.2644, 0.2644, 0.2644],
        [0.0620, 0.0620, 0.0620],
        ".tif",
    ],
    "NOCI_BW": ["NOCI_BW", [0.4001, 0.4001, 0.4001], [0.1432, 0.1432, 0.1432], ".tif"],
}

'''
NOCI-extrabbox
[0.3909, 0.3933, 0.3702],
[0.1587, 0.1429, 0.1269],
'''

class BuildingDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        training=False,
        txt_name: str = "train.txt",
        data_name="WHU",
        predict=False,
        predict_txt_name="predict.txt",
        image_folder="train/image",
        label_folder="train/label",
        boundary_folder="boundary",
        dataset_folder="dataset",
    ):
        self.name, self.mean, self.std, self.shuffix = mean_std_dict[data_name]
        self.predict = predict
        if self.name == "Mass":
            self.imgs_dir = os.path.join(dataset_dir, "train", "image")
            self.labels_dir = os.path.join(dataset_dir, "train", "label")
            self.dis_dir = os.path.join(dataset_dir, "boundary")
            txt_path = os.path.join(dataset_dir, dataset_folder, txt_name)
            assert os.path.exists(txt_path), "file '{}' does not exist.".format(
                txt_path
            )
            with open(os.path.join(txt_path), "r") as f:
                file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
            self.scale = 1
            self.training = training
            self.images = [
                os.path.join(self.imgs_dir, x + self.shuffix) for x in file_names
            ]
            if not self.predict:
                self.labels = [
                    os.path.join(self.labels_dir, x + self.shuffix) for x in file_names
                ]
                self.dis = [os.path.join(self.dis_dir, x + ".mat") for x in file_names]
                assert (len(self.images) == len(self.labels)) & (
                    len(self.images) == len(self.dis)
                )

            logging.info(f"Creating dataset with {len(self.images)} examples")

        else:
            # mode = txt_name.split(".")[0]
            self.imgs_dir = os.path.join(dataset_dir, image_folder)
            self.labels_dir = os.path.join(dataset_dir, label_folder)
            self.dis_dir = os.path.join(dataset_dir, boundary_folder)
            txt_path = os.path.join(dataset_dir, "dataset", txt_name)
            assert os.path.exists(txt_path), "file '{}' does not exist.".format(
                txt_path
            )
            with open(os.path.join(txt_path), "r") as f:
                file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
            self.scale = 1
            self.training = training
            self.images = [
                os.path.join(self.imgs_dir, x + self.shuffix) for x in file_names
            ]
            if not self.predict:
                self.labels = [
                    os.path.join(self.labels_dir, x + self.shuffix) for x in file_names
                ]
                self.dis = [os.path.join(self.dis_dir, x + ".mat") for x in file_names]
                assert (len(self.images) == len(self.labels)) & (
                    len(self.images) == len(self.dis)
                )

            logging.info(f"Creating dataset with {len(self.images)} examples")

        # 影像预处理方法
        self.transform = iaa.Sequential(
            [
                iaa.Rot90([0, 1, 2, 3]),
                iaa.VerticalFlip(p=0.5),
                iaa.HorizontalFlip(p=0.5),
            ]
        )

    def __len__(self):
        return len(self.images)

    def _load_mat(self, filename):
        return io.loadmat(filename)

    def _load_maps(
        self,
        filename,
    ):
        dct = self._load_mat(filename)
        distance_map = dct["depth"].astype(np.int32)
        return distance_map

    def __getitem__(self, index):
        if self.name == "Mass":
            # img_file = self.images[index]
            # img = np.array(Image.open(img_file))
            # labels = readTif(self.labels[index])
            # width = labels.RasterXSize
            # height = labels.RasterYSize
            # label = labels.ReadAsArray(0, 0, width, height)
            # label = label[0, :, :] / 255

            img_file = self.images[index]
            img = np.array(Image.open(img_file))
            if not self.predict:
                label_file = self.labels[index]
                label = (
                    np.array(Image.open(label_file).convert("P")).astype(np.int16)
                    / 255.0
                )
        elif self.name == "WHU":
            img_file = self.images[index]
            img = np.array(Image.open(img_file))
            if not self.predict:
                label_file = self.labels[index]
                label = (
                    np.array(Image.open(label_file).convert("P")).astype(np.int16)
                    / 255.0
                )
        elif self.name == "Inria":
            img_file = self.images[index]
            img = np.array(Image.open(img_file))
            if not self.predict:
                label_file = self.labels[index]
                label = (
                    np.array(Image.open(label_file).convert("P")).astype(np.int16)
                    / 255.0
                )
        elif (
            self.name == "NOCI"
            or self.name == "NOCI_BW_poor"
            or (self.name == "NOCI_BW")
        ):
            img_file = self.images[index]
            img = np.array(Image.open(img_file))

            if not self.predict:
                label_file = self.labels[index]
                label = (
                    np.array(Image.open(label_file).convert("P")).astype(np.int16)
                    / 255.0
                )

        # 利用_load_maps获取得到的distance_map和angle_map
        if self.training:
            distance_map = self._load_maps(self.dis[index])
            distance_map = np.array(distance_map)
            img, label = self.transform(
                image=img,
                segmentation_maps=np.stack(
                    (label[np.newaxis, :, :], distance_map[np.newaxis, :, :]), axis=-1
                ).astype(np.int32),
            )

            label, distance_map = label[0, :, :, 0], label[0, :, :, 1]

        img = transF.to_tensor(img.copy())

        if not self.predict:
            label = (transF.to_tensor(label.copy()) > 0).int()
        # 标准化
        img = transF.normalize(img, self.mean, self.std)
        if self.training:  # training
            return {
                "image": img.float(),
                "label": label.float(),
                "distance_map": distance_map,
                "name": self.images[index],
            }
        elif self.predict:  # prediction
            return {"image": img.float(), "name": self.images[index]}
        else:  # evaluation
            return {
                "image": img.float(),
                "label": label.float(),
                "name": self.images[index],
            }


def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset is None:
        print(fileName + "can not open the file")
    return dataset
