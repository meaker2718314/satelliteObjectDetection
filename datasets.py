import os
import torch
from torch.utils.data import Dataset
import math
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms import v2 as T
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes


def get_transform(train=True):
    transforms = []
    if train:
        transforms.append(T.RandomVerticalFlip(0.4))
        transforms.append(T.RandomHorizontalFlip(0.4))
        transforms.append(
            T.RandomChoice([
                T.RandomPerspective(distortion_scale=0.25),
                T.GaussianBlur(kernel_size=7),
                T.GaussianBlur(kernel_size=1),  # Identity transf.
            ], p=[1.5, 1.5, 2])
        )
        transforms.append(
            T.RandomChoice([
                T.ColorJitter(hue=0.15, contrast=0.15, brightness=0.1, saturation=0.15),
                T.RandomGrayscale(),
                T.ColorJitter(hue=None, contrast=None, brightness=None, saturation=None)  # Identity transf.
            ], p=[2, 1, 2])
        )

    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def retrieve_json(direct):
    import json
    with open(direct) as file:
        data = json.load(file)
        return sorted(data, key=lambda datapoint: datapoint['file_name'])


def loc_index(json_data, image):
    low = 0
    high = len(json_data) - 1

    # Binary search (list of Json dicts is ordered on 'file_name')

    while low <= high:

        mid = math.floor((low + high) / 2)
        found_img = json_data[mid]['file_name']
        if found_img > image:
            high = mid - 1
        elif found_img < image:
            low = mid + 1
        else:
            return mid

    raise Exception('Error: .JSON Data not found')


def json_to_bounds(mask_data):
    invalid_labels = [{}, 'Skip']

    label = mask_data['label']

    if label in invalid_labels:
        return None

    boxes = label[list(label.keys())[0]]
    box_tensors = []

    for box in boxes:
        corners = box['geometry']
        x1, y1 = 1e5, 1e5
        x2, y2 = -1e5, -1e5

        for point in corners:
            x1, y1 = min(x1, point['x']), min(y1, point['y'])
            x2, y2 = max(x2, point['x']), max(y2, point['y'])

        box_tensors.append(torch.tensor([x1, y1, x2, y2], dtype=torch.int64))

    return torch.stack(box_tensors, dim=0)


def no_bounds(mask_data):
    return json_to_bounds(mask_data) is None


class TankDataset(Dataset):
    def __init__(self, image_dir, mask_dir, data_cutoffs=(0.0, 1.0), transform=None, img_dup=1):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = []

        import random

        random.seed(1)

        all_images = os.listdir(self.image_dir)
        random.shuffle(all_images)

        n_images = len(all_images)
        start = math.ceil(n_images * data_cutoffs[0])
        stop = math.ceil(n_images * data_cutoffs[1])

        init_sample = os.listdir(self.image_dir)[start: stop]

        self.masks = retrieve_json(mask_dir)
        filtered_sample = []

        for img in init_sample:
            idx = loc_index(self.masks, img)
            if no_bounds(self.masks[idx]):  # Exclude negative samples
                continue
            filtered_sample.append(img)

        for k in range(img_dup):
            self.images.extend(filtered_sample)

        self.json_idxs = []
        for img in self.images:
            self.json_idxs.append(loc_index(self.masks, img))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)
        img = tv_tensors.Image(read_image(img_path))

        mask_data = self.masks[self.json_idxs[index]]

        boxes = json_to_bounds(mask_data)

        n_obj = boxes.size()[0]
        labels = torch.ones((n_obj,), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # dx * dy
        iscrowd = torch.zeros((n_obj,), dtype=torch.int64)

        target = {
            'boxes': tv_tensors.BoundingBoxes(boxes, format='XYXY', canvas_size=img.shape[1:]),
            'labels': labels,
            'image_id': index,
            'area': area,
            'iscrowd': iscrowd,
        }
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target
