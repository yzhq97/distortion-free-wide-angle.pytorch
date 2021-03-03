import cv2
import os
import numpy as np
from torch.utils.data import Dataset
from stereographic import get_uniform_stereo_mesh
from perception import get_face_masks, get_object_masks


class ImageDataset(Dataset):

    def __init__(self, args, root='data'):

        self.Q = args.Q
        self.mesh_ds_ratio = args.mesh_ds_ratio
        self.data_list = []
        for names in os.listdir(root):
            if names.endswith(".jpg"):
                self.data_list.append(os.path.join(root, names))
        self.data_list = sorted(self.data_list)


    def get_image_by_file(self, file, classes=None):

        data_name = file
        fov = int(data_name.split('/')[-1].split('.')[0].split('_')[-1])

        image = cv2.imread(data_name)
        H, W, _ = image.shape

        Hm = H // self.mesh_ds_ratio
        Wm = W // self.mesh_ds_ratio

        if classes is None:
            seg_mask, box_masks = get_face_masks(image)
        else:
            seg_mask, box_masks = get_object_masks(image, classes=classes)

        seg_mask = cv2.resize(seg_mask.astype(np.float32), (Wm, Hm))
        box_masks = [cv2.resize(box_mask.astype(np.float32), (Wm, Hm)) for box_mask in box_masks]
        box_masks = np.stack(box_masks, axis=0)
        seg_mask_padded = np.pad(seg_mask, [[self.Q, self.Q], [self.Q, self.Q]], "constant")
        box_masks_padded = np.pad(box_masks, [[0, 0], [self.Q, self.Q], [self.Q, self.Q]], "constant")

        mesh_uniform_padded, mesh_stereo_padded = get_uniform_stereo_mesh(image, fov * np.pi / 180, self.Q, self.mesh_ds_ratio)

        radial_distance_padded = np.linalg.norm(mesh_uniform_padded, axis=0)
        half_diagonal = np.linalg.norm([H + 2 * self.Q * self.mesh_ds_ratio, W + 2 * self.Q * self.mesh_ds_ratio]) / 2.
        ra = half_diagonal / 2.
        rb = half_diagonal / (2 * np.log(99))
        correction_strength = 1 / (1 + np.exp(-(radial_distance_padded - ra) / rb))

        return image, mesh_uniform_padded, mesh_stereo_padded, correction_strength, seg_mask_padded, box_masks_padded


    def __getitem__(self, index):

        index = index % len(self.data_list)
        data_name = self.data_list[index]

        return self.get_image_by_file(data_name)


    def __len__(self):
        return len(self.data_list)