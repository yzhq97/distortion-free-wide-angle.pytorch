import cv2
import numpy as np
import matplotlib.pyplot as plt


def FOV2f(fov, diagnoal):

    f = diagnoal / (2 * np.tan(fov / 2))
    return f


def correct(image, fov):

    h, w, _ = image.shape
    d = min(h, w)
    f = FOV2f(fov, d)
    r0 = d / (2 * np.tan(0.5 * np.arctan(d / (2 * f))))

    x = (np.arange(0, w, 1) - w / 2).astype(np.float32)
    y = (np.arange(0, h, 1) - h / 2).astype(np.float32)
    x, y = np.meshgrid(x, y)

    coords = np.stack([x, y], axis=-1)
    rp = np.linalg.norm(coords, axis=-1)
    ru = r0 * np.tan(0.5 * np.arctan(rp / f))

    x = x / ru * rp + w / 2
    y = y / ru * rp + h / 2

    out = cv2.remap(image, x, y, interpolation=cv2.INTER_LINEAR)

    return out


def get_uniform_stereo_mesh(image, fov, Q, mesh_ds_ratio):

    H, W, _ = image.shape
    Hm = H // mesh_ds_ratio + 2 * Q
    Wm = W // mesh_ds_ratio + 2 * Q
    d = min(Hm, Wm) * mesh_ds_ratio
    f = FOV2f(fov, d)
    r0 = d / (2 * np.tan(0.5 * np.arctan(d / (2 * f))))

    x = (np.arange(0, Wm, 1)).astype(np.float32) - (Wm // 2) + 0.5
    y = (np.arange(0, Hm, 1)).astype(np.float32) - (Hm // 2) + 0.5
    x = x * mesh_ds_ratio
    y = y * mesh_ds_ratio
    x, y = np.meshgrid(x, y)

    mesh_uniform = np.stack([x, y], axis=0)
    rp = np.linalg.norm(mesh_uniform, axis=0)
    ru = r0 * np.tan(0.5 * np.arctan(rp / f))

    x = x / ru * rp
    y = y / ru * rp
    mesh_stereo = np.stack([x, y], axis=0)

    return mesh_uniform, mesh_stereo


if __name__ == "__main__":

    image = cv2.imread("data/2_97.jpg")[::4, ::4, :]
    out = correct(image, np.pi * 97 / 180)
    plt.imshow(image[:, :, ::-1])
    plt.axis("off")
    plt.show()
    plt.imshow(out[:, :, ::-1])
    plt.axis("off")
    plt.show()