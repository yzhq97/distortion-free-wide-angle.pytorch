import cv2
import io
import numpy as np
import matplotlib.pyplot as plt


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_overlay_flow(image, flow, color="aquamarine", ratio=0.7):

    H, W, _ = image.shape
    Hm, Wm, _ = flow.shape

    fig = plt.figure(figsize=(W, H), dpi=1, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis("equal")
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow((image * ratio).astype(np.uint8))
    X = np.arange(0.5, Wm, 1).astype(np.float32) * (H / Hm)
    Y = np.arange(0.5, Hm, 1).astype(np.float32) * (W / Wm)
    X, Y = np.meshgrid(X, Y)
    q = ax.quiver(X, Y, flow[:, :, 0], -flow[:, :, 1], color=color, scale=1.0, scale_units='xy')

    # plt.show()

    im = get_img_from_fig(fig, dpi=1)
    im = (im / 255.).astype(np.float32)

    plt.close(fig)

    return im