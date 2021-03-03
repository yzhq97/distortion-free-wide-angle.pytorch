import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import cv2
import argparse
from torch import optim
from src.data import ImageDataset
from src.energy import Energy
from src.visualization import get_overlay_flow


parser = argparse.ArgumentParser(description='Distortion-Free-Wide-Angle-Portraits-on-Camera-Phones')
parser.add_argument('--file', type=str, required=True)

parser.add_argument('--num_iter', type=int, default=200, help="number of optimization steps")
parser.add_argument('--lr', type=float, default=0.5, help="learning rate")
parser.add_argument('--Q', type=int, default=4, help="number of padding vertices")
parser.add_argument('--mesh_ds_ratio', type=int, default=40, help="the pixel-to-vertex ratio")

parser.add_argument('--naive', type=int, default=0, help="if set True, perform naive orthographic correction")
parser.add_argument('--face_energy', type=float, default=4, help="weight of the face energy term")
parser.add_argument('--similarity', type=int, default=1, help="weight of similarity tranformation constraint")
parser.add_argument('--line_bending', type=float, default=4, help="weight of the line bending term")
parser.add_argument('--regularization', type=float, default=0.5, help="weight of the regularization term")
parser.add_argument('--boundary_constraint', type=float, default=4, help="weight of the mesh boundary constraint")


if __name__ == '__main__':

    # get arguments and input

    args = parser.parse_args()
    dataset = ImageDataset(args)

    print("loading {}".format(args.file))

    _, filename = os.path.split(args.file)
    filename, _ = os.path.splitext(filename)
    image, mesh_uniform_padded, mesh_stereo_padded, correction_strength, seg_mask_padded, box_masks_padded = dataset.get_image_by_file(
        args.file)

    out_dir = "results/{}".format(
        filename)
    os.makedirs(out_dir, exist_ok=True)

    if args.naive:
        trivial_mask = np.ones_like(correction_strength)
        box_masks_padded = trivial_mask[np.newaxis, :, :]
        seg_mask_padded = trivial_mask
        options = {
            "face_energy": 4,
            "similarity": False,
            "line_bending": 0,
            "regularization": 0,
            "boundary_constraint": 0
        }
    else:
        options = {
            "face_energy": args.face_energy,
            "similarity": args.similarity,
            "line_bending": args.line_bending,
            "regularization": args.regularization,
            "boundary_constraint": args.boundary_constraint
        }

    # load the optimization model
    print("loading the optimization model")
    model = Energy(options, mesh_uniform_padded, mesh_stereo_padded, correction_strength, box_masks_padded,
                   seg_mask_padded, args.Q)
    optim = optim.Adam(model.parameters(), lr=args.lr)

    # perform optimization
    print("optimizing")
    for i in range(args.num_iter):
        optim.zero_grad()
        loss = model.forward()
        # print("step {}, loss = {}".format(i, loss.item()))
        loss.backward()
        optim.step()

    # calculate optical flow from the optimized mesh
    print("calculating optical flow")
    H, W, _ = image.shape
    mesh_uniform = mesh_uniform_padded[:, args.Q:-args.Q, args.Q:-args.Q].transpose([1, 2, 0])
    mesh_target = mesh_stereo_padded[:, args.Q:-args.Q, args.Q:-args.Q].transpose([1, 2, 0])
    mesh_optimal = model.mesh.detach().cpu().numpy()
    # mesh_optimal = mesh_target
    mesh_optimal = mesh_optimal[:, args.Q:-args.Q, args.Q:-args.Q].transpose([1, 2, 0])

    flow = mesh_uniform - mesh_optimal

    # warp the input image with the optical flow
    print("warping image")
    map_optimal = cv2.resize(mesh_optimal, (W, H))
    x, y = map_optimal[:, :, 0] + W // 2, map_optimal[:, :, 1] + H // 2
    out = cv2.remap(image, x, y, interpolation=cv2.INTER_LINEAR)

    # output
    cv2.imwrite(os.path.join(out_dir, "{}_input.jpg".format(filename)), image)

    overlay_flow = get_overlay_flow(image[:, :, ::-1], flow, ratio=0.7)
    overlay_flow = (255 * overlay_flow[:, :, ::-1]).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "{}_flow.jpg".format(filename)), overlay_flow)

    cv2.imwrite(os.path.join(out_dir, "{}_output.jpg".format(filename)), out)

    print("results saved in {}".format(out_dir))
