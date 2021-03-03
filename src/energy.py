import torch
import torch.nn as nn


class Energy(nn.Module):

    def __init__(self, options, source_mesh, target_mesh, correction_strength, box_masks, face_mask, Q, dtype=torch.float32):
        """
        :param target_mesh:  torch.tensor, 2 x Hm x Wm
        :param correction_strength:  torch.tensor, Hm x Wm
        :param box_masks: torch.tensor, K x Hm x Wm
        :param face_mask: torch.tensor, Hm x Hw
        K: number of faces
        """
        super(Energy, self).__init__()

        self.opt = options

        self.source_mesh = torch.tensor(source_mesh, dtype=dtype)
        self.target_mesh = torch.tensor(target_mesh, dtype=dtype)
        self.mesh = torch.tensor(source_mesh, dtype=dtype)
        self.correction_strength = torch.tensor(correction_strength, dtype=dtype)
        self.box_masks = torch.tensor(box_masks, dtype=dtype)
        self.face_mask = torch.tensor(face_mask, dtype=dtype)

        self.Hm = self.target_mesh.size(1)
        self.Wm = self.target_mesh.size(2)
        self.K = self.box_masks.size(0)
        self.Q = Q

        self.similarity = torch.tensor([1., 0.], dtype=dtype).unsqueeze(0).repeat(self.K, 1)
        self.translation = torch.zeros([self.K, 2], dtype=dtype)

        self.source_mesh = nn.Parameter(self.source_mesh, requires_grad=False)
        self.target_mesh = nn.Parameter(self.target_mesh, requires_grad=False)
        self.mesh = nn.Parameter(self.mesh, requires_grad=True)
        self.correction_strength = nn.Parameter(self.correction_strength, requires_grad=False)
        self.box_masks = nn.Parameter(self.box_masks, requires_grad=False)
        self.face_mask = nn.Parameter(self.face_mask, requires_grad=False)
        self.similarity = nn.Parameter(self.similarity, requires_grad=True)
        self.translation = nn.Parameter(self.translation, requires_grad=True)


    def get_similarity_matrices(self):

        similarity_matrices = torch.zeros([self.K, 2, 2], dtype=self.similarity.dtype, device=self.similarity.device)
        similarity_matrices[:, 0, 0] = self.similarity[:, 0]
        similarity_matrices[:, 0, 1] = self.similarity[:, 1]
        similarity_matrices[:, 1, 0] = -self.similarity[:, 1]
        similarity_matrices[:, 1, 1] = self.similarity[:, 0]
        return  similarity_matrices

    def forward(self):

        # face energy term
        if self.opt["similarity"]:
            similarity_matrices = self.get_similarity_matrices()
            transformed_target = torch.matmul(similarity_matrices, self.target_mesh.view(2, self.Hm * self.Wm))
            transformed_target = transformed_target.view(self.K, 2, self.Hm, self.Wm)
            transformed_target = transformed_target + self.translation.view(self.K, 2, 1, 1)
            target_mesh = transformed_target
        else:
            target_mesh = self.target_mesh

        masks = self.box_masks.view(self.K, 1, self.Hm, self.Wm) * self.face_mask.view(1, 1, self.Hm, self.Wm)
        face_energy = masks * self.correction_strength * (self.mesh - target_mesh) ** 2
        similarity_regularizer = (self.similarity[:, 0] - 1.) ** 2
        face_energy = face_energy.mean()
        if self.opt["similarity"]: face_energy += 1e4 * similarity_regularizer.mean()

        # boundary constraint
        left = (self.mesh[0, self.Q:self.Hm-self.Q, self.Q] -
                self.source_mesh[0, self.Q:self.Hm - self.Q, self.Q]) ** 2
        right = (self.mesh[0, self.Q:self.Hm - self.Q, self.Wm - self.Q - 1] -
                 self.source_mesh[0, self.Q:self.Hm - self.Q, self.Wm - self.Q - 1]) ** 2
        top = (self.mesh[1, self.Q, self.Q:self.Wm-self.Q] -
               self.source_mesh[1, self.Q, self.Q:self.Wm - self.Q]) ** 2
        bottom = (self.mesh[1, self.Hm - self.Q - 1, self.Q:self.Wm-self.Q] -
                  self.source_mesh[1, self.Hm - self.Q - 1, self.Q:self.Wm - self.Q]) ** 2
        boundary_constraint = left.mean() + right.mean() + top.mean() + bottom.mean()

        # Line-Bending Term
        coordinate_padding_u = torch.zeros_like(self.correction_strength[1:, :]).unsqueeze(0)
        mesh_diff_u = self.mesh[:, 1:, :] - self.mesh[:, :-1, :]
        mesh_diff_u = torch.cat((mesh_diff_u, coordinate_padding_u), dim=0)
        source_mesh_diff_u = self.source_mesh[:, 1:, :] - self.source_mesh[:, :-1, :]
        unit_source_mesh_diff_u = source_mesh_diff_u / torch.norm(source_mesh_diff_u, dim=0).unsqueeze(0)
        unit_source_mesh_diff_u = torch.cat((unit_source_mesh_diff_u, coordinate_padding_u), dim=0)
        line_bending_u_loss = torch.square(torch.norm(torch.cross(mesh_diff_u, unit_source_mesh_diff_u, dim=0), dim=0))

        coordinate_padding_v = torch.zeros_like(self.correction_strength[:, 1:]).unsqueeze(0)
        mesh_diff_v = self.mesh[:, :, 1:] - self.mesh[:, :, :-1]
        mesh_diff_v = torch.cat((mesh_diff_v, coordinate_padding_v), dim=0)
        source_mesh_diff_v = self.source_mesh[:, :, 1:] - self.source_mesh[:, :, :-1]
        unit_source_mesh_diff_v = source_mesh_diff_v / torch.norm(source_mesh_diff_v, dim=0).unsqueeze(0)
        unit_source_mesh_diff_v = torch.cat((unit_source_mesh_diff_v, coordinate_padding_v), dim=0)
        line_bending_v_loss = torch.square(torch.norm(torch.cross(mesh_diff_v, unit_source_mesh_diff_v, dim=0), dim=0))

        line_bending_term = (line_bending_u_loss.mean() + line_bending_v_loss.mean()) / 2

        # Regularization Term
        regularization_u_loss = torch.square(torch.norm(mesh_diff_u, dim=0))
        regularization_v_loss = torch.square(torch.norm(mesh_diff_v, dim=0))
        regularization_term = (regularization_u_loss.mean() + regularization_v_loss.mean()) / 2

        energy = self.opt["face_energy"] * face_energy \
                 + self.opt["line_bending"] * line_bending_term \
                 + self.opt["regularization"] * regularization_term \
                 + self.opt["boundary_constraint"] * boundary_constraint

        return energy


def _unit_test():

    import matplotlib.pyplot as plt
    import torch.optim as optim
    import numpy as np

    Hm = 30
    Wm = 40
    Q = 4

    x = torch.arange(Wm + 2 * Q) - (Wm / 2 + Q)
    y = torch.arange(Hm + 2 * Q) - (Hm / 2 + Q)
    y, x = torch.meshgrid(y, x)
    uniform_mesh = torch.stack([x, y], dim=0)
    target_mesh = uniform_mesh * 1.1

    trivial_mask = torch.ones([Hm + 2 * Q, Wm + 2 * Q])

    options = {
        "face_energy": 4,
        "similarity": True,
        "line_bending": 0.5,
        "regularization": 2,
        "boundary_constraint": 2
    }

    model = Energy(options, uniform_mesh, target_mesh, trivial_mask, trivial_mask.unsqueeze(0), trivial_mask, Q)
    optimizer = optim.SGD(model.parameters(), lr=1e-7)

    for i in range(10):
        optimizer.zero_grad()
        loss = model.forward()
        print(i, loss)
        loss.backward()
        optimizer.step()

    diff = (model.mesh - uniform_mesh).detach().numpy().transpose([1, 2, 0])

    h, w, _ = diff.shape
    X = np.arange(0, w, 1).astype(np.float32)
    Y = np.arange(0, h, 1).astype(np.float32)
    X, Y = np.meshgrid(X, Y)
    plt.quiver(X, Y, diff[:, :, 0], diff[:, :, 1])
    plt.show()

    plt.imshow(diff[:, :, 0])
    plt.show()
    plt.imshow(diff[:, :, 1])
    plt.show()




if __name__ == "__main__":
    _unit_test()