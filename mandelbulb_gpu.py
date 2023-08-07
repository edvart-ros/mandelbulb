import torch
import time
import open3d as o3d
import numpy as np
import copy

def get_r(v):
    return torch.norm(v, dim=-1)

def get_phi(v):
    return torch.atan2(v[:,1], v[:,0])

def get_theta(v):
    return torch.acos(v[:,2] / get_r(v))


n = 51
threshold = 1+1e-8
x = torch.linspace(-threshold, threshold, 200, device='cuda')
y = x.clone()
z = x.clone()
x, y, z = torch.meshgrid(x, y, z, indexing='xy')

C = torch.zeros((x.numel(), 3), device='cuda')
C[:, 0] = x.flatten()
C[:, 1] = y.flatten()
C[:, 2] = z.flatten()

#the points within radius 1 (ball of points)
C = C[torch.sqrt(C[:,0]**2 + C[:,1]**2 + C[:,2]**2) <= 1.0, :]
v = C.clone()
    
for i in range(100):
    r = get_r(v)
    diverging = r > threshold

    phi = get_phi(v)
    theta = get_theta(v)

    r_n = torch.pow(r, n)
    v1 = r_n * torch.sin(n*theta) * torch.cos(n*phi) + C[:,0]
    v2 = r_n * torch.sin(n*theta) * torch.sin(n*phi) + C[:,1]
    v3 = r_n * torch.cos(n*theta) + C[:,2]
    
    v_new = torch.stack([v1, v2, v3]).transpose(0, 1)
    v = torch.where(diverging.view(-1, 1), v, v_new)

v = v.cpu().numpy()
norm_result = np.linalg.norm(v, axis=-1)
idxs = np.where(norm_result<=threshold)

w = v[idxs]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(w)
pcd.estimate_normals()
o3d.visualization.draw_geometries([pcd])
