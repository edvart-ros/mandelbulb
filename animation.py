import time
import numpy as np
import open3d as o3d

def get_r(v):
    return np.linalg.norm(v, axis=-1)

def get_phi(v):
    return np.arctan(v[:,1], v[:,0])

def get_theta(v):
    return np.arccos(np.divide(v[:,2], get_r(v)))


n = 3
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
z = np.linspace(-1, 1, 100)
x, y, z = np.meshgrid(x, y, z)
v = np.zeros((x.size, 3))
v[:, 0] = x.ravel()
v[:, 1] = y.ravel()
v[:, 2] = z.ravel()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(v)

r = get_r(v)
phi = get_phi(v)
theta = get_theta(v)

w1 = np.multiply(np.sin(n*theta), np.cos(n*phi))
w2 = np.multiply(np.sin(n*theta),np.sin(n*phi))
w3 = np.cos(n*theta)
w = np.array([w1, w2, w3]).transpose()

diff = w-v

frames = []
frames.append(v)

for i in np.arange(0, 1, 0.002):
    v = (v + diff*i)/np.max(np.linalg.norm(v, axis=-1))
    frames.append(v)
    print(i)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)

for frame in frames:
    pcd.points = o3d.utility.Vector3dVector(v)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
