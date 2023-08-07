import numpy as np
import time

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
v_orig = v


tik = time.time()
for i in range(1000):
    r = get_r(v)
    phi = get_phi(v)
    theta = get_theta(v)

    v1 = np.multiply(np.sin(n*theta), np.cos(n*phi))
    v2 = np.multiply(np.sin(n*theta), np.sin(n*phi))
    v3 = np.cos(n*theta)
    v = np.array([v1, v2, v3]).transpose()

