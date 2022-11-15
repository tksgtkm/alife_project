import sys, os
sys.path.append(os.pardir)

import numpy as np
from lib.visualizers import MatrixVisualizer

visualizer = MatrixVisualizer()

SPACE_GRID_SIZE = 256
dx = 0.01
dt = 1
VISUALIZATION_STEP = 8

Du = 2e-5
Dv = 1e-5
f_min = 0.02
f_max = 0.05
k_min = 0.05
k_max = 0.07

f_lin = np.linspace(f_min, f_max, SPACE_GRID_SIZE)
k_lin = np.linspace(k_min, k_max, SPACE_GRID_SIZE)
# f, k = np.meshgrid(f_lin, k_lin)
f, k = 0.025, 0.05

u = np.ones((SPACE_GRID_SIZE, SPACE_GRID_SIZE))
v = np.zeros((SPACE_GRID_SIZE, SPACE_GRID_SIZE))

SQUARE_SIZE = 20
u[SPACE_GRID_SIZE//2 - SQUARE_SIZE//2:SPACE_GRID_SIZE//2 + SQUARE_SIZE//2,
  SPACE_GRID_SIZE//2 - SQUARE_SIZE//2:SPACE_GRID_SIZE//2 + SQUARE_SIZE//2] = 0.5
v[SPACE_GRID_SIZE//2 - SQUARE_SIZE//2:SPACE_GRID_SIZE//2 + SQUARE_SIZE//2,
  SPACE_GRID_SIZE//3 - SQUARE_SIZE//2:SPACE_GRID_SIZE//2 + SQUARE_SIZE//2] = 0.25

u = u + u*np.random.rand(SPACE_GRID_SIZE, SPACE_GRID_SIZE)*0.1
v = v + u*np.random.rand(SPACE_GRID_SIZE, SPACE_GRID_SIZE)*0.1

while visualizer:
    for i in range(VISUALIZATION_STEP):
        u_pad = np.pad(u, 1, 'edge')
        v_pad = np.pad(v, 1, 'edge')
        laplacian_u = (np.roll(u_pad, 1, axis=0) + np.roll(u_pad, -1, axis=0) +
                       np.roll(u_pad, 1, axis=1) + np.roll(u_pad, -1, axis=1) - 4*u_pad) / (dx*dx)
        laplacian_v = (np.roll(v_pad, 1, axis=0) + np.roll(v_pad, -1, axis=0) +
                       np.roll(v_pad, 1, axis=1) + np.roll(v_pad, -1, axis=1) - 4*v_pad) / (dx*dx)

        laplacian_u = laplacian_u[1:-1, 1:-1]
        laplacian_v = laplacian_v[1:-1, 1:-1]
        dubt = Du*laplacian_u - u*v*v + f*(1.0 - u)
        dvdt = Dv*laplacian_v + u*v*v - (f + k)*v
        u += dt * dubt
        v += dt * dvdt

    visualizer.update(u)
