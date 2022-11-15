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
f, k = 0.025, 0.08

u = np.ones((SPACE_GRID_SIZE, SPACE_GRID_SIZE))
v = np.zeros((SPACE_GRID_SIZE, SPACE_GRID_SIZE))

SQUARE_SIZE = 20
u[SPACE_GRID_SIZE//2 - SQUARE_SIZE//2:SPACE_GRID_SIZE//2 + SQUARE_SIZE//2,
  SPACE_GRID_SIZE//2 - SQUARE_SIZE//2:SPACE_GRID_SIZE//2 + SQUARE_SIZE//2] = 0.5
v[SPACE_GRID_SIZE//2 - SQUARE_SIZE//2:SPACE_GRID_SIZE//2 + SQUARE_SIZE//2,
  SPACE_GRID_SIZE//3 - SQUARE_SIZE//2:SPACE_GRID_SIZE//2 + SQUARE_SIZE//2] = 0.25

u += np.random.rand(SPACE_GRID_SIZE, SPACE_GRID_SIZE)*0.1
v += np.random.rand(SPACE_GRID_SIZE, SPACE_GRID_SIZE)*0.1

while visualizer:
    for i in range(VISUALIZATION_STEP):
        laplacian_u = (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
                       np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4*u) / (dx*dx)
        laplacian_v = (np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) +
                       np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 4*v) / (dx*dx)

        dubt = Du*laplacian_u
        dvdt = Dv*laplacian_v
        u += dt * dubt
        v += dt * dvdt

    visualizer.update(u)
