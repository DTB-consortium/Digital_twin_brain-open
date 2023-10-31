# -*- coding: utf-8 -*- 
# @Time : 2023/5/4 16:23 
# @Author : lepold
# @File : draw_brain.py

import numpy as np
from mayavi import mlab

points = []
surfaces = []
file = open("../data/BrainMesh_Ch2withCerebellum.nv", "r")
file.readline()
file.readline()
while True:
    line = file.readline().split()
    if len(line) == 3:
        point = list(map(float, line))
        points.append(point)
    if len(line) == 1:
        break

while True:
    line = file.readline().split()
    if len(line) == 3:
        surface = list(map(int, line))
        surfaces.append(surface)
    else:
        break
file.close()

points = np.array(points)
lines = np.array(surfaces)
scalars = np.full(points.shape[0], np.nan)

# label = np.load("points_label_99.npy")
# mask = np.isin(label, np.array([79,80,81, 82]))
# scalar = scalars[:, 3]
# scalar[~mask] = np.nan

fig = mlab.figure(size=(12, 8), bgcolor=(1, 1, 1))
dx, dy, dz = points[:, 0], points[:, 1], points[:, 2]

cc = ['Accent', 'Blues', 'BrBG', 'BuGn', 'BuPu', 'CMRmap', 'Dark2', 'GnBu', 'Greens', 'Greys', 'OrRd', 'Oranges',
      'PRGn', 'Paired', 'Pastel1', 'Pastel2', 'PiYG', 'PuBu', 'PuBuGn', 'PuOr', 'PuRd', 'Purples', 'RdBu', 'RdGy',
      'RdPu', 'RdYlBu', 'RdYlGn', 'Reds', 'Set1', 'Set2', 'Set3', 'Spectral', 'Vega10', 'Vega20', 'Vega20b', 'Vega20c',
      'Wistia', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot', 'autumn', 'binary', 'black-white', 'blue-red', 'bone',
      'brg', 'bwr', 'cool', 'coolwarm', 'copper', 'cubehelix', 'file', 'flag', 'gist_earth', 'gist_gray', 'gist_heat',
      'gist_ncar', 'gist_rainbow', 'gist_stern', 'gist_yarg', 'gnuplot', 'gnuplot2', 'gray', 'hot', 'hsv', 'inferno',
      'jet', 'magma', 'nipy_spectral', 'ocean', 'pink', 'plasma', 'prism', 'rainbow', 'seismic', 'spectral', 'spring',
      'summer', 'terrain', 'viridis', 'winter']
ll = len(cc)

# for i in np.arange(100, 101, 1):
#     id = np.random.choice(ll, 1)
#     color = cc[int(id)]
#     brainPoints = mlab.points3d(dx, dy, dz, scalars[:, 0], scale_mode='none', scale_factor=.5, colormap='jet')  # scalars[:, 0], scale_mode='none', scale_factor=.5, colormap='jet'
#     # brainSurface = mlab.triangular_mesh(dx, dy, dz, lines - 1, scalars=scalars[:, 3*i], colormap=color)
#     # fig.scene.x_plus_view()
#     # brainSurface.actor.mapper.scalar_visibility = False
#     # brainSurface.actor.property.emissive_factor = np.array([1., 1., 1.])
#     # brainSurface.actor.property.opacity = 0.3294
#     mlab.savefig("./brain_fig/status.png"%i)

# brainPoints = mlab.points3d(dx, dy, dz, mask_points=180, scale_factor=5,
#                             colormap='jet')  # scalars[:, 0], scale_mode='none', scale_factor=.5, colormap='jet'
brainSurface = mlab.triangular_mesh(dx, dy, dz, lines - 1, scale_mode='none', opacity=0.2)  # scalars=scalars,
# brainSurface.module_manager.scalar_lut_manager.lut.nan_color = 1, 1, 1, 0.1
fig.scene.x_plus_view()
brainSurface.actor.mapper.scalar_visibility = False
# brainSurface.actor.property.emissive_factor = np.array([1., 1., 1.])
# brainSurface.actor.property.opacity = 0.3294
# mlab.savefig("../results/brain.tiff")
mlab.show()
