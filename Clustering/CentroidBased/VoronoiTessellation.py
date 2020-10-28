import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
vor = Voronoi(points)
voronoi_plot_2d(vor)
plt.show()

# The Voronoi vertices
print(vor.vertices)
# There is a single finite Voronoi region, and four finite Voronoi ridges
print(vor.regions)
# The ridges are perpendicular between lines drawn between the following input points
print(vor.ridge_points)