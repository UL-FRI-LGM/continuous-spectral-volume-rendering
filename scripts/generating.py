import numpy as np

size = 128
V = np.zeros((size, size, size), dtype=np.uint8)
X, Y, Z = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), np.linspace(-1, 1, size))

# Define the range for each cube and its density
cube1_density = 75  # Density of the first cube
cube2_density = 150  # Density of the second cube

# Define bounds for the first cube
cube1_bounds = np.logical_and(np.maximum(np.abs(X + 0.5), np.maximum(np.abs(Y), np.abs(Z))) < 0.4,
                              np.maximum(np.abs(X), np.maximum(np.abs(Y), np.abs(Z))) < 0.8)

# Define bounds for the second cube
cube2_bounds = np.logical_and(np.maximum(np.abs(X - 0.5), np.maximum(np.abs(Y), np.abs(Z))) < 0.4,
                              np.maximum(np.abs(X), np.maximum(np.abs(Y), np.abs(Z))) < 0.8)

# Apply densities to the specified regions
V[cube1_bounds] += cube1_density
V[cube2_bounds] += cube2_density

# Uncomment to use matplotlib to visualize one slice of the volume
# from matplotlib import pyplot as plt
# plt.imshow(V[size//2, :, :], cmap='gray', vmin=0, vmax=255)
# plt.show()

# Save the volume data to a file
V.tofile('../datasets/two_cubes128.raw')
