import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def max_scattering_angle_cosine(wavelength):
    normalized = (wavelength - 400.0) / 300.0  # Normalize to 0-1 range
    max_angle_degrees = 10.0 + 10.0 * (1.0 - normalized)  # angle ranges from 10 to 20 degrees
    max_angle_radians = np.radians(max_angle_degrees)
    return np.cos(max_angle_radians)

def sample_custom_angle_cosine(wavelength):
    cos_theta_max = max_scattering_angle_cosine(wavelength)
    rand = np.random.uniform()  # Random value between 0 and 1
    return cos_theta_max + (1.0 - cos_theta_max) * rand

def scatter_direction(wavelength, incoming_direction):
    cos_theta = sample_custom_angle_cosine(wavelength)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = np.pi / 2  # All scattering to the right, varying only in the up-down direction slightly

    w = incoming_direction
    u = np.cross([0, 1, 0], w)
    u = u / np.linalg.norm(u)
    v = np.cross(w, u)

    scatter_dir = sin_theta * np.cos(phi) * u + sin_theta * np.sin(phi) * v + cos_theta * w
    return scatter_dir

# Define the wavelength range and incoming direction
incoming_dir = np.array([1, 0, 0])  # Light coming from the x-direction
wavelengths = np.linspace(380, 720, 100)
directions = np.array([scatter_direction(w, incoming_dir) for w in wavelengths])

# Plotting
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Assign colors based on wavelength ranges
colors = ['violet', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red']
color_bins = [380, 440, 490, 510, 580, 645, 700, 721]  # Note the last bin includes 720 now
color_indices = np.digitize(wavelengths, color_bins) - 1

for i, dir in enumerate(directions):
    ax.quiver(0, 0, 0, dir[0], dir[1], dir[2], color=colors[color_indices[i]], length=0.1, linewidth=0.5)

ax.set_xlim([-0.1, 0.1])
ax.set_ylim([-0.1, 0.1])
ax.set_zlim([-0.1, 0.1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Directional Scattering by Wavelength')
plt.show()
