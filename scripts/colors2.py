import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def cie1931_wavelength_to_xyz_fit(wavelength):
    wave = wavelength
    x = (0.362 * math.exp(-0.5 * ((wave - 442.0) * (0.0624 if wave < 442.0 else 0.0374)) ** 2)
         + 1.056 * math.exp(-0.5 * ((wave - 599.8) * (0.0264 if wave < 599.8 else 0.0323)) ** 2)
         - 0.065 * math.exp(-0.5 * ((wave - 501.1) * (0.0490 if wave < 501.1 else 0.0382)) ** 2))

    y = (0.821 * math.exp(-0.5 * ((wave - 568.8) * (0.0213 if wave < 568.8 else 0.0247)) ** 2)
         + 0.286 * math.exp(-0.5 * ((wave - 530.9) * (0.0613 if wave < 530.9 else 0.0322)) ** 2))

    z = (1.217 * math.exp(-0.5 * ((wave - 437.0) * (0.0845 if wave < 437.0 else 0.0278)) ** 2)
         + 0.681 * math.exp(-0.5 * ((wave - 459.0) * (0.0385 if wave < 459.0 else 0.0725)) ** 2))

    return [x, y, z]


def srgb_xyz2rgb_postprocess(c):
    if c > 1:
        c = 1
    elif c < 0:
        c = 0

    if c <= 0.0031308:
        return c * 12.92
    else:
        return 1.055 * math.pow(c, 1 / 2.4) - 0.055


def srgb_xyz2rgb(xyz):
    x, y, z = xyz
    rl = 3.2406255 * x - 1.537208 * y - 0.4986286 * z
    gl = -0.9689307 * x + 1.8757561 * y + 0.0415175 * z
    bl = 0.0557101 * x - 0.2040211 * y + 1.0569959 * z

    return [srgb_xyz2rgb_postprocess(rl), srgb_xyz2rgb_postprocess(gl), srgb_xyz2rgb_postprocess(bl)]


def wavelength_to_rgb(wavelength):
    xyz = cie1931_wavelength_to_xyz_fit(wavelength)
    rgb = srgb_xyz2rgb(xyz)

    r = int(rgb[0] * 255)
    g = int(rgb[1] * 255)
    b = int(rgb[2] * 255)

    return [r, g, b]


print(wavelength_to_rgb(500))


def visualize_spectrum():
    wavelengths = np.arange(380, 780, 1)
    colors = []

    for w in wavelengths:
        rgb = wavelength_to_rgb(w)
        colors.append(rgb)

    # Create a figure and a subplot
    fig, ax = plt.subplots(figsize=(10, 2), constrained_layout=True)
    ax.imshow([colors], aspect='auto', extent=[380, 780, 0, 1])
    ax.set_xlabel('Wavelength (nm)')
    ax.set_yticks([])
    ax.set_title('Visible Light Spectrum')

    plt.show()


# visualize_spectrum()

wavelengths = [
    300, 305, 310, 315, 320, 325, 330, 335, 340, 345,
    350, 355, 360, 365, 370, 375, 380, 385, 390, 395,
    400, 405, 410, 415, 420, 425, 430, 435, 440, 445,
    450, 455, 460, 465, 470, 475, 480, 485, 490, 495,
    500, 505, 510, 515, 520, 525, 530, 535, 540, 545,
    550, 555, 560, 565, 570, 575, 580, 585, 590, 595,
    600, 605, 610, 615, 620, 625, 630, 635, 640, 645,
    650, 655, 660, 665, 670, 675, 680, 685, 690, 695,
    700, 705, 710, 715, 720, 725, 730, 735, 740, 745,
    750, 755, 760, 765, 770, 775, 780
]

values = [
    0.00, 0.00, 0.00, 0.00, 0.03, 0.33, 1.60, 4.17, 8.76, 15.61,
    24.24, 31.93, 37.94, 43.65, 47.99, 47.23, 46.01, 49.19, 52.63, 67.11,
    81.45, 85.97, 90.20, 90.96, 91.75, 88.40, 85.08, 93.98, 102.94, 109.14,
    115.49, 116.30, 117.08, 115.89, 114.62, 115.30, 115.97, 112.55, 109.12, 109.48,
    109.82, 109.12, 108.41, 106.93, 105.42, 106.84, 108.25, 106.54, 104.83, 104.57,
    104.29, 102.16, 100.00, 98.03, 96.01, 95.50, 94.97, 91.19, 87.44, 87.84,
    88.27, 87.86, 87.41, 86.16, 84.90, 82.50, 80.13, 80.05, 79.97, 77.97,
    75.97, 75.83, 75.69, 76.44, 77.15, 74.94, 72.77, 68.58, 64.42, 65.04,
    65.62, 66.60, 67.57, 61.51, 55.50, 59.00, 62.46, 64.50, 66.48, 61.13,
    55.83, 48.09, 40.41, 49.09, 57.70, 56.00, 54.30
]


def rejection_sampling(spd_func, x_range, num_samples):
    x_min, x_max = x_range
    x_vals = np.linspace(x_min, x_max, 10000)
    y_vals = spd_func(x_vals)
    y_max = np.max(y_vals)

    samples = []
    while len(samples) < num_samples:
        x_sample = np.random.uniform(x_min, x_max)
        y_sample = np.random.uniform(0, y_max)

        if y_sample < spd_func(x_sample):
            samples.append(x_sample)

    return np.array(samples)


interp_spd = interp1d(wavelengths, values, kind='cubic')
sampled_wavelengths = rejection_sampling(interp_spd, (380, 700), 10000)



# wavelengths = np.random.uniform(380, 700, 1000)
# wavelengths = norm.rvs(loc=555, scale=80, size=1000)

# Calculate the average XYZ by summing contributions and then dividing
total_xyz = np.array([0.0, 0.0, 0.0])
for w in sampled_wavelengths:
    xyz = cie1931_wavelength_to_xyz_fit(w)
    total_xyz += np.array(xyz)

average_xyz = total_xyz / len(sampled_wavelengths)

# Convert the average XYZ to RGB
final_rgb = srgb_xyz2rgb(average_xyz)

### Step 2: Display the Final RGB Color
fig, ax = plt.subplots()
ax.imshow([[final_rgb]], aspect='auto')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(f"Final RGB Color from Averaged Spectrum: {final_rgb}")
plt.show()

# Plotting the sampled wavelengths
plt.hist(sampled_wavelengths, bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Sampled Wavelengths')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Frequency')
plt.show()