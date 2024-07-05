import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

def gaussian(lambda_val, peak, width):
    return np.exp(-0.5 * ((lambda_val - peak) / width) ** 2)

def gaussianLn(lambda_val, peak, width):
    return np.exp(-0.5 * ((np.log(lambda_val) - np.log(peak)) / width) ** 2)

def xBar(wavelength):
    lambda_val = float(wavelength)
    peak = 595.8
    width = 33.33
    peak2 = 446.8
    width2 = 19.44
    return 1.065 * gaussian(lambda_val, peak, width) + 0.366 * gaussian(lambda_val, peak2, width2)

def yBar(wavelength):
    lambda_val = float(wavelength)
    peak = 556.3
    width = 0.075
    return 1.014 * gaussianLn(lambda_val, peak, width)

def zBar(wavelength):
    lambda_val = float(wavelength)
    peak = 449.8
    width = 0.051
    return 1.839 * gaussianLn(lambda_val, peak, width)

def RGB_to_XYZ(rgb):
    # Inverse gamma correction
    linear_rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    # Transformation matrix from sRGB to XYZ (D65)
    M = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505]
    ])
    return np.dot(M, linear_rgb)


def XYZ_to_RGB(xyz):
    # Transformation matrix from XYZ to linear RGB (D65)
    M = np.array([
        [ 3.2406, -1.5372, -0.4986],
        [-0.9689,  1.8758,  0.0415],
        [ 0.0557, -0.2040,  1.0570]
    ])
    rgb_linear = np.dot(M, xyz)

    # # Gamma correction
    rgb = np.where(rgb_linear <= 0.0031308, 12.92 * rgb_linear, 1.055 * (rgb_linear ** (1 / 2.4)) - 0.055)
    return rgb.clip(0, 1)

def test_wavelength_functions():
    wavelengths = range(380, 780, 10)  # Test wavelengths from 380 to 700 nm, every 10 nm
    results = [(w, xBar(w), yBar(w), zBar(w)) for w in wavelengths]
    for w, x, y, z in results:
        rgb = XYZ_to_RGB(np.array([x, y, z]))
        print(f"Wavelength: {w}; XYZ: {x:.4f},  {y:.4f}, {z:.4f}; RGB: {255*rgb[0]:.4f}, {255*rgb[1]:.4f}, {255*rgb[2]:.4f}")

def visualize_spectrum():
    wavelengths = np.arange(380, 700, 1)
    colors = []

    for w in wavelengths:
        x, y, z = xBar(w), yBar(w), zBar(w)
        rgb = XYZ_to_RGB(np.array([x, y, z]))
        colors.append(rgb)

    # Create a figure and a subplot
    fig, ax = plt.subplots(figsize=(10, 2), constrained_layout=True)
    ax.imshow([colors], aspect='auto', extent=[380, 720, 0, 1])
    ax.set_xlabel('Wavelength (nm)')
    ax.set_yticks([])
    ax.set_title('Visible Light Spectrum')

    plt.show()

# Run the test function
# test_wavelength_functions()

# Run the visualization
visualize_spectrum()


# rgb_sample = np.array([0.95047, 1.0, 1.08883])  # Sample RGB color
# xyz = RGB_to_XYZ(rgb_sample)
# rgb_converted_back = XYZ_to_RGB(xyz)

# print("Original RGB:", rgb_sample)
xyz=np.array([0.24860, 0.13255, 0.04863])
to_rgb = XYZ_to_RGB(xyz)
print("XYZ:", xyz)
print("to RGB:", f"{255*to_rgb[0]:.4f}, {255*to_rgb[1]:.4f}, {255*to_rgb[2]:.4f}")

def integrate_function(func, start=380, end=780):
    result, _ = quad(func, start, end)
    return result

x_integral = integrate_function(xBar)
print("Integral of xBar over the visible spectrum:", x_integral)

y_integral = integrate_function(yBar)
print("Integral of yBar over the visible spectrum:", y_integral)

z_integral = integrate_function(zBar)
print("Integral of zBar over the visible spectrum:", z_integral)

# Generate wavelengths using rejection sampling based on the specified SPD
# wavelengths = rejection_sampling(spd_function, (380, 700), 1000)

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


# sampled_waves = [450]
# total_xyz = np.array([0.0, 0.0, 0.0])
# for wl in sampled_waves:
#     x = xBar(wl)
#     y = yBar(wl)
#     z = zBar(wl)
#     total_xyz += np.array([x, y, z])
#
# average_xyz = total_xyz / len(sampled_waves)
#
# final_rgb = XYZ_to_RGB(average_xyz)
#
# ### Step 2: Display the Final RGB Color
# fig, ax = plt.subplots()
# ax.imshow([[average_xyz]], aspect='auto')
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_title(f"Handpicking white: {average_xyz}")
# plt.show()

interp_spd = interp1d(wavelengths, values, kind='cubic')
sampled_wavelengths = rejection_sampling(interp_spd, (380, 700), 10000)
# sampled_wavelengths = np.random.uniform(500, 600, 1000)

# Plotting the sampled wavelengths
plt.hist(sampled_wavelengths, bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Sampled Wavelengths')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Frequency')
plt.show()


# Calculate the average XYZ by summing contributions and then dividing
total_xyz = np.array([0.0, 0.0, 0.0])
for wl in sampled_wavelengths:
    x = xBar(wl)
    y = yBar(wl)
    z = zBar(wl)
    total_xyz += np.array([x, y, z])

average_xyz = total_xyz / len(sampled_wavelengths)

# Convert the average XYZ to RGB
Y_average = average_xyz[1]
print(Y_average)
scaling_factor = 0.33
scaling_factor = 1.000 / Y_average
final_rgb = XYZ_to_RGB(average_xyz)

### Step 2: Display the Final RGB Color
fig, ax = plt.subplots()
ax.imshow([[average_xyz]], aspect='auto')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(f"Final XYZ Averaged Spectrum: {average_xyz}")
plt.show()

fig, ax = plt.subplots()
ax.imshow([[final_rgb]], aspect='auto')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(f"Final RGB Color from Averaged Spectrum: {final_rgb}")
plt.show()


