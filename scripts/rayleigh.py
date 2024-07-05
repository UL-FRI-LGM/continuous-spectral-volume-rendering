import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def estimate_scattering(wavelength):
    """
    Estimate the percentage of light scattered based on wavelength using an exponential decay model.

    Parameters:
        wavelength (float): Wavelength in nanometers.

    Returns:
        float: Estimated percentage of light scattered.
    """
    # Coefficients for the exponential model, adjusted to fit the typical Rayleigh scattering pattern
    a = 0.25  # Maximum scattering at shortest wavelength in the visible range
    b = 0.015  # Decay rate adjusted to fit the graph visually

    # Exponential decay function to model the scattering
    return a * np.exp(-b * (wavelength - 380))

# Wavelength range and plotting
wavelengths = np.linspace(380, 700, 320)
scattering_estimates = [estimate_scattering(w) for w in wavelengths]

plt.figure(figsize=(10, 5))
plt.plot(wavelengths, scattering_estimates, color='blue')
plt.title("Estimated Rayleigh Scattering vs. Wavelength")
plt.xlabel("Wavelength (nanometers)")
plt.ylabel("Percent of light scattered")
plt.grid(True)
plt.show()

def integrate_function(func, start=380, end=780):
    result, _ = quad(func, start, end)
    return result

x_integral = integrate_function(estimate_scattering)
print("Integral of xBar over the visible spectrum:", x_integral)