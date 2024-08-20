import numpy as np
from scipy.stats import ks_2samp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Define the data sets
theta_1 = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70])
y_1 = np.array([0.6917045, 0.33001, 0.205897667, 0.14704275, 
                0.0980285, 0.095246833, 0.081935367, 0.07583225, 0.067779889])

theta_2 = np.array([15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
y_2 = np.array([0.45209475, 0.2542795, 0.169510333, 0.125202556, 
                0.102404333, 0.08334225, 0.079912367, 0.067883733, 
                0.064169133, 0.062593639])
				
theta1_err = np.array([0.288675135, 0.288675135, 0.288675135, 0.288675135, 0.288675135, 0.288675135, 0.288675135, 0.288675135, 0.288675135])
theta2_err = np.array([0.288675135, 0.288675135, 0.288675135, 0.288675135, 0.288675135, 0.288675135, 0.288675135, 0.288675135, 0.288675135, 0.288675135])

y1_err = np.array([0.010004169, 0.002401446, 0.001016545, 0.000371317, 0.000174308, 0.000209181, 0.000160097, 8.89887E-05, 0.000130036])
y2_err = np.array([0.003306689, 0.001265001, 0.000453508, 0.00023662, 0.000243005, 0.000179936, 0.000117251 ,0.000135099, 0.000113491, 7.4294E-05])




best_offset = None
best_p_value = 0

# Store results for the best offset
best_y1 = None
best_y2 = None

offset_pval = []
offsets = []

# Test different offsets from -20 to 20
for offset in np.arange(-20, 31, 0.01):
    # Define the interpolation function for y_2
    interp_y2 = interp1d(theta_2 + offset, y_2, bounds_error=False, fill_value="extrapolate")

    # Interpolate y_2 at the theta_1 points
    interpolated_y2 = interp_y2(theta_1)
    
    a = np.column_stack((theta_1, y_1)) # your x
    b = np.column_stack((theta_1, interpolated_y2)) # your y
    # Perform the K-S test
    ks_stat, p_value = ks_2samp(y_1, interpolated_y2)
    mse = (((a-b)/np.sqrt (y1_err.mean()**2 + y2_err.mean()**2))**2).mean(axis=1)
    offset_pval.append (mse [1])
    offsets.append (offset)
    
    # Check if this is the best p-value so far
    if p_value > best_p_value:
        best_p_value = p_value
        best_offset = offset
        best_y1 = y_1
        best_y2 = interpolated_y2
best_offset = offsets [offset_pval.index (min (offset_pval))]

print(f"Best theta offset: {best_offset} with MSE of {min (offset_pval)}")
print(f"Center is at: {best_offset/2}")
print(f"Best K-S test p-value: {best_p_value}")

# Plot y(theta) for offset = 0
plt.figure(figsize=(14, 7))

plt.subplot(1, 3, 1)
plt.plot(theta_1, y_1, 'bo-', label='Set 1', zorder=1)
plt.errorbar(theta_1, y_1, yerr=y1_err, xerr=theta1_err, fmt = "." ,color='c', zorder=5)
plt.plot(theta_2, y_2, 'ro-', label='Set 2 (offset=0)', zorder=1)
plt.errorbar(theta_2, y_2, yerr=y2_err, xerr=theta2_err, fmt = "." ,color='c', zorder=5)
plt.xlabel(r'$\theta$ (degrees)')
plt.ylabel('Count')
plt.title(r'Count($\theta$) for offset = 0')
plt.legend()
plt.text(0.09, 0.91, 'a.', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', weight='bold')

# Plot y(theta) for the best offset
plt.subplot(1, 3, 2)
plt.plot(theta_1, y_1, 'bo-', label='Set 1', zorder=1)
plt.errorbar(theta_1, y_1, yerr=y1_err, xerr=theta1_err, fmt = "." ,color='c', zorder=5)
plt.plot(theta_2 + best_offset, y_2, 'ro-', label=f'Set 2 (offset={best_offset})', zorder=1)
plt.errorbar(theta_2 + best_offset, y_2, yerr=y2_err, xerr=theta2_err, fmt = "." ,color='c', zorder=5)
# plt.plot(theta_1, best_y2, 'go-', label='Set 2 (Kolmagorov})')
plt.xlabel(r'$\theta$ (degrees)')
plt.ylabel('Count')
plt.title(fr'Count($\theta$) for optimal offset = {best_offset}')
plt.legend()
plt.text(0.09, 0.91, 'b', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', weight='bold')

plt.tight_layout()

plt.subplot(1, 3, 3)
plt.plot(np.arange(-20, 31, 0.01), (np.array([offset_pval]).T), 'm-', label='Difference (Set 1 - Set 2)')
plt.xlabel("Offset (degrees)")
plt.ylabel('MSE')
plt.title('MSE')

plt.legend()
plt.text(0.09, 0.91, 'c.', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', weight='bold')

plt.tight_layout()
plt.savefig ("S:\\nitzan\\docs\\לימודים\\תואר ראשון\\תשפד\\LabC\\compton\\MSE.eps" , format='eps')
plt.show()

# Plot the CDFs for the best offset
# plt.figure(figsize=(14, 7))

# plt.subplot(1, 2, 1)
# plt.plot(np.sort(y_1), np.linspace(0, 1, len(y_1)), 'b-', label='CDF Set 1')
# plt.plot(np.sort(y_2), np.linspace(0, 1, len(y_2)), 'r-', label='CDF Set 2 (offset=0)')
# plt.xlabel('y')
# plt.ylabel('CDF')
# plt.title('CDFs for offset = 0')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(np.sort(best_y1), np.linspace(0, 1, len(best_y1)), 'b-', label='CDF Set 1')
# plt.plot(np.sort(best_y2), np.linspace(0, 1, len(best_y2)), 'g-', label=f'CDF Set 2 (offset={best_offset})')
# plt.xlabel('y')
# plt.ylabel('CDF')
# plt.title(f'CDFs for optimal offset = {best_offset}')
# plt.legend()

# plt.tight_layout()
# plt.show()