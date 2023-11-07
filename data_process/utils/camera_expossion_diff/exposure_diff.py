import numpy as np
import matplotlib.pyplot as plt

path = "/home/tony/Downloads/info.txt"
exposure_time = np.loadtxt(path)

exposure_time_diff = np.diff(exposure_time)
exposure_time_diff_mean = np.mean(exposure_time_diff) 
print("min",np.min(exposure_time_diff) * 10e-6)
print("max",np.max(exposure_time_diff) * 10e-6)
print("mean",np.mean(exposure_time_diff) * 10e-6)
plt.axhline(y = exposure_time_diff_mean, color = "r", linestyle = "-")
x = np.arange(0,exposure_time_diff.shape[0],1)
plt.plot(x,exposure_time_diff)
plt.show()