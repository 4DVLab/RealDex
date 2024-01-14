import pybullet as p

# Start PyBullet in direct mode (no GUI)
p.connect(p.DIRECT)

# Define input and output file paths
input_obj = "fixed.obj"
output_convex_hulls_obj = "fixed_vhacd.obj"
log_file = "vhacd_log.txt"

# Perform VHACD
p.vhacd(input_obj, output_convex_hulls_obj, log_file, alpha=0.04, resolution=100000)

# Disconnect PyBullet
p.disconnect()
