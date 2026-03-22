import numpy as np
import matplotlib.pyplot as plt
from taylorDiagram import TaylorDiagram

# Raw Data
raw_ref_std = 18.63

raw_rand_std1 = 18.65
raw_rand_std2 = 18.51
raw_rand_std3 = 18.26

raw_hds_std1 = 18.64
raw_hds_std2 = 18.68
raw_hds_std3 = 18.67

raw_rand_corr1 = 345.86
raw_rand_corr2 = 343.91
raw_rand_corr3 = 337.36

raw_hds_corr1 = 346.25
raw_hds_corr2 = 346.12
raw_hds_corr3 = 346.64

# Normalize Data

norm_ref = raw_ref_std / raw_ref_std	# Reference

norm_rand1 = raw_rand_std1 / raw_ref_std	# Normalized for random sensor placement
norm_rand2 = raw_rand_std2 / raw_ref_std
norm_rand3 = raw_rand_std3 / raw_ref_std

norm_hds1 = raw_hds_std1 / raw_ref_std
norm_hds2 = raw_hds_std2 / raw_ref_std
norm_hds3 = raw_hds_std3 / raw_ref_std

corr_rand1 = raw_rand_corr1/(raw_ref_std*raw_rand_std1)		# Correlation for random
corr_rand2 = raw_rand_corr2/(raw_ref_std*raw_rand_std2)
corr_rand3 = raw_rand_corr3/(raw_ref_std*raw_rand_std3)

corr_hds1 = raw_hds_corr1/(raw_ref_std*raw_hds_std1)		# Correlation for hds
corr_hds2 = raw_hds_corr2/(raw_ref_std*raw_hds_std2)
corr_hds3 = raw_hds_corr3/(raw_ref_std*raw_hds_std3)

print(f"Normalized Reference: {norm_ref:.2f}")
print(f"Normalized Random 1 :    {norm_rand1:.2f}")
print(f"Normalized Random 2 :    {norm_rand2:.2f}")
print(f"Normalized Random 3 :    {norm_rand3:.2f}")
print(f"Normalized HDS 1 :       {norm_hds1:.2f}")
print(f"Normalized HDS 2 :       {norm_hds2:.2f}")
print(f"Normalized HDS 3 :       {norm_hds3:.2f}")

# 3. Setup the Diagram with Reference = 1.0
fig = plt.figure()
dia = TaylorDiagram(norm_ref, fig=fig, label='Reference', srange=(0, 1.5))

# 4. Add "Random" Strategy (Square Marker)
# Correlation is still ~0
dia.add_sample(norm_rand1, corr_rand1, marker='s', ms=10, label='Random1', mfc='blue')
dia.add_sample(norm_rand2, corr_rand2, marker='s', ms=10, label='Random2', mfc='blue')
dia.add_sample(norm_rand3, corr_rand3, marker='s', ms=10, label='Random3', mfc='blue')

# 5. Add "HDS" Strategy (Star Marker)
# Correlation is still ~0.14
dia.add_sample(norm_hds1, corr_hds1, marker='*', ms=10, label='HDS1', mfc='red')
dia.add_sample(norm_hds2, corr_hds2, marker='*', ms=10, label='HDS2', mfc='red')
dia.add_sample(norm_hds3, corr_hds3, marker='*', ms=10, label='HDS3', mfc='red')

# 6. Add contours (RMS Error)
# These curves show the relative error distance from the reference
contours = dia.add_contours(levels=6, colors='0.5')
plt.clabel(contours, inline=1, fontsize=10)

plt.legend(loc='upper right')
plt.title("Normalized Sensor Placement: Random vs. HDS")
plt.show()