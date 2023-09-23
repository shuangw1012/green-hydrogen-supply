#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 14:30:56 2023

@author: admin-shuang
"""

import matplotlib.pyplot as plt
import numpy as np
case1_data = [1121,714,50,294]
case2_data = [1038,662,31,186]
case3_data = [897,572,15,187]
case4_data = [969,618,19,60]
case5_data = [972,620,19,26]
case6_data = [891,568,15,183]
case7_data = [923,589,18,145]

plt.figure(figsize=(12, 8))  # Adjust the figure size if needed

# Plot histogram for case 1
plt.subplot(3, 3, 1)
plt.hist(case1_data, bins=5, color='blue', alpha=0.7)
plt.title('Case 1')

# Plot histogram for case 2
plt.subplot(3, 3, 2)
plt.hist(case2_data, bins=7, color='green', alpha=0.7)
plt.title('Case 2')

# Continue this pattern for the remaining cases (case 3 to case 7)
# ...

# Add overall title
plt.suptitle('Cost Breakdown Histograms')

# Adjust the layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()