#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 09:33:58 2018

@author: group
"""
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats

x = np.array([1.5,2,2.5,3,3.5,4,4.5,5,5.5,6])
y = np.array([10.35,12.3,13,14.0,16,17,18.2,20,20.7,22.5])
gradient, intercept, r_value, p_value, std_err = stats.linregress(x,y)
mn=np.min(x)
mx=np.max(x)
x1=np.linspace(mn,mx,500)
y1=gradient*x1+intercept
print(y1)
plt.plot(x,y,'ob')
plt.plot(x1,y1,'-r')
plt.show()