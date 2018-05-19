from random import randint, sample, random, normalvariate
import matplotlib.pyplot as plt
import numpy as np
import recursive_p_value
import time_series

def create_line_with_slope(num_points, start_pt, end_pt, slope, intercept, noise_level):

# intercept is the starting value at start_pt    
    interval = (end_pt-start_pt)/(num_points-1)
    points = np.arange(start_pt, end_pt+interval, interval)
    values= intercept + slope*(points-start_pt) 
    noisy_values = values+ np.random.normal(0,1,len(values))/noise_level
    end_val= values[-1]
#    plt.plot(points,values,'.')
#    plt.plot(points,noisy_values,'.')
#    plt.show()
    return points, values, noisy_values, end_val

#%%
points1, values1, noisy_values1, end_val1 = create_line_with_slope(10, 0.1, 0.3, 1.2, 0.25,20)
points2, values2, noisy_values2, end_val2 = create_line_with_slope(10, 0.3, 0.44, -1.5, end_val1,10)
points3, values3, noisy_values3, end_val3 = create_line_with_slope(20, 0.44, 0.70, 2, end_val2,15)
points4, values4, noisy_values4, end_val4 = create_line_with_slope(10, 0.70, 0.99, 0.2, 0.3,20)
#line3 = create_line_with_slope(10, 0.1, 0.3, 1.2, 0.25,10)
#line4 = create_line_with_slope(10, 0.1, 0.3, 1.2, 0.25,10)
points = np.concatenate((points1,points2[1:],points3[1:],points4[1:]), axis=0)
values = np.concatenate((values1,values2[1:],values3[1:],values4[1:]), axis=0) 
noisy_values = np.concatenate((noisy_values1,noisy_values2[1:],noisy_values3[1:], noisy_values4[1:]), axis=0)  

low = noisy_values.min()
high = noisy_values.max()
noisy_values = (noisy_values-low)/(high-low)
values = (values-low)/(high-low)

plt.plot(points,values,'.')
plt.plot(points,noisy_values,'.')
plt.show()

#%%
series = time_series.time_series(points, noisy_values, [])
rp = recursive_p_value.recursive_linear_fit()

