import io_utils
import time_series
import recursive_p_value
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as stm
import statistic_tests.mkt as mkt
from statistic_tests import wald_wolf
from statistic_tests import run_test

#%%
# src = 'comparable_samples.jl'
# src_type = 'jl'
# input_utility = io_utils.parse_input(src_type, src, 'temp.jl','RP')
# input_utility.parse_jl_file(src, 'ts')
# series_list = []

# %% This can't finish running
# i=0
# for x in input_utility.parse_jl_file(src, 'ts'):
#     series = time_series.time_series(times=x[1], values=x[2], time_labels=x[0])
#     series_list.append(series)
    #plt.plot(series.times,series.values)
    #plt.show()
    #print i
    #i+=1
#%%    Try 52,53, 54, 56, 98, 100, 104 For Choppy
#%%


def beyond_linear(t, x):
    #ts is a time series you want to analyze 
    nonlinear = {"trend" : "up", "oscillation" : "undecided"}

    # x= np.array(series.values)
    # t = np.array(series.times)
    # plt.plot(series.times,series.values)
    # plt.show()

    #Mann-Kendall test
    MK, m, c, pUp = mkt.test(np.array(t), np.array(x), eps=1E-3, alpha=0.05, Ha="up")
    MK, m, c, pDown = mkt.test(np.array(t), np.array(x), eps=1E-3, alpha=0.05, Ha="down")
    # plt.plot(series.times,series.values,'.')
    # plt.plot(series.times,series.values,'-')
    #
    # yHat = m*np.array(series.times)+c
    # plt.plot(series.times, yHat,'-')
    # plt.show()
    
    if pUp <= 0.05 and pDown >0.05 :
        detrended_x = x- (m*np.array(t)+c)
        nonlinear["trend"]="up"
    elif pUp>0.05 and pDown <=0.05:
        detrended_x = x- (m*np.array(t)+c)
        nonlinear["trend"]="down"
    else:
        detrended_x = x-x.mean()
        nonlinear["trend"]="flat"
    
    pAdf = stm.adfuller(detrended_x, maxlag=None, regression='ct', autolag='AIC', store=False, regresults=False)[1]
    fewPass = wald_wolf.runstest(detrended_x, 0.0, passfail = True)
    diff_x = []
    for i in range(len(x)-1):
        diff_x.append(x[i+1]-x[i])
    print "This is the diff"
    print diff_x
    fewRun = wald_wolf.runstest(np.array(diff_x), 0.0, passfail = True)
#    fewRun = run_test.compute_run_test(x) # use the wald_wolf instead
    print 'fewPass:'+str(fewPass) 
    print 'fewRun:'+str(fewRun)
    print 'stationary:'+str(pAdf <= 0.05)
    if (fewPass) and (fewRun) and (pAdf > 0.05): 
        nonlinear["oscillation"]="inconsistent non-choppy"
    elif (not fewPass) and (not fewRun) and (pAdf > 0.05): 
        nonlinear["oscillation"]="inconsistent choppy"
    elif (fewPass) and (fewRun) and (pAdf <= 0.05):
        nonlinear["oscillation"]="consistent non-choppy"
    elif (fewPass) and (not fewRun) and (pAdf <= 0.05):
        nonlinear["oscillation"]="consistent local-choppy"
    elif (fewPass) and (not fewRun) and (pAdf > 0.05):
        nonlinear["oscillation"]="inconsistent local-choppy"
    elif (not fewPass) and (pAdf <= 0.05): 
        nonlinear["oscillation"]="consistent very-choppy"
    else:
        nonlinear["oscillation"]="undecided"

        
    return nonlinear, 0, 0
#%%
#
# l1 = [52,53,54,56,98,100, 104]
# l2 = [4,17,19,21,34,37,38,41,46,49]
# choppy = [52,53,54,68,91,98,4,16,17,21,33,35,38,39,41,43,100,104]
# non_choppy = [51,56,58,59,64,96,11,47,49,105]
# l=np.arange(100,125,1)
# for i in choppy:
#     print i
#     series = series_list[i]
#     print beyond_linear(np.array(series.times), np.array(series.values))
#
# #%%
