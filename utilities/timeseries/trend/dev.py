import io_utils
import trend_analysis
import recursive_p_value
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import linear_fit

src = 'ready_to_compare.jl'
src_type = 'jl'
input_utility = io_utils.parse_input(src_type, src, 'temp.jl')
input_utility.parse_jl_file(src, 'ts')
series_list = []

rfe = recursive_p_value.recursive_linear_fit_error()
# %% This can't finish running
i = 0
for x in input_utility.parse_jl_file(src, 'ts'):
    series = trend_analysis.time_series(times=x[1], values=x[2], time_labels=x[0])
    series_list.append(series)

# %%
# series_list = series_list[0:500]
# %%
# prev_result is dictionary
series_p_lines = []
i = 0
for series in series_list:
    series_p_lines.append((series, rfe.get_lines(series)[1]))
    print i
    i = i + 1

series_p_lines = np.array(series_p_lines)
# %%

series_ave_lines = []
i = 0
for series in series_list:
    lines = [[series.times_labels[0], series.times_labels[-1], 0, np.array(series.values).mean(),
              series.times[0], series.times[-1]]]
    series_ave_lines.append((series, lines))
    print i
    i = i + 1

series_ave_lines = np.array(series_ave_lines)
# %%
lf = linear_fit.linear_fit()
i = 0
series_pwlf_lines = []
kept = []
for series in series_list:
    pwlf_sample = lf.analyze_series_with_points(series)
    # print pwlf_sample
    try:
        pwlf_lines = input_utility.get_line_segment(pwlf_sample, series.times_labels, series.times)
        series_pwlf_lines.append((series, pwlf_lines))
        print i
        kept.append(i)
    except:
        continue
    i = i + 1

series_pwlf_lines = np.array(series_pwlf_lines)


# %%

# %%
class error:
    def __init__(self):
        self.initial = 0

    def rSqMse(self, yHat, y):
        yBar = y.mean()
        SST = ((y - yBar) ** 2).sum()
        SSE = float(((y - yHat) ** 2).sum())
        #        SSR = float(((yBar - yHat) ** 2).sum())
        MSE = SSE / float(len(y))
        if SST > 0:
            return [1 - SSE / SST, MSE]
        elif SST == 0:
            return [1, MSE]

    def analyze_error(self, series, lines):
        rSquare = []
        MSE = []
        length = []
        np_times = np.array(series.times)
        np_values = np.array(series.values)
        tot_length = len(np_times)
        for i in range(len(lines)):
            line = lines[i]
            # for line in lines:

            xVal = np_times[(np_times >= line[4]) & (np_times <= line[5])]
            yVal = np_values[(np_times >= line[4]) & (np_times <= line[5])]
            yHat = line[2] * xVal + line[3]
            temp = self.rSqMse(yHat, yVal)
            rSquare.append(temp[0])
            MSE.append(temp[1])
            if i > 0 and i < len(lines) - 1:
                length.append((len(xVal) - 1) / float(tot_length))
            elif len(lines) == 1:
                length.append((len(xVal)) / float(tot_length))
            else:
                length.append((len(xVal) - 0.5) / float(tot_length))

        return np.array(MSE), np.array(rSquare), np.array(length)

    def metrices(self, series, lines, option, power):
        MSE, rSquare, length = self.analyze_error(series, lines)
        if option == 1:
            # power = 1.50
            return (rSquare * (length ** power)).sum()
        if option == 2:
            # power = 0
            return (MSE * (length ** power)).sum()
        if option == 3:
            # power = 3
            return (MSE / (length ** power)).sum()
        if option == 4:
            # power = 0.2
            return (MSE / (0.1 + length ** power)).sum()
        if option == 5:
            # power = 0.5
            return (rSquare * (np.exp(length ** power))).sum()
        if option == 6:
            return (length ** 2).sum()
        if option == 7:
            return len(length)


# %%

class bootstrap:
    def __init__(self):
        self.initial = 0

    # values is numpy array
    def resample(self, values):
        N = len(values)
        samples = np.random.choice(values, (N ** 5, N), replace=True)
        return samples

    def distances(self, values):
        samples1 = self.resample(values)
        samples2 = self.resample(values)
        samples3 = self.resample(values)
        return np.linalg.norm(samples1 - values, axis=1), np.linalg.norm(samples2 - samples3, axis=1)

    def p_dist(self, values):
        dist1, dist2 = self.distances(values)
        return 1 - binom.cdf((dist1 > dist2).sum(), len(dist1), 0.5)


# %%
def filter(a):
    a = a[~np.isnan(a)]
    a = a[~np.isinf(a)]
    return a


# %%

def get_error(series_lines, option, power):
    error_list = []
    e = error()
    i = 0
    for sl in series_lines:

        try:
            error_list.append(e.metrices(sl[0], sl[1], option, power))

        except:
            print "there was error"
            print i

        i += 1
    return np.array(error_list)


# %%
series_p_lines = series_p_lines[kept]

# %%
series_ave_lines = series_ave_lines[kept]
# %%
opt = 7
power = 1.25
y1 = get_error(series_p_lines, opt, power)
y2 = get_error(series_pwlf_lines, opt, power)
y3 = get_error(series_ave_lines, opt, power)

# %%
rel = y1 / y2
rel = rel[~np.isnan(rel)]
win_rate = (rel > 1).sum() / float(len(rel))
# plt.hist(rel)
plt.hist(rel[(-1 < rel) & (rel < 2)], bins=np.arange(-1, 2, 0.25))
plt.xlabel('relative performance')
plt.ylabel('number of seires')
plt.title('relative performance: p-algo over pwlf-algo')
# plt.title('relative performance: p-algo over const-algo')
print win_rate
# %%
rel = y1 / y3
rel = rel[~np.isnan(rel)]
win_rate = (rel > 1).sum() / float(len(rel))
# plt.hist(rel)
plt.hist(rel[(-1 < rel) & (rel < 2)], bins=np.arange(-1, 2, 0.25))
plt.xlabel('relative performance')
plt.ylabel('number of seires')
# plt.title('relative performance: p-algo over pwlf-algo')
plt.title('relative performance: p-algo over const-algo')
print win_rate


# %%

def plot_sl(series, lines):
    xHat = np.linspace(min(series.times), max(series.times), num=100)
    yHat = rfe.predict(xHat, lines)
    plt.plot(series.times, series.values)
    plt.plot(xHat, yHat)
    plt.show()


# %%
# %%
# plot two histograms

# y1 = np.random.normal(-2, 2, 1000)
# y2 = np.random.normal(2, 2, 5000)

def plot_hists(y1, y2):
    colors = ['b', 'g']

    # plots the histogram
    fig, ax1 = plt.subplots()
    ax1.hist([y1, y2], color=colors)
    ax1.set_xlim(0, 1)
    ax1.set_ylabel("Count")
    plt.tight_layout()
    plt.show()
    # %%