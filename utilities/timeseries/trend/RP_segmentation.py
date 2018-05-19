import logging
import matplotlib.pyplot as plt

from interpret import descriptor
import numpy as np
import statsmodels.api as sm
from scipy import stats
from interpret import visualization

from statistic_tests import choppy_test as choppy


class line:
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept


class recursive_linear_fit:
    stop_threshold = 0.025
    variance_threshold = 0.05

    def __init__(self):
        self.initial = 0

    # a better version
    def pValue(self, X, Y, alphaOld, betaOld):
        n = len(X)
        Xc = sm.add_constant(X)
        line = sm.OLS(Y, Xc).fit()
        alpha, beta = line.params[0], line.params[1]

        yHat = line.predict(Xc)

        xBar = X.mean()
        yBar = Y.mean()
        Sxx = ((X - xBar) ** 2).sum()
        Sqx = (X ** 2).sum()
        Sxy = ((X - xBar) * (Y - yBar)).sum()
        Syy = ((Y - yBar) ** 2).sum()
        sigmaSq = ((Y - yHat) ** 2).sum()
        S = np.sqrt(sigmaSq / (n - 2))
        beta_diff = abs(beta - betaOld)
        alpha_diff = abs(alpha - alphaOld)
        if beta_diff > 0.0001 and S > 0.0001:
            betaStat = abs((beta - betaOld) * np.sqrt(Sxx) / S)
            p_beta = stats.t.sf(betaStat, n - 2) * 2
        elif beta_diff * np.sqrt(Sxx) < 0.001 * S:
            p_beta = 0.5
        elif beta_diff * np.sqrt(Sxx) > 1000 * S:
            p_beta = 0.0
        else:
            p_beta = 1.0

        if alpha_diff > 0.0001 and S > 0.0001:
            alphaStat = abs((alpha - alphaOld) / (S * np.sqrt(Sqx / (n * Sxx))))
            p_alpha = stats.t.sf(alphaStat, n - 2) * 2
        elif alpha_diff * np.sqrt(Sxx) < 0.001 * S:
            p_alpha = 0.5
        elif beta_diff * np.sqrt(Sxx) > 1000 * S:
            p_alpha = 0.0
        else:
            p_alpha = 1.0

        return alpha, beta, p_alpha, p_beta

    def compute_p_value(self, x, y, prev_slope, prev_intercept):
        alpha, beta, p_alpha, p_beta = self.pValue(np.array(x), np.array(y), prev_intercept, prev_slope)
        left_line = line(beta, alpha)
        return left_line, [p_alpha, p_beta]

    def bottom_up_merge(self, line_segments, x, y, x_labels):
        print 'inside bottom up'
        candidates_inds = []
        for i in range(len(line_segments) - 1):
            start_x = x.index(line_segments[i][4])
            mid_x = x.index(line_segments[i][5])
            print 'before error'
            print line_segments[i+1]
            end_x = x.index(line_segments[i+1][5])
            tmp_left = self.compute_p_value(x[start_x:end_x], y[start_x:end_x], line_segments[i][2], line_segments[i][3])
            tmp_right = self.compute_p_value(x[start_x:end_x], y[start_x:end_x], line_segments[i+1][2], line_segments[i+1][3])
            left_p_w = tmp_left[1][1]*(mid_x-start_x)/(end_x-start_x+1)
            right_p_w = tmp_right[1][1]*(end_x - mid_x)/(end_x - start_x+1)
            if min(right_p_w,left_p_w) > 0.05:
                print "adding candidate for merging"
                print line_segments[i][5]
                candidates_inds.append((i, tmp_left[0], left_p_w+right_p_w, start_x, end_x))

        if len(candidates_inds) == 0:
            return
        else:
            max_p_ind = 0
            for i in range(len(candidates_inds)):
                if candidates_inds[i][2] > candidates_inds[max_p_ind][2]:
                    max_p_ind = i
            # merge them here and call recursion
            print 'merging'
            c = candidates_inds[max_p_ind]
            line_segments.pop(c[0])
            line_segments.pop(c[0])
            print c[0]
            line_segments.insert(c[0], [x_labels[c[3]], x_labels[c[4]], c[1].slope, c[1].intercept, x[c[3]], x[c[4]]])
            self.bottom_up_merge(line_segments, x, y, x_labels)


    def non_linear_fit(self, x, x_labels, y, line_seqments, prev_slope, prev_intercept):
        trend, slope = choppy.beyond_linear(x, y)

        print 'Non linear trend:'
        print x
        print trend, descriptor.describe_freq(slope[1]), descriptor.analyze_magnitude(slope[0])

        # line_seqments.append([x_labels[0], x_labels[-1], prev_slope, prev_intercept, x[0], x[-1]])


    def find_linear_fit(self, x, x_labels, y, prev_slope, prev_intercept, line_seqments, x_p, y_lp, y_rp, length, non_linear):
        division_point, lines, side = self.find_division_point(x, y, prev_slope, prev_intercept, x_p, y_lp, y_rp, length)
        if division_point == -1:  # it must return the whole line and add it to line_seqments
            if recursive_linear_fit.compute_MSE(x, y, prev_slope, prev_intercept) > self.variance_threshold:
                self.non_linear_fit(x, x_labels, y, line_seqments, prev_slope, prev_intercept)
            line_seqments.append([x_labels[0], x_labels[-1], prev_slope, prev_intercept, x[0], x[-1]])
            return
        logging.info("This is considered the division point in this step: ")
        logging.info(division_point)
        logging.info("[" + str(x_labels[0]) + " " + str(x_labels[-1]) + "] point is: " + str(x_labels[division_point]))
        left_length = length*len(x[0:division_point + 1])/len(x)
        right_length = length*len(x[division_point:])/len(x)
        # it should run for the right and left side


        self.find_linear_fit(x[0:division_point + 1], x_labels[0:division_point + 1], y[0:division_point + 1],
                             lines[0].slope, lines[0].intercept, line_seqments, x_p, y_lp, y_rp, left_length, False)
        self.find_linear_fit(x[division_point:], x_labels[division_point:], y[division_point:], lines[1].slope,
                             lines[1].intercept, line_seqments, x_p, y_lp, y_rp, right_length, False)

    # threshold
    def find_division_point(self, x, y, prev_slope, prev_intercept, x_p, y_lp, y_rp, length):
        logging.info("Trying to find the division point")
        mult = 5
        # first it has to compute L_m for each of the points in the series
        if len(x) <= 5:
            return -1, None, None
        slope_candidates = []
        intercept_candidates = []
        sub_candidates = []
        intercept_sub_candidates = []
        for i in range(len(x) - 2):
            left_line, left_p = self.compute_p_value(x[0:i + 2], y[0:i + 2], prev_slope, prev_intercept)
            right_line, right_p = self.compute_p_value(x[i + 1:], y[i + 1:], prev_slope, prev_intercept)
            slope_candidate = []
            intercept_candidate = []
            sub_candidate = []
            logging.info("point: (" + str(x[i]) + ', ' + str(y[i]) + ')' + "[left_pvalue: " + str(left_p) + "right_pvalue:" + str(right_p) + "]")
            x_p.append(x[i])
            y_lp.append(left_p[0] + left_p[1])
            y_rp.append(right_p[0] + right_p[1])
            #left_length
            ll = length*(i+1.5)/len(x)
            rl = length*((len(x) - i - 1.5)/len(x))
            record = (1.0/ll)*self.L_function(left_p, True)+ (1.0/rl)*self.L_function(right_p, True)
            record_l = (1.0/ll)*self.L_function(left_p, True)
            record_r = (1.0/rl)*self.L_function(right_p, True)
            if left_p[0] > self.stop_threshold or left_p[1] > self.stop_threshold or right_p[0] > self.stop_threshold or right_p[1] > self.stop_threshold or record > mult*self.stop_threshold:
                if left_p[0] < self.stop_threshold and left_p[1] < self.stop_threshold and record_l < mult*self.stop_threshold:
                    sub_candidate.append(record_l)
                    sub_candidate.append(i+1)
                    sub_candidate.append([left_line, right_line])
                    sub_candidate.append('Right')
                    sub_candidates.append(sub_candidate)
                elif right_p[0] < self.stop_threshold and left_p[1] < self.stop_threshold and record_r < mult*self.stop_threshold:
                    sub_candidate.append(record_r)
                    sub_candidate.append(i + 1)
                    sub_candidate.append([left_line, right_line])
                    sub_candidate.append('Left')
                    sub_candidates.append(sub_candidate)

                continue  # not a good point for being a division point


            slope_candidate.append(record)
            slope_candidate.append(i + 1)
            slope_candidate.append([left_line, right_line])
            slope_candidates.append(slope_candidate)

        for i in range(len(x) - 2):
            left_line, left_p = self.compute_p_value(x[0:i + 2], y[0:i + 2], prev_slope, prev_intercept)
            right_line, right_p = self.compute_p_value(x[i + 1:], y[i + 1:], prev_slope, prev_intercept)
            intercept_candidate = []
            intercept_sub_candidate = []
            logging.info("point: (" + str(x[i]) + ', ' + str(y[i]) + ')' + "[left_pvalue: " + str(left_p) + "right_pvalue:" + str(right_p) + "]")
            x_p.append(x[i])
            y_lp.append(left_p[0] + left_p[1])
            y_rp.append(right_p[0] + right_p[1])

            ll = length*(i+1.5)/len(x)
            rl = length*((len(x) - i - 1.5)/len(x))
            record = (1.0/ll)*self.L_function(left_p, False)+ (1.0/rl)*self.L_function(right_p, False)
            record_l = (1.0/ll)*self.L_function(left_p, False)
            record_r = (1.0/rl)*self.L_function(right_p, False)
            if left_p[0] > self.stop_threshold or left_p[1] > self.stop_threshold or right_p[0] > self.stop_threshold or right_p[1] > self.stop_threshold or record > mult*self.stop_threshold:
                if left_p[0] < self.stop_threshold and left_p[1] < self.stop_threshold and record_l < mult*self.stop_threshold:
                    intercept_sub_candidate.append(record_l)
                    intercept_sub_candidate.append(i+1)
                    intercept_sub_candidate.append([left_line, right_line])
                    intercept_sub_candidate.append('Right')
                    intercept_sub_candidates.append(intercept_sub_candidate)
                elif right_p[0] < self.stop_threshold and left_p[1] < self.stop_threshold and record_r < mult*self.stop_threshold:
                    intercept_sub_candidate.append(record_r)
                    intercept_sub_candidate.append(i + 1)
                    intercept_sub_candidate.append([left_line, right_line])
                    intercept_sub_candidate.append('Left')
                    intercept_sub_candidates.append(intercept_sub_candidate)

                continue  # not a good point for being a division point


            intercept_candidate.append(record)
            intercept_candidate.append(i + 1)
            intercept_candidate.append([left_line, right_line])
            intercept_candidates.append(intercept_candidate)

        if len(slope_candidates) > 0:
            min_index = 0
            for i in range(len(slope_candidates)):
                if slope_candidates[i][0] < slope_candidates[min_index][0]:
                    min_index = i
            return slope_candidates[min_index][1], slope_candidates[min_index][2], None
        if len(intercept_candidates) > 0:
            min_index = 0
            for i in range(len(intercept_candidates)):
                if intercept_candidates[i][0] < intercept_candidates[min_index][0]:
                    min_index = i
            return intercept_candidates[min_index][1], intercept_candidates[min_index][2], None
        # find the minimum here now

        if len(sub_candidates) > 0:
                min_index = 0
                for i in range(len(sub_candidates)):
                    if sub_candidates[i][0] < sub_candidates[min_index][0]:
                        min_index = i
                return sub_candidates[min_index][1], sub_candidates[min_index][2], sub_candidates[min_index][3]
        if len(intercept_sub_candidates) > 0:
                min_index = 0
                for i in range(len(intercept_sub_candidates)):
                    if intercept_sub_candidates[i][0] < intercept_sub_candidates[min_index][0]:
                        min_index = i
                return intercept_sub_candidates[min_index][1], intercept_sub_candidates[min_index][2], intercept_sub_candidates[min_index][3]

        return -1, None, None


    def L_function(self, p_value, slope):
        if slope:
            return p_value[1]*2
        return p_value[0]*2
        # return p_value[0]+p_value[1]


    def rSqMse(self, yHat, y):
        yBar = y.mean()
        SST = ((y - yBar) ** 2).sum()
        SSE = float(((y - yHat) ** 2).sum())
        MSE = SSE / float(len(y))
        return [1 - SSE / SST, MSE]

    def get_lines(self, series, anomaly_info):
        out_json = {'linear fits': [], 'anomaly points': [anomaly_info[0]]}
        lines = []
        # find the first line here
        logging.info("before p value computation")
        first_line, dummy_p = self.compute_p_value(series.times, series.values, 0, 0)
        logging.info("first estimated_line: ")
        prev_slope = first_line.slope
        prev_intercept = first_line.intercept
        logging.info(str(prev_slope) + " , " + str(prev_intercept))
        x_p = []
        y_lp = []
        y_rp = []
        non_linear = True
        self.find_linear_fit(series.times, series.times_labels, series.values, prev_slope, prev_intercept, lines, x_p, y_lp, y_rp, 1.0, non_linear)
        out_json['linear fits'] = self.create_output_Intervals(lines)


        xHat = np.linspace(min(series.times), max(series.times), num=200)
        yHat = self.predict(xHat, lines)

        plt.plot(xHat, yHat, '.')
        plt.plot(series.times, series.values, 'o')
        plt.show()


        self.bottom_up_merge(lines, series.times, series.values, series.times_labels)

        xHat = np.linspace(min(series.times), max(series.times), num=200)
        yHat = self.predict(xHat, lines)
        plt.plot(xHat, yHat, '.')
        plt.plot(series.times, series.values, 'o')
        print 'after merge' + str(len(lines))
        plt.show()

        # self.visualize(lines, series, [elm[1] for elm in anomaly_info[1]], [elm[0] for elm in anomaly_info[1]])
        print 'recursive p_value '
        return True, out_json, xHat, yHat, [elm[1] for elm in anomaly_info[1]],[elm[0] for elm in anomaly_info[1]], lines


    def simple(self):
        return 2

    def analyze_error(self, series):
        rSquare = []
        MSE = []
        length = []
        lines = self.get_lines(series)
        np_times = np.array(series.times)
        np_values = np.array(series.values)
        tot_length = len(np_times)
        for line in lines:
            xVal = np_times[(np_times >= line[4]) & (np_times <= line[5])]
            yVal = np_values[(np_times >= line[4]) & (np_times <= line[5])]
            yHat = line[2] * xVal + line[3]
            temp = self.rSqMse(yHat, yVal)
            rSquare.append(temp[0])
            MSE.append(temp[1])
            length.append(len(xVal) / float(tot_length))
        return np.array(MSE), np.array(rSquare), np.array(length)

    def metrices(self, series, option):
        MSE, rSquare, length = self.analyze_error(series)
        if option == 1:
            power = 1.5
            return (rSquare * (length ** power)).sum()
        if option == 2:
            power = 0.5
            return (MSE * (length ** power)).sum()
        if option == 3:
            power = 5
            return (MSE / (length ** power)).sum()
        if option == 4:
            power = 2
            return (MSE / (1 + length ** power)).sum()
        if option == 5:
            return (rSquare * (np.exp(length ** 0.5))).sum()
        if option == 6:
            return (length ** 2).sum()


    def predict(self, x_array, lines):
        y = []
        for x in x_array:
            y.append(self.predict_y(x, lines))
        return y

    def predict_y(self, x, lines):
        for line in lines:
            if x >= line[4] and x <= line[5]:
                return line[2] * x + line[3]

    # creates the output dictionary for the given fitted lines
    # needs some improvement for small data set
    def create_output_Intervals(self, fitted_lines):
        Interval_descriptoins = []
        for line in fitted_lines:
            description = descriptor.describe_slope(line[3])
            Interval_descriptoins.append({"start": line[0], "end": line[1], "description": description,
                                          "meta_data": {"slope": line[2], "intercept": line[3]}})

        return Interval_descriptoins

    @classmethod
    def de_trend(cls, x, y, slope, intercept):
        de_trended = []
        for i in range(len(x)):
            de_trended.append(y[i] - (slope*x[i]+intercept))
        return de_trended

    @classmethod
    def compute_MSE(cls, x, y, slope, intercept):
        res = 0
        for i in range(len(x)):
            res += (y[i] - (slope*x[i]+intercept))*(y[i] - (slope*x[i]+intercept))
        res *= 10.0 / len(x)
        print "This is the threshold: " + str(res)
        return res