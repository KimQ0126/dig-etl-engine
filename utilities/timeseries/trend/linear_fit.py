import logging
import numpy as np
import pwlf
from sets import Set
from interpret import visualization
from interpret import descriptor

class piecewise_linear_fit:
    error_step_hyper_param = 0.005 # the cost of adding one more line to best number of lines
    y_axis_scale = 1 # the scale of the y-axis after removing anomaly points. This scale is used in describing the linear fits tangents
    x_axis_scale = 1 # should we scale the time-axis?
    num_iteration = 0 # number of iteration to find the best linear fit among those found
    epsilon = 0.0001

    def __init__(self):
        self.initial = 0

    # a better version
    def analyze_series_with_points(self, series, anomaly_info):
        try:
            out_json = {'linear fits': [], 'anomaly points': [anomaly_info[0]]}
            if (len(series.times) <= 6):
                out_json['linear fits'] = self.linear_fit_for_small_data(series)
                return out_json

            break_points = Set([])
            num_lines, res, best_linear_fit = self.estimate_number_of_lines(np.array([t for t in series.times]), np.array([v for v in series.values]))
            points = [min(series.times), max(series.times)]
            for x in break_points:
                if x > 0 and x < len(series.times)-1:
                    points.append(series.times[x])

        # writing the description for the piecewise linear
            out_json['linear fits'] = self.create_output_Intervals(series, series, best_linear_fit, sorted(points))

        # this part is only for plotting
            xHat = np.linspace(min(series.times), max(series.times), num=100)
            yHat = best_linear_fit.predict(xHat)
        #    visualization.draw_plot(xHat, yHat, series.times, series.values, [elm[1] for elm in anomaly_info[1]], [elm[0] for elm in anomaly_info[1]], 'PWLF')
            return out_json#, xHat, yHat, anomaly_info
        except:
            return [{"start":0, "end":0, "description":"Problem Occured", "meta_data":{}}]

    # finds the lines connecting the points in data. It's for the case that the length of series is small
    def linear_fit_for_small_data(self, series):
        out_json = []
        if len(series.times) == 1:
            return [{"start":series.times_labels[0], "end":series.times_labels[0], "description":"single_point", "meta_data":{}}]
        for i in range(len(series.times) - 1):
            slope = self.y_axis_scale *(series.values[i + 1] - series.values[i]) / (series.times[i + 1] - series.times[i])
            meta_data = {"slope":slope, "intercept": series.values[i] - slope * series.times[i]}
            out_json.append({"start":series.times_labels[i], "end":series.times_labels[i+1], "description":  descriptor.describe_slope(slope), "meta_data":meta_data})
        return out_json

    # find the appropriate number of lines and the best linear fit:
    def estimate_number_of_lines(self, x, y):
        # starting from 1 line and adding and comparing the errors:
        prevError = float('Inf')
        e, tmp_pwlf, tmp_fit = self.compute_linear_error(3, x, y)
        for i in range(max(1, (len(x)/ 3))):
            error, curr_pwlf, fit = self.compute_linear_error(i + 3, x, y)
            estimated_cost = error / len(x) + self.error_step_hyper_param * (i + 3)
            logging.info('linear error ' + str(error / len(x)))
            logging.info('estimated cost with ' + str(i + 3) + ' lines is: ' + str(estimated_cost))
            if estimated_cost > prevError:
                return i + 3, fit, curr_pwlf
            prevError = estimated_cost
            tmp_fit = fit
            tmp_pwlf = curr_pwlf
        e, tmp_pwlf, tmp_fit = self.compute_linear_error(len(x)/3, x, y)
        return len(x)/3, tmp_fit, tmp_pwlf

    # given the number of lines, it computes the piecewise linear fit with the specified number of lines. Also computes the sqaured error of the piecewise fit
    def compute_linear_error(self, num_lines, x, y):  # if number of lines is 1 or 2 try to find another solution
        Linear_error = 0
        myPWLF = pwlf.piecewise_lin_fit(x, y)
        res = myPWLF.fit(num_lines) # this line is necessary although it seems useless
        for i in range(len(x)):
            xi = x[i]
            yH = myPWLF.predict(xi)
            Linear_error += (yH - y[i]) * (yH - y[i])
        return Linear_error/ (max(y)-min(y))** 2, myPWLF, res

    # describe the features of a fitted line in an interval. The features are slope of linear regression, intercept of the line and description
    def describe_fitted_line(self, x_start, x_end, pwlf, y_magnitude):
        y_end = pwlf.predict(x_end)
        y_start = pwlf.predict(x_start)
        slope = (y_end - y_start) / (x_end - x_start)
        return slope, y_start - slope * x_start, descriptor.describe_slope(slope)


    # creates the output dictionary for the given fitted lines
    # needs some improvement for small data set
    def create_output_Intervals(self, refined_series, original_series, best_linear_fit, fitted_lines):
        Interval_descriptoins = []
        last_visited_interval_ind = 0
        slope, intercept, description = self.describe_fitted_line(fitted_lines[0], fitted_lines[1],best_linear_fit, (self.y_axis_scale / (max(refined_series.values) - min(refined_series.values))))
        Interval_descriptoins.append({"start": original_series.times_labels[last_visited_interval_ind], "description": description,"meta_data": {"slope": slope[0], "intercept": intercept[0]}})

        for i in range(len(fitted_lines) - 2):
            slope, intercept, description = self.describe_fitted_line(fitted_lines[i+1], fitted_lines[i + 2], best_linear_fit, (self.y_axis_scale / (max(refined_series.values) - min(refined_series.values))))
            # search the values in x_copy then find the string in labels
            for j in range(len(original_series.times) - 1):
                if fitted_lines[i+1]-self.epsilon == original_series.times[j+1]:
                    last_visited_interval_ind = j+1
            Interval_descriptoins.append({"start":original_series.times_labels[last_visited_interval_ind], "description":description, "meta_data":{"slope":slope[0], "intercept":intercept[0]}})
        # the end of each interval is the start of another interval
        for i in range(len(Interval_descriptoins)-1):
            Interval_descriptoins[i]["end"] = Interval_descriptoins[i+1]["start"]
        Interval_descriptoins[len(Interval_descriptoins)-1]["end"] = original_series.times_labels[-1]
        return Interval_descriptoins


    # given an estimate of break points find the closest point in our time series to these points
    def find_fitted_lines_break_points(self, original_series, fitted_lines):
        last_visited_interval_ind = 0
        break_points = Set([])
        for i in range(len(fitted_lines) - 2):
            for j in range(len(original_series.times) - 1):
                d1 = fitted_lines[i+1] - original_series.times[j]
                d2 = original_series.times[j+1] - fitted_lines[i+1]
                if d1 > 0 and d2 > 0:
                    if d1 < d2:
                        last_visited_interval_ind = j
                    else:
                        last_visited_interval_ind = j+1

            break_points.add(last_visited_interval_ind)
        return break_points
