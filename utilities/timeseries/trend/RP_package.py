import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats


class line:
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept

    def set_endpoints(self, start, end, start_val, end_val):
        self.start_index = start
        self.end_index = end
        self.start = start_val
        self.end = end_val

class p_value_computer:
    @classmethod
    def compute_line_p_value(cls, x_values, y_values, prev_intercept, prev_slope):
        num_points = len(x_values)
        Xc = sm.add_constant(x_values)
        line = sm.OLS(y_values, Xc).fit()
        intercept, slope = line.params[0], line.params[1]
        y_estimate = line.predict(Xc)
        x_mean = x_values.mean()
        Sxx = ((x_values - x_mean) ** 2).sum()
        Sqx = (x_values ** 2).sum()
        sigmaSq = ((y_values - y_estimate) ** 2).sum()
        S = np.sqrt(sigmaSq / (num_points - 2))
        slope_diff = abs(slope - prev_slope)
        intercept_diff = abs(intercept - prev_intercept)

        significant, p_beta = cls.compute_p_value(slope_diff, np.sqrt(Sxx), S)
        if not significant:
            betaStat = abs(slope_diff * np.sqrt(Sxx) / S)
            p_beta = stats.t.sf(betaStat, num_points - 2) * 2
        significant, p_alpha = cls.compute_p_value(intercept_diff, np.sqrt(Sxx), S)
        if not significant:
            alphaStat = abs((intercept_diff) / (S * np.sqrt(Sqx / (num_points * Sxx))))
            p_alpha = stats.t.sf(alphaStat, num_points - 2) * 2

        return intercept, slope, p_alpha, p_beta

    @classmethod
    def compute_p_value(cls, a, w_a, variance):
        if a * w_a < 0.001 * variance:
            return True, 1.0
        elif a * w_a > 1000 * variance:
            return True, 0.0
        else:
            return False, 0

class recursive_segmenting:
    confidence_band = 0.95
    min_segment_size = 5
    threshold_w = 5

    def __init__(self, x, y):
        self.lines = []
        self.x_data = x
        self.y_data = y
        self.x_axis = {'shift':0.0, 'scale':1.0}
        self.y_axis = {'shift':0.0, 'scale':1.0}


    @classmethod
    def normalize_array(cls, x, axis):
        x_normal = []
        min_x = min(x)
        max_x = max(x)
        if max_x - min_x > 0:
            axis['shift'] = min_x
            axis['scale'] = max_x - min_x
            for i in range(len(x)):
                x_normal.append(1.0 * (x[i]-min_x)/(max_x-min_x))
        else:
            axis['shift'] = min_x - 1
            x_normal = [1.0 for i in x]
        return x_normal

    def normalize(self):
        self.x_data = self.normalize_array(self.x_data, self.x_axis)
        self.y_data = self.normalize_array(self.y_data, self.y_axis)

    def fit(self):
        lines = []
        first_line = self.compare_lines(self.x_data, self.y_data, line(0, 0))
        self.find_linear_fit(self.x_data, [i for i in range(len(self.x_data))], self.y_data, first_line['line'], lines, 1.0)
        self.bottom_up_merge(lines, self.x_data, self.y_data)
        self.lines = lines
        # print 'slope and intercept of the first line are these ones respectively: '
        # print lines[0].slope, lines[0].intercept
        # # xHat = np.linspace(min(self.x_data), max(self.x_data), num=100)
        # # yHat = self.predict(xHat)
        # # plt.plot(xHat, yHat, '.')
        # plt.plot(self.x_data, self.y_data)
        plt.show()

        return lines#self.get_break_points()

    # find the best piecewise linear fit recursively
    def find_linear_fit(self, x, x_index, y, prev_line, line_seqments, length):
        division_point, lines = self.find_division_point(x, y, prev_line, length)
        if division_point == -1:  # No good division point exists
            prev_line.set_endpoints(x_index[0], x_index[-1], x[0], x[-1])
            line_seqments.append(prev_line)
            return

        left_line_len = length * len(x[0:division_point + 1])/len(x)
        right_line_len = length*len(x[division_point:])/len(x)
        self.find_linear_fit(x[0:division_point + 1], x_index[0:division_point + 1], y[0:division_point + 1], lines['left_line'], line_seqments, left_line_len)
        self.find_linear_fit(x[division_point:], x_index[division_point:], y[division_point:], lines['right_line'], line_seqments, right_line_len)

    # select the best division point based on the p_value difference
    def find_division_point(self, x, y, prev_line, length):
        if len(x) <= self.min_segment_size:
            return -1, None

        div_candidates = {'slope':[], 'one_side_slope':[], 'intercept':[], 'one_side_intercept':[]}
        for i in range(2, len(x)-3):
            left_line = self.compare_lines(x[0:i+1], y[0:i+1], prev_line)
            right_line= self.compare_lines(x[i:], y[i:], prev_line)
            self.check_point_features(i, length, len(x), left_line, right_line, div_candidates['slope'], div_candidates['one_side_slope'], True)
            self.check_point_features(i, length, len(x), left_line, right_line, div_candidates['intercept'], div_candidates['one_side_intercept'], False)

        selection_priority = ['slope', 'intercept', 'one_side_slope', 'one_side_intercept']
        for label in selection_priority:
            if len(div_candidates[label]) > 0:
                min_index = self.find_min_index(div_candidates[label], 'weighted_p_value')
                return div_candidates[label][min_index]['index'], div_candidates[label][min_index]['lines']
        return -1, None

    # compare the properties of the linear regression of (x,y) points against the given line
    def compare_lines(self, x, y, prev_line):
        intercept, slope, p_intercept, p_slope = p_value_computer.compute_line_p_value(np.array(x), np.array(y), prev_line.intercept, prev_line.slope)
        new_line = line(slope=slope, intercept=intercept)
        return {'line':new_line, 'p_value':{'slope':p_slope, 'intercept':p_intercept}}

    # check the features of the point in order to be selected as the division point in the given line segement.
    # It will be added to the candidate set if it's left and right segment(after division) p_values are significantly different from the current linear regression
    # if only one segment after division passes the significant test it will be added to partial candidate_set
    def check_point_features(self, div_index, length, segment_len, left_line, right_line, candidate_set, partial_candidates_set, slope_p_value):
        left_segment_weight = self.compute_segment_weight(div_index + 0.5, length / segment_len, slope_p_value, left_line['p_value'])
        right_segment_weight = self.compute_segment_weight(segment_len - div_index - 0.5, length / segment_len, slope_p_value, right_line['p_value'])
        total_weight = right_segment_weight + left_segment_weight

        if not self.check_threshold_cond(left_line['p_value']) or not self.check_threshold_cond(right_line['p_value']) or not self.significant_diff(total_weight/self.threshold_w):
            if self.check_threshold_cond(left_line['p_value']) and self.significant_diff(left_segment_weight/self.threshold_w):
                partial_candidates_set.append(self.create_div_candidate(left_segment_weight, div_index, left_line['line'], right_line['line']))
            elif self.check_threshold_cond(right_line['p_value']) and self.significant_diff(right_segment_weight/self.threshold_w):
                partial_candidates_set.append(self.create_div_candidate(right_segment_weight, div_index, left_line['line'], right_line['line']))
        else:
            candidate_set.append(self.create_div_candidate(total_weight, div_index, left_line['line'], right_line['line']))

    def compute_segment_weight(self, left_segment_length, real_length,  slope_phase, p_value):
        left_segment_prop = left_segment_length * real_length
        weight = (1.0 / left_segment_prop) * self.p_value_transform(p_value, slope_phase)
        return weight

    def check_threshold_cond(self, line):
        if self.significant_diff(line['intercept']) and self.significant_diff(line['slope']):
            return True
        return False

    def significant_diff(self, p_value):
        if p_value < (1 - self.confidence_band)/2:
            return True
        return False

    def create_div_candidate(self, weight, index, line_left, line_right):
        candidate = dict()
        candidate['weighted_p_value'] = (weight)
        candidate['index'] = index
        candidate['lines'] = {'left_line':line_left, 'right_line':line_right}
        return candidate

    def p_value_transform(self, p_value, slope):
        if slope:
            return p_value['slope']*2
        return p_value['intercept']*2

    def find_min_index(self, array, key):
        index = 0
        for i in range(len(array)):
            if array[i][key] < array[index][key]:
                index = i
        return index

    def predict(self, x_array):
        y = []
        for x in x_array:
            y.append(self.predict_y(x*self.x_axis['scale'] + self.x_axis['shift'])*self.y_axis['scale'] + self.y_axis['shift'])
        return y

    def predict_y(self, x):
        if x < self.lines[0].start:
            return self.lines[0].slope*x + self.lines[0].intercept

        for line in self.lines:
            if x >= line.start and x <= line.end:
                return line.slope * x + line.intercept
        return self.lines[-1].slope*x + self.lines[-1].intercept

    def get_break_points(self):
        break_points = [x.start*self.x_axis['scale']+self.x_axis['shift'] for x in self.lines]
        break_points.append(self.x_axis['scale']*self.lines[-1].end + self.x_axis['shift'])
        return break_points
    # think about representation

    @classmethod
    def bottom_up_merge(cls, lines, x, y):
        merge_points = []
        for i in range(len(lines) - 1):
            start_x = lines[i].start_index
            mid_x = lines[i].end_index
            end_x = lines[i + 1].end_index
            left_length = 1.0 * (mid_x - start_x) / len(x)
            right_length = 1.0 * (end_x - mid_x) / len(x)
            dense_side = i if right_length > left_length else i+1 # TODO: is the side selection correct?
            #
            # merge_p_values = p_value_computer.compute_line_p_value(np.array(x[start_x:end_x]), np.array(y[start_x:end_x]), lines[dense_side].intercept, lines[dense_side].slope)
            # slope_diff = abs((merge_p_values[1] - lines[dense_side].slope) / merge_p_values[1])
            #
            # if slope_diff < 0.25 and (left_length < 0.15 or right_length < 0.15):
            #     merge_points.append((i, merge_p_values[0], merge_p_values[1], slope_diff, start_x, end_x))


            merge_p_values = p_value_computer.compute_line_p_value(np.array(x[start_x:end_x]), np.array(y[start_x:end_x]),
                                                            lines[dense_side].intercept, lines[dense_side].slope)
            radian_diff = abs(np.arctan(merge_p_values[1]) - np.arctan(lines[dense_side].slope))
            shorter_length = right_length if end_x - mid_x < mid_x - start_x else left_length

            if radian_diff < 0.05:
                merge_points.append((i, merge_p_values[0], merge_p_values[1], radian_diff * shorter_length, start_x, end_x))

        if len(merge_points) != 0:
            max_p_ind = 0
            for i in range(len(merge_points)):
                if merge_points[i][3] > merge_points[max_p_ind][3]:
                    max_p_ind = i

            division_point = merge_points[max_p_ind]
            # remove left and right line segments
            lines.pop(division_point[0])
            lines.pop(division_point[0])
            new_line = line(division_point[2], division_point[1])
            new_line.set_endpoints(division_point[4], division_point[5], x[division_point[4]], x[division_point[5]])
            lines.insert(division_point[0], new_line)
            cls.bottom_up_merge(lines, x, y)

class merger:
    @classmethod
    def bottom_up_merge(cls, line_segments, x, y):
        candidates_inds = []
        for i in range(len(line_segments) - 1):
            start_x = x.index(line_segments[i].start)
            mid_x = x.index(line_segments[i].end)
            end_x = x.index(line_segments[i + 1].end)
            if end_x - mid_x < mid_x - start_x:
                tmp = p_value_computer.compute_line_p_value(np.array(x[start_x:end_x]), np.array(y[start_x:end_x]),
                                                            line_segments[i].intercept, line_segments[i].slope)
                radian_diff = abs(np.arctan(tmp[1]) - np.arctan(line_segments[i].slope))
                shorter_length = 1.0 * (end_x - mid_x) / len(x)
            else:
                tmp = p_value_computer.compute_line_p_value(np.array(x[start_x:end_x]), np.array(y[start_x:end_x]),
                                                            line_segments[i + 1].intercept, line_segments[i + 1].slope)
                radian_diff = abs(np.arctan(tmp[1]) - np.arctan(line_segments[i + 1].slope))
                shorter_length = 1.0 * (mid_x - start_x + 1) / len(x)
            if radian_diff < 0.05:
                candidates_inds.append((i, tmp[0], tmp[1], radian_diff * shorter_length, start_x, end_x))
