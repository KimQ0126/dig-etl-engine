import logging
import matplotlib.pyplot as plt
from interpret import descriptor
import numpy as np
from scipy import stats
from RP_package import recursive_segmenting, p_value_computer, line


class recursive_linear_fit:

    def __init__(self):
        self.lines = []

    def fit(self, series, anomaly_info):
        out_json = {'linear fits': [], 'anomaly points': [anomaly_info[0]]}
        # TODO: do something for small time serieses -> find a better method other than what it is now
        # todo: filter constant serieses -> assign 'flat' -> done
        try:
            if self.is_constant(series):
                out_json['linear fits'] =[{"start":series.times[0], "end":series.times[0], "description": descriptor.describe_slope(0), "meta_data":{"intercept":series.values[0], "slope":0}}]
                return out_json

            if len(series.times) < 10:
                out_json['linear fits'] = self.linear_fit_for_small_data(series)
                return out_json

            # check for the constant series here
            rs = recursive_segmenting(series.times, series.values)

            self.lines = rs.fit()
            out_json['linear fits'] = self.create_output_Intervals(series.times_labels)
            return out_json
        except:
            logging.error("There was a problem with P-value computation in this series(The following are times and values")
            logging.error(str(series.times_labels))
            logging.error(str(series.values))
            return [{"start": 0, "end": 0, "description": "Problem Occured", "meta_data": {}}]

    def is_constant(self, series):
        for i in range(len(series.values)-1):
            if series.values[i] != series.values[i+1]:
                return False
        return True

    # creates the output dictionary for the given fitted lines
    # needs some improvement for small data set
    def create_output_Intervals(self, labels):
        Interval_descriptoins = []
        for line in self.lines:

            Interval_descriptoins.append({"start": labels[line.start_index], "end": labels[line.end_index],"description": descriptor.describe_slope(line.slope),"meta_data":{"intercept": line.intercept, "slope":line.slope}})
        return Interval_descriptoins

    def linear_fit_for_small_data(self, series):
        out_json = []
        if len(series.times) == 1:
            return [{"start":series.times_labels[0], "end":series.times_labels[0], "description":"single_point", "meta_data":{}}]
        for i in range(len(series.times) - 1):
            slope = (series.values[i + 1] - series.values[i]) / (series.times[i + 1] - series.times[i])
            meta_data = {"slope":slope, "intercept": series.values[i] - slope * series.times[i]}
            out_json.append({"start":series.times_labels[i], "end":series.times_labels[i+1], "description":  descriptor.describe_slope(slope), "meta_data":meta_data})
        return out_json