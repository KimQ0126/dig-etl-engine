import numpy as np
import linear_fit
import argparse
import io_utils
import recursive_p_value
import logging
import time_series

# dictionary format: {start: time, end: time, description: string, metadata: {slope: val, intercept: val}}
class trend_analysis:
    def __init__(self, src, dst, src_type):
        self.file_type = src_type
        self.src = src
        self.dst = dst
        self.input_utility = io_utils.parse_input(src_type, src, dst)


    # default method for trend analysis is assumed to be linear fit
    def run_trend_analysis(self, analysis_type, ts_keyword):

        if analysis_type == "lf":
            self.input_utility.set_method_type('piecewise linear')
            lf = linear_fit.piecewise_linear_fit()
            if self.file_type == 'json':
                for x_labels, xarray, yarray, attrs in self.input_utility.parse_json_file(self.src, ts_keyword):
                    series = time_series.time_series(times=xarray, values=yarray, time_labels=x_labels)
                    self.input_utility.append_trends_to_series(lf.analyze_series_with_points(series, series.remove_anomalies())[1], attrs)
                self.input_utility.save_output()
            elif self.file_type == 'jl':
                for x_labels, xarray, yarray, attrs in self.input_utility.parse_jl_file(self.src, ts_keyword):
                    series = time_series.time_series(times=xarray, values=yarray, time_labels=x_labels)
                    self.input_utility.save_and_store_output(lf.analyze_series_with_points(series, series.remove_anomalies())[1], attrs)
                self.input_utility.end_output()
        else:
            self.input_utility.set_method_type('recursive segmentation')
            rf = recursive_p_value.recursive_linear_fit()
            step = 0
            # This part is for raw Knoema datasets
            # if self.file_type == 'jl':
            #     for x_labels, xarray, yarray in self.input_utility.parse_raw_knoema(self.src):
            #         print "The number of series is: " + str(step)
            #         step += 1
            #         series = time_series(times=xarray, values=yarray, time_labels=x_labels)
            #         rf.analyze_series_with_points(series)

            if self.file_type == 'json':
                for x_labels, xarray, yarray, attrs in self.input_utility.parse_json_file(self.src, ts_keyword):
                    series = time_series.time_series(times=xarray, values=yarray, time_labels=x_labels)
                    series.normalize()
                    self.input_utility.append_trends_to_series(rf.fit(series, series.remove_anomalies()), attrs)
                self.input_utility.save_output()
            elif self.file_type == 'jl':
                for x_labels, xarray, yarray, attrs in self.input_utility.parse_jl_file(self.src, ts_keyword):

                    series = time_series.time_series(times=xarray, values=yarray, time_labels=x_labels)
                    # tmp = series.remove_anomalies()
                    # print tmp
                    #
                    series.normalize()
                    self.input_utility.save_and_store_output(rf.fit(series, series.remove_anomalies()), attrs)
                self.input_utility.end_output()
        return



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help='time series input file')
    ap.add_argument("file_type", help='the original data file which time serieses were extracted from')
    ap.add_argument("outfile", help='file to write results')
    ap.add_argument("analysis_type", help="the type of intended trend analysis")
    ap.add_argument("ts_key", help="the key used for time series in json file")
    args = ap.parse_args()
    logging.basicConfig(filename='test.log', level=logging.DEBUG)
    ta = trend_analysis(args.input, args.outfile, args.file_type)
    ta.run_trend_analysis(args.analysis_type, args.ts_key)

# #
# ta = trend_analysis('dev_data/wfpvam.json', 'wfpvam.trends.json', 'json')
# ta.run_trend_analysis('rp', 'ts')