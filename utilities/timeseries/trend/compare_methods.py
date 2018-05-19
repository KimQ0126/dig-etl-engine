import io_utils
import linear_fit
import recursive_p_value
import time_series
from interpret import visualization




class compare_methods:
    def __init__(self, src, dst, src_type):
        self.file_type = src_type
        self.src = src
        self.dst = dst
        self.input_utility = io_utils.parse_input(src_type, src, dst, 'compare')


    # default method for trend analysis is assumed to be linear fit
    def compare_two_methods(self, ts_keyword, visualize):
        lf = linear_fit.piecewise_linear_fit()
        rf = recursive_p_value.recursive_linear_fit()
        if self.file_type == 'json':
            for x_labels, xarray, yarray, attrs in self.input_utility.parse_json_file(self.src, ts_keyword):
                series = time_series.time_series(times=xarray, values=yarray, time_labels=x_labels)
                anomaly_info = series.remove_anomalies()
                lf_res = lf.analyze_series_with_points(series, anomaly_info)
                rf_res = rf.get_lines(series, anomaly_info)
                if visualize:

                    print rf_res[4]
                    if lf_res[0]:
                        visualization.draw_comparison_plot(lf_res[2], lf_res[3], series.times, series.values, [elm[1] for elm in anomaly_info[1]], [elm[0] for elm in anomaly_info[1]], 'pwlf', rf_res[2], rf_res[3], 'RP')


        elif self.file_type == 'jl':
            step = 0
            for x_labels, xarray, yarray, attrs in self.input_utility.parse_jl_file(self.src, ts_keyword):
                    series = time_series.time_series(times=xarray, values=yarray, time_labels=x_labels)
                # print 'new approach for times: '
                    print series.times
                    print series.times_labels
                    anomaly_info = series.remove_anomalies()

                # try:
                    step += 1
                    print str(step) + '------------------------------'
                 #   lf_res = lf.analyze_series_with_points(series, series.remove_anomalies())
                    rf_res = rf.get_lines(series, anomaly_info)
                    if visualize:
                        visualization.draw_plot(rf_res[2], rf_res[3], series.times, series.values, [elm[1] for elm in anomaly_info[1]], [elm[0] for elm in anomaly_info[1]], 'RP')
                        # if lf_res[0]:
                        #     visualization.draw_comparison_plot(lf_res[2], lf_res[3], series.times, series.values,
                        #                                    [elm[1] for elm in anomaly_info[1]],
                        #                                    [elm[0] for elm in anomaly_info[1]], 'pwlf', rf_res[2],
                        #                                    rf_res[3], 'RP')
                # except:
                #      print "Problem occured"


# ta = trend_analysis('comparable_samples.jl', 'tst.jl', 'jl')
# ta.run_trend_analysis('lf', 'ts')

ta = compare_methods('dev_data/comparable_samples.jl', 'tst.jl', 'jl')
ta.compare_two_methods('ts', False)