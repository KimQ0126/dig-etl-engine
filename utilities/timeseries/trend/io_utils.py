import json
import re
import logging

class parse_input:
    description_word = 'time_series_description'

    def __init__(self, file_type, file_path, out_path):
        self.type = file_type
        self.src = file_path
        self.dst = out_path

    def set_method_type(self, analysis_type):
        self.method_name = analysis_type

    def parse_line(self, line, series_alias):
        line_json = json.loads(line)
        return self.modify_input(line_json[series_alias], True)

    # find the bug here
    def match_date(self, date):
        day_patterns = [r'(\d+) (\d+) (\d+)', r'(\d+)-(\d+)-(\d+)']
        for x in day_patterns:
            matchObj = re.match(x, date, re.M | re.I)
            if matchObj:
                if int(matchObj.group(1)) > 1000:  # this is perhaps the year
                    return True, int(matchObj.group(1)), int(matchObj.group(2)), int(matchObj.group(3))
                elif int(matchObj.group(3)) > 1000:
                    return True, int(matchObj.group(3)), int(matchObj.group(2)), int(matchObj.group(1))
        patterns = [r'(\d+) (\d+)', r'(\d+)-(\d+)']
        for x in patterns:
            matchObj = re.match(x, date, re.M | re.I)
            if matchObj:

                if int(matchObj.group(1)) > 1000: # this is perhaps the year
                    return True, int(matchObj.group(1)), int(matchObj.group(2)), 0
                elif int(matchObj.group(2)) > 1000:
                    return True, int(matchObj.group(2)), int(matchObj.group(1)), 0

        return False, 0, 0, 0

    def parse_date(self, date):
        if 'BIO' in date:
            date = date[4:]
        check_patterns = self.match_date(date)
        if check_patterns[0]:
            return  check_patterns[1], check_patterns[2], check_patterns[3]

        d = re.split("( +)|T", date)[0]
        splitted_date = d.split("-")
        year = int(splitted_date[0])
        month = 0 # default value is zero
        day = 0 # default value is zero
        if len(splitted_date)>1:
            month = int(splitted_date[1])
        if len(splitted_date)>2:
            day = int(splitted_date[2])
        return int(year), int(month), int(day)

    def is_greater(self, date, date1):
        year, month, day = self.parse_date(date)
        year1, month1, day1  = self.parse_date(date1)
        if year > year1:
            return True
        if year1 > year:
            return False
        if month > month1:
            return True
        if month < month1:
            return False
        if day > day1:
            return True
        return False

    def get_date_difference(self, date1, date2):
        # print 'The difference of dates: ' + str(date1) + ' ' + str(date2)
        year, month, day = self.parse_date(date1)
        # print 'parsed to: '+ str(self.parse_date(date1))
        year2, month2, day2 = self.parse_date(date2)
        # print 'parsed to: ' + str(self.parse_date(date2))
        diff = (year - year2) * 365
        diff += (month - month2) * 30 # some inaccuracies here
        diff += day - day2
        # print str(1.0 * diff / 365)
        return 1.0 * diff / 365

# assuming the dates formats are yy/month/dayT00:00:00
    def modify_input(self, time_series, normalizing):
        xarray = []
        yarray = []
        x_labels = []
        for i in range(len(time_series)):
            if time_series[i][1] == '' or time_series[i][1] == " " or time_series[i][1] == 'N/A':
                continue
            # print time_series[i][1]
            yarray.append(time_series[i][1])
            x_labels.append(time_series[i][0])

        if len(x_labels) == 0:
            return [], [], [], {'min':0, 'max':0}#, 'avg':0}

    # sort the input here:
        xarray.append(0)
        for i in range(len(x_labels) - 1):
            xarray.append(self.get_date_difference(x_labels[i+1], x_labels[0]))

        sorted_indices = sorted(range(len(xarray)), key=lambda x: xarray[x])
        y = [yarray[i] for i in sorted_indices]
        x = [x_labels[i] for i in sorted_indices]
        # scale the x here should be moved somewhere else -> move to the linear fit part
        # if normalizing:
        #     max_x = max(xarray)
        #     max_y = max(y)
        #     min_y = min(y)
        #     if max_x > 0:
        #         for i in range(len(xarray)):
        #             xarray[i] *= 1.0/max_x
        #     if max_y != 0 and max_y != min_y:
        #         for i in range(len(y)):
        #             y[i] = (y[i]- min_y)*1.0/(max_y - min_y)

        attributes = {'min':min(yarray), 'max':max(yarray)}#, 'avg':1.0*sum(yarray)/len(yarray)}
        return x, [xarray[i] for i in sorted_indices], y, attributes

    def append_trends_to_series(self, trend, attrs):
        if self.description_word not in self.current_series_ref.keys():
            self.current_series_ref[self.description_word] = dict()
        self.current_series_ref[self.description_word][self.method_name] = trend
        self.current_series_ref['attributes'] = attrs

    def save_output(self):
        with open(self.dst, mode='w') as output:
            json.dump(self.source_json, output)
            output.close()

    def end_output(self):
        self.out_file.close()

    def save_and_store_output(self, trend, attrs):
        if self.description_word not in self.source_json.keys():
            # print 'adding word to the keys of the input'
            self.source_json[self.description_word] = dict()
        self.source_json[self.description_word][self.method_name] = trend
        self.source_json['attributes'] = attrs
        json.dump(self.source_json, self.out_file)
        self.out_file.write("\n")

    def parse_jl_file(self, in_file, ts_key):
        num = 0
        self.out_file = open(self.dst, mode='w')
        with open(in_file) as f:
            for line in f:
                num += 1
                self.source_json = json.loads(line)
                yield self.modify_input(self.source_json[ts_key], True)

    def parse_json_file(self, infile, ts_key):
        with open(infile) as f:
            for line in f:
                self.source_json = x = json.loads(line)
                for sheet in x:
                    for time_s in sheet:
                        try:
                            self.current_series_ref = time_s
                            tmp = self.modify_input(time_s[ts_key], True)
                            if len(tmp[0]) == 0:
                                self.append_trends_to_series([], {})
                                continue
                            yield tmp
                        except:
                            self.append_trends_to_series([], {})
                            logging.error("series problem with reading and parsing the input for this series")
                            logging.error(str(self.current_series_ref))
                            continue



    def parse_raw_knoema(self, infile):
        with open(infile) as f:
            for line in f:
                res = []
                json_obj = json.loads(line)
                ts_data = json_obj['data']
                for time_point in ts_data:
                    res.append([time_point['Time'], time_point['Value']])
                yield self.modify_input(res, True)

    @classmethod
    def find_element_index(cls, array, element):
        for i in range(len(array)):
            if array[i] == element:
                return i
        return -1


    def get_line_segment(self, ts_line, x_label, x):
        # x_label,x, y = self.modify_input(ts_line['ts'], True)
        # ts_descr = ts_line[key]
        linear_fits = ts_line["linear fits"]
        line_segements = []
        for line in linear_fits:
            m_data = line["meta_data"]
            line_descr = [line["start"], line["end"], float(m_data["slope"]), float(m_data["intercept"])]
            # finding the scaled value for time axis
            start_index = parse_input.find_element_index(x_label, line["start"])
            end_index = parse_input.find_element_index(x_label, line["end"])
            line_descr.append(float(x[start_index]))
            line_descr.append(float(x[end_index]))
            line_segements.append(line_descr)
        return line_segements
