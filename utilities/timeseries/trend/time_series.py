import logging

class time_series:
    anomaly_threshold = 10
    anomaly_abs_threshold = 0.3
    def __init__(self, times, values, time_labels):
        self.times = times
        self.values = values
        self.times_labels = time_labels
        self.x_axis = {"shift":0.0, "scale":1.0}
        self.y_axis = {"shift":0.0, "scale":1.0}

    def remove_anomalies(self):
        anomalies = self.find_anomaly_knn()
        anomaly_info = []
        anomaly_labels = []
        for i in reversed(anomalies):
            anomaly_labels.append(self.times_labels.pop(i))
            anomaly_info.append((self.values.pop(i), self.times.pop(i)))
        return anomaly_labels, anomaly_info

    def find_anomaly_knn(self):
        if (len(self.times) <= 10):  # it does not makes sense to have anomaly in a small sample of times
            return []

        num_points = len(self.times)
        anomaly_values = []
        anomaly_indices = []

        # get the distance of the point from it's nearest neighbor
        for i in range(len(self.values)):
            try:
                if i - 2 >= 0 and i + 2 < len(self.values):
                # tmp = min(abs(self.values[i] - self.values[i - 1])*0.65+abs(self.values[i] - self.values[i - 2])*0.35, abs(self.values[i] - self.values[i + 1])*0.65+abs(self.values[i] - self.values[i + 2])*0.35)
                    tmp = min(abs((self.values[i] - self.values[i-1])/(self.times[i] - self.times[i-1])) + abs((self.values[i] - self.values[i-2])/(self.times[i] - self.times[i-2])), abs((self.values[i] - self.values[i+1])/(self.times[i] - self.times[i+1])) + abs((self.values[i] - self.values[i+2])/(self.times[i] - self.times[i+2])))
                    anomaly_values.append(tmp)
                elif i - 1 >= 0 and i + 1 < len(self.values):
                    tmp = min(abs((self.values[i] - self.values[i-1])/(self.times[i] - self.times[i-1])), abs((self.values[i] - self.values[i+1])/(self.times[i] - self.times[i+1])))
                    anomaly_values.append(tmp)
                elif i - 1 < 0:
                    anomaly_values.append(abs(self.values[i] - self.values[i + 1]))
                else:
                    anomaly_values.append(abs(self.values[i] - self.values[i - 1]))
            except:
                continue

        # check for the high left and right derivative of each candidate since we are only intersted in anomalies like ...../\.....
        # print anomaly_values
        for i in range(len(anomaly_values)):
            if anomaly_values[i] > (max(self.values) - min(self.values)) * self.anomaly_threshold:
                if i == 0 or i == len(self.values) - 1:
                    logging.info("anomaly detected at the beginning or end of interval! check it in all cases")
                    #  anomaly_indices.append(i)
                elif (self.values[i] - self.values[i - 1]) * (self.values[i + 1] - self.values[i]) < 0:
                    anomaly_indices.append(i)

        while len(anomaly_indices) * 10 > num_points:
            min_index = anomaly_indices[0]
            for i in range(len(anomaly_indices)):
                if anomaly_values[min_index] > anomaly_values[i]:
                    min_index = i
            anomaly_indices.pop(min_index)

        return anomaly_indices

    def find_anomaly_d(self):
        if (len(self.times) <= 10):  # it does not makes sense to have anomaly in a small sample of times
            return []

        anomaly_values = []
        anomaly_indices = []

        # get the distance of the point from it's nearest neighbor
        for i in range(len(self.values)):
            if i - 1 >= 0 and i + 1 < len(self.values):
                tmp = min(abs(self.values[i] - self.values[i - 1]), abs(self.values[i] - self.values[i + 1]))
                anomaly_values.append(tmp)
            elif i - 1 < 0:
                anomaly_values.append(abs(self.values[i] - self.values[i + 1]))
            else:
                anomaly_values.append(abs(self.values[i] - self.values[i - 1]))

        # check for the high left and right derivative of each candidate since we are only intersted in anomalies like ...../\.....
        # print anomaly_values
        for i in range(len(anomaly_values)):
            if anomaly_values[i] > (max(self.values) - min(self.values)) * self.anomaly_threshold:
                if i == 0 or i == len(self.values) - 1:
                    logging.info("anomaly detected at the beginning or end of interval! check it in all cases")
                    #  anomaly_indices.append(i)
                elif (self.values[i] - self.values[i - 1]) * (self.values[i + 1] - self.values[i]) < 0:
                    anomaly_indices.append(i)
        return anomaly_indices

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
        self.times = self.normalize_array(self.times, self.x_axis)
        self.values = self.normalize_array(self.values, self.y_axis)

