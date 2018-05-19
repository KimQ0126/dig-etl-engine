import numpy as np
import computeErrors

#use e = error(0)
# e.metrices(series, lines, option)
# series is an object of class time series
# lines is like line_segments Pegah defined in recuresive_p_value.py
# option is 1-6 see formula for details

class error:
    
    def __init__(self):
        self.initial = 0     

    def rSqMse(self, yHat , y): 
        yBar = y.mean()
        SST = ((y - yBar) ** 2).sum()
        SSE = float(((y - yHat) ** 2).sum())
        MSE = SSE/float(len(y))
        return [1-SSE/SST, MSE]
    
    def analyze_error(self, series, lines):
        rSquare = []
        MSE = [] 
        length = [] 
        np_times = np.array(series.times)
        np_values = np.array(series.values)
        tot_length = len(np_times)
        for i in range(len(lines)):
                line = lines[i]
    #            print line
                xVal = np_times[(np_times>= line[4]) & (np_times<= line[5])]
                yVal = np_values[(np_times>= line[4]) & (np_times<= line[5])]

                yHat = line[2]*xVal+line[3]
                temp = self.rSqMse(yHat,yVal)
                rSquare.append(temp[0])
                MSE.append(temp[1])
                if len(lines) == 1:
                    length.append(len(xVal) / float(tot_length))
                elif i > 0 and i < len(lines) - 1:
                    length.append((len(xVal)-1)/float(tot_length))
                else:
                    length.append((len(xVal)-0.5)/float(tot_length))

        return np.array(MSE), np.array(rSquare), np.array(length)
        
    def metrices(self, series, lines, option):

        MSE, rSquare, length = self.analyze_error(series, lines)
        if option == 1: 
            power = 1.5
            return (rSquare*(length**power)).sum()
        if option == 2: 
            power = 0.5
            return (MSE*(length**power)).sum()
        if option == 3: 
            power = 5
            return (MSE/(length**power)).sum()
        if option == 4: 
            power = 2
            return (MSE/(1+length**power)).sum()
        if option == 5: 
            return (rSquare*(np.exp(length**0.5))).sum()
        if option == 6:
            return (length**2).sum()

