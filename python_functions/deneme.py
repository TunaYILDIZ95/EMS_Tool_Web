import pandas as pd
import numpy as np
import scipy.io as sio

class DataReader():
    def __init__(self, file_name, file_location):
        self.file_name = file_name
        self.file_location = file_location
        self.file_path = f'{self.file_location}/{self.file_name}'
        self.shunt_susceptance, self.FromTo, self.y, self.y_shunt,self.BusList, self.bus_data, self.PL, self.QL, self.PG, self.QG, self.V, self.theta, self.type = ([] for _ in range(13))


    def data_parses(self):
        with open(self.file_path) as file_in:
            while True:
                line = file_in.readline()
                if line[0:4] == "BRAN":
                    line = file_in.readline()
                    while line[1:5] != "-999":
                        self.FromTo.append([float(line[0:5]), float(line[5:11]), 1, 1])
                        self.y.append([complex(float(line[16:26]), float(line[26:37])), complex(0, 0), complex(0, 0)])
                        self.y_shunt.append([float(line[37:48]), 0, 0])
                        line = file_in.readline()
                        self.y.append([complex(float(line[16:26]), float(line[26:37])), complex(float(line[48:59]), float(line[59:70])), complex(0, 0)])
                        self.y_shunt.append([float(line[37:48]), float(line[70:81]), 0])
                        line = file_in.readline()
                        self.y.append([complex(float(line[16:26]), float(line[26:37])), complex(float(line[48:59]), float(line[59:70])), complex(float(line[81:92]), float(line[92:103]))])
                        self.y_shunt.append([float(line[37:48]), float(line[70:81]), float(line[103:114])])
                        line = file_in.readline()

                if line[0:3] == "BUS":
                    line = file_in.readline()
                    while line[1:5] != "-999":
                        bus_name = float(line[0:5])
                        type = float(line[25:27])
                        self.BusList.append(bus_name)
                        self.bus_data.append([bus_name, 1])
                        self.PL.append(float(line[45:50]))
                        self.QL.append(float(line[54:59]))
                        self.PG.append(float(line[63:68]))
                        self.QG.append(float(line[72:77]))
                        self.V.append(float(line[28:33]))
                        self.theta.append(np.deg2rad(0))
                        self.type.append(type)

                        if float(line[120:132]) != 0:
                            self.shunt_susceptance.append([float(line[0:5]), float(line[120:132])])

                        line = file_in.readline()
                        self.bus_data.append([bus_name, 2])
                        self.PL.append(float(line[45:50]))
                        self.QL.append(float(line[54:59]))
                        self.PG.append(float(line[63:68]))
                        self.QG.append(float(line[72:77]))
                        self.V.append(float(line[28:33]))
                        self.theta.append(np.deg2rad(-120))
                        self.type.append(type)

                        line = file_in.readline()
                        self.bus_data.append([bus_name, 3])
                        self.PL.append(float(line[45:50]))
                        self.QL.append(float(line[54:59]))
                        self.PG.append(float(line[63:68]))
                        self.QG.append(float(line[72:77]))
                        self.V.append(float(line[28:33]))
                        self.theta.append(np.deg2rad(120))
                        self.type.append(type)

                        line = file_in.readline()

                if line[0:3] == "END":
                    break

    def Y_bus_creation(self):
        nbranch = len(self.FromTo)
        nbus = len(self.BusList)
        self.Yshunt = np.zeros(shape=(nbus*3,nbus*3), dtype = complex)
        self.Yn = np.zeros(shape=(nbus*3,nbus*3), dtype = complex)
        y = np.array(self.y)
        y_shunt = np.array(self.y_shunt)
        for i in range(nbranch):
            Fr = self.BusList.index(self.FromTo[i][0])
            To = self.BusList.index(self.FromTo[i][1])
            y_temp = y[i*3:i*3+3, 0:3]
            y_temp[0, 1] = y_temp[1, 0]
            y_temp[0, 2] = y_temp[2, 0]
            y_temp[1, 2] = y_temp[2, 1]
            y_s_temp = y_shunt[i*3:i*3+3, 0:3]
            y_s_temp[0, 1] = y_s_temp[1, 0]
            y_s_temp[0, 2] = y_s_temp[2, 0]
            y_s_temp[1, 2] = y_s_temp[2, 1]
            self.Yn[Fr * 3: Fr * 3 + 3, Fr * 3: Fr * 3 + 3] = self.Yn[Fr * 3: Fr * 3 + 3, Fr * 3: Fr * 3 + 3] + y_temp + y_s_temp
            self.Yn[To * 3: To * 3 + 3, To * 3: To * 3 + 3] = self.Yn[To * 3: To * 3 + 3, To * 3: To * 3 + 3] + y_temp + y_s_temp
            self.Yn[Fr * 3: Fr * 3 + 3, To * 3: To * 3 + 3] = self.Yn[Fr * 3: Fr * 3 + 3, To * 3: To * 3 + 3] - y_temp
            self.Yn[To * 3: To * 3 + 3, Fr * 3: Fr * 3 + 3] = self.Yn[To * 3: To * 3 + 3, Fr * 3: Fr * 3 + 3] - y_temp
            self.Yshunt[Fr * 3: Fr * 3 + 3, To * 3: To * 3 + 3] = self.Yshunt[Fr * 3: Fr * 3 + 3, To * 3: To * 3 + 3] + y_s_temp
            self.Yshunt[To * 3: To * 3 + 3, Fr * 3: Fr * 3 + 3] = self.Yshunt[To * 3: To * 3 + 3, Fr * 3: Fr * 3 + 3] + y_s_temp

        for i in range(len(self.shunt_susceptance)):
            location = self.BusList.index(self.shunt_susceptance[i][0])
            self.Yn[location * 3, location * 3] = self.Yn[location * 3, location * 3] + complex(0, self.shunt_susceptance[i][1])
            self.Yn[location * 3 + 1, location * 3 + 1] = self.Yn[location * 3 + 1, location * 3 + 1] + complex(0, self.shunt_susceptance[i][1])
            self.Yn[location * 3 + 2, location * 3 + 2] = self.Yn[location * 3 + 2, location * 3 + 2] + complex(0, self.shunt_susceptance[i][1])



