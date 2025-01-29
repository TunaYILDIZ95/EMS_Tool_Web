import numpy as np
import math
from scipy.linalg import solve

class PowerFlow():
    def __init__(self, args):
        self.PL = np.array(args.PL)
        self.QL = np.array(args.QL)
        self.PG = np.array(args.PG)
        self.QG = np.array(args.QG)
        self.Yn = args.Yn
        self.V = np.array(args.V)
        self.theta = np.array(args.theta)
        self.BusList = args.BusList
        self.type = np.array(args.type)
        self.nbus = len(self.BusList)
        self.FromTo = args.FromTo
        self.links = []
        self.nodes = []
        self.Q_limits = args.Q_limits

        self.P_known = np.subtract(self.PG, self.PL)
        self.Q_known = np.subtract(self.QG, self.QL)

        G = self.Yn.real
        B = self.Yn.imag
        pv_buses = np.where(self.type == 2)[0]
        for pv_bus in pv_buses:
            connected_lines = np.where(np.absolute(self.Yn[pv_bus,:]) > 0.0001)[0]
            value = 0
            for line in connected_lines:
                value += self.V[line] * self.V[line] * (G[pv_bus,line] * math.sin(self.theta[pv_bus]-self.theta[line])-B[pv_bus,line]*math.cos(self.theta[pv_bus]-self.theta[line]))
            self.Q_known[pv_bus] = value

    def current_mismatch_function(self, G, B):
        delta_I_r = np.zeros(shape=(self.nbus*3, 1))
        delta_I_m = np.zeros(shape=(self.nbus*3, 1))

        pv_buses = np.where(self.type == 2)[0]
        for i in range(3*self.nbus):
            value_real = 0
            value_imag = 0
            connected_lines = np.where(np.absolute(self.Yn[i,:]) > 0.0001)[0]
            for k in connected_lines:
                value_real += self.V[k] * (G[i, k] * math.cos(self.theta[k]) - B[i, k] * math.sin(self.theta[k]))
                value_imag += self.V[k] * (G[i, k] * math.sin(self.theta[k]) + B[i, k] * math.cos(self.theta[k]))
            if i not in pv_buses:
                delta_I_r[i] = ((self.P_known[i] * math.cos(self.theta[i]) + self.Q_known[i] * math.sin(self.theta[i])) / self.V[i]) - value_real
                delta_I_m[i] = ((self.P_known[i] * math.sin(self.theta[i]) - self.Q_known[i] * math.cos(self.theta[i])) / self.V[i]) - value_imag
            else:
                delta_I_r[i] = ((self.P_known[i] * math.cos(self.theta[i]) + (self.Q_known[i] - self.QL[i]) * math.sin(self.theta[i])) / self.V[i]) - value_real
                delta_I_m[i] = ((self.P_known[i] * math.sin(self.theta[i]) - (self.Q_known[i] - self.QL[i]) * math.cos(self.theta[i])) / self.V[i]) - value_imag

        self.delta_I = np.concatenate((delta_I_r, delta_I_m), axis=0)

    def power_flow_jacobian(self):
        pv_buses = np.where(self.type == 2)[0]

        J1 = np.zeros(shape=(self.nbus*3,self.nbus*3),dtype=complex)
        J2 = np.zeros(shape=(self.nbus * 3, self.nbus * 3), dtype=complex)
        J3 = np.zeros(shape=(self.nbus * 3, self.nbus * 3), dtype=complex)
        J4 = np.zeros(shape=(self.nbus * 3, self.nbus * 3), dtype=complex)

        G = self.Yn.real
        B = self.Yn.imag

        th = 10e-6
        error = 1
        while error > th:
            PowerFlow.current_mismatch_function(self, G, B)
            self.delta_I = np.delete(self.delta_I, (0, 1, 2, self.nbus*3, self.nbus*3+1,self.nbus*3+2), axis=0)
            error = max(self.delta_I)
            for i in range(self.nbus * 3):
                connected_lines = np.where(np.absolute(self.Yn[i, :]) > 0.0001)[0]
                for k in connected_lines:
                    if i == k:
                        J1[i, k] = self.V[i] * (G[i, k] * math.sin(self.theta[k]) + B[i, k] * math.cos(self.theta[i])) \
                                                    - ((self.P_known[i] * math.sin(self.theta[i]) - self.Q_known[i] * math.cos(self.theta[i])) / self.V[i])

                        J3[i, k] = -self.V[k] * (G[i, k] * math.cos(self.theta[k]) - B[i, k] * math.sin(self.theta[k])) \
                                                    + ((self.P_known[i] * math.cos(self.theta[i]) + self.Q_known[i] * math.sin(self.theta[i])) / self.V[i])

                        if not k in pv_buses:
                            J2[i, k] = -(G[i, i] * math.cos(self.theta[i]) - B[i, k] * math.sin(self.theta[i]))\
                                                        - ((self.P_known[i] * math.cos(self.theta[i]) + self.Q_known[i] * math.sin(self.theta[i])) / self.V[i]**2)

                            J4[i, k] = -(G[i, k] * math.sin(self.theta[i]) + B[i, k] * math.cos(self.theta[i])) \
                                                        - ((self.P_known[i] * math.sin(self.theta[i]) - self.Q_known[i] * math.cos(self.theta[i])) / self.V[i]**2)
                        else:
                            J2[i, k] = math.sin(self.theta[i]) / self.V[i]
                            J4[i, k] = -math.cos(self.theta[i]) / self.V[i]
                    else:
                        J1[i, k] = self.V[k] * (G[i, k] * math.sin(self.theta[k]) + B[i, k] * math.cos(self.theta[k]))
                        J3[i, k] = -self.V[k] * (G[i, k] * math.cos(self.theta[k]) - B[i, k] * math.sin(self.theta[k]))

                        if not k in pv_buses:
                            J2[i, k] = -(G[i, k] * math.cos(self.theta[k]) - B[i,k] * math.sin(self.theta[k]))
                            J4[i, k] = -(G[i, k] * math.sin(self.theta[k]) + B[i,k] * math.cos(self.theta[k]))
                        else:
                            J2[i, k] = 0
                            J4[i, k] = 0

            self.Jacobian_1 = np.concatenate((J1, J2), axis=1)
            self.Jacobian_2 = np.concatenate((J3, J4), axis=1)
            self.Jacobian = np.concatenate((self.Jacobian_1,self.Jacobian_2), axis=0)
            self.Jacobian = np.delete(self.Jacobian, (0 , 1, 2, self.nbus*3, self.nbus*3+1,self.nbus*3+2), axis = 0)
            self.Jacobian = np.delete(self.Jacobian, (0, 1, 2, self.nbus * 3, self.nbus * 3 + 1, self.nbus * 3 + 2), axis = 1)

            X = -solve(self.Jacobian, self.delta_I).real
            self.theta[3:len(self.theta)] = self.theta[3:len(self.theta)] + np.transpose(X[0:(self.nbus-1)*3])
            for i in range((self.nbus-1)*3):
                if not i in np.subtract(pv_buses, 3):
                    self.V[3+i] = self.V[3+i] + np.transpose(X[(self.nbus-1)*3+i])
                else:
                    self.Q_known[3+i] = self.Q_known[3+i] + np.transpose(X[(self.nbus-1)*3+i])
                    if self.Q_known[3+i] < self.Q_limits["min"][3+i]:
                        self.Q_known[3+i] = self.Q_limits["min"][3+i] - self.QL[3+i]
                        self.type[3+i] = 0
                    elif self.Q_known[3+i] > self.Q_limits["max"][3+i]:
                        self.Q_known[3+i] = self.Q_limits["max"][3+i] - self.QL[3+i]
                        self.type[3+i] = 0

            
        self.theta = np.rad2deg(self.theta)

    def connectivity_creator(self):
        for line in self.FromTo:
            self.links.append({"source": line[0],"target": line[1], "value":1})
        
        pv_buses = np.where(self.type == 2)[0] + 1
        for bus in self.BusList:
            if bus in pv_buses/3:
                self.nodes.append({ "id": bus, "group":1, "value":1.01})
            else:
                self.nodes.append({ "id": bus, "group":2, "value":1.01})  

