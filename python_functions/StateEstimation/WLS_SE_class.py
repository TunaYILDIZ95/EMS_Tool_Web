import numpy as np
import pandas as pd
from dss import dss
from scipy.sparse import csc_matrix
from scipy.optimize import linprog
from scipy.linalg import qr
import copy
from operator import itemgetter
from cmath import rect
import itertools
from scipy.io import savemat
import ctypes
# import pulp
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import linalg

class StateEstimation:
    def __init__(self, instance):
        ### Main Class
        self.instance = instance
        ### System Data
        self.systemSize = instance.systemSizeIsland
        self.nodeOrder = instance.nodeOrderIsland
        self.busTobusConnection = instance.busTobusConnectionIsland
        self.lineSusceptance = instance.lineSusceptanceIsland
        self.lineConductance = instance.lineConductanceIsland
        self.base_power_kw = instance.base_power_kw
        self.voltageInitial = instance.voltageInitialIsland
        self.thetaInitial = instance.thetaInitialIsland
        self.initial_switchStates = instance.initial_switchStates

        ### Measurement Data
        ## SCADA Measurement
        self.realInjection = instance.realInjectionIsland.copy()
        self.reacInjection = instance.reacInjectionIsland.copy()
        self.realFlow = instance.realFlowIsland.copy()
        self.reacFlow = instance.reacFlowIsland.copy()
        ## PMU Measurement
        self.voltageMagPMU = instance.voltageMagPMUIsland.copy()
        self.voltageAnglePMU = instance.voltageAnglePMUIsland.copy()
        
        # self.voltageMagPMU = np.array([])
        # self.voltageAnglePMU = np.array([])
        self.currentMagPMU = instance.currentMagPMU.copy()
        self.currentAnglePMU = instance.currentAnglePMU.copy()

        self.voltages = instance.voltagesIsland.copy()
        self.angles = instance.angles.copy()

        self.trueVoltages = instance.trueVoltages.copy()
        self.trueAngles = instance.trueAngles.copy()

        ### Topology Data
        self.switchErrors = instance.switchErrors
        self.instance.openSwitchFlowsIdx = self.extract_open_switch_flows()

        ### True Measurements
        self.externalSimulationTrue = instance.externalSimulationTrue
        
        ### States
        voltageState = []
        thetaState = []

        ### Bad Data
        # self.badData_WLS = instance.badData_WLS
        # self.badData_LAV = instance.badData_LAV
        self.badData_WLS = pd.DataFrame(columns=["Measurement Idx", "Type", "Phase", "Solver"])
        self.badData_LAV = pd.DataFrame(columns=["Measurement Idx", "Type", "Phase", "Solver"])

    ### Measurement Functions
    def createMeasurementFunction(self, voltageState=None, thetaState=None, swStates=None, voltageMeas=None,realInjection=None, reacInjection=None, realFlow=None, reacFlow=None, voltageMagPMU=None, voltageAnglePMU=None, switchData=None,Ybus=None):
        G = np.real(Ybus)
        B = np.imag(Ybus)
        h_API = np.zeros(len(realInjection))
        h_RPI = np.zeros(len(reacInjection))
        for node in range(len(realInjection)):
            fromNode = self.nodeOrder.index(realInjection[node, 0].upper() + "." + realInjection[node, 1])
            connectedLines = np.where(self.busTobusConnection[fromNode, :] != 0)[0]
            for lineNode in connectedLines:
                if len(switchData) != 0 and len(np.intersect1d(np.where((switchData[:, 1].astype(int) == fromNode)),np.where((switchData[:, 3].astype(int) == lineNode)))) > 0:
                    idx = np.where((switchData[:, 1].astype(int) == fromNode) & (switchData[:, 3].astype(int) == lineNode))[0]
                    h_API[node] += swStates[idx[0]]
                    h_RPI[node] += swStates[idx[0] + len(switchData)]
                elif len(switchData) != 0 and len(np.intersect1d(np.where((switchData[:, 3].astype(int) == fromNode)),np.where((switchData[:, 1].astype(int) == lineNode)))) > 0:
                    idx = np.where((switchData[:, 3].astype(int) == fromNode) & (switchData[:, 1].astype(int) == lineNode))[0]
                    h_API[node] += -swStates[idx[0]]
                    h_RPI[node] += -swStates[idx[0] + len(switchData)]
                else:
                    Gprime = G[fromNode, lineNode]
                    Bprime = B[fromNode, lineNode]
                    ### Real Power Injection
                    h_API[node] += voltageState[fromNode] * voltageState[lineNode] * (Gprime * np.cos(thetaState[fromNode] - thetaState[lineNode]) + Bprime * np.sin(thetaState[fromNode] - thetaState[lineNode]))
                    ### Reactive Power Injection
                    h_RPI[node] += voltageState[fromNode] * voltageState[lineNode] * (Gprime * np.sin(thetaState[fromNode] - thetaState[lineNode]) - Bprime * np.cos(thetaState[fromNode] - thetaState[lineNode]))


        h_APF = np.zeros(len(realFlow))
        h_RPF = np.zeros(len(reacFlow))
        for node in range(len(realFlow)):
            fromNode = realFlow[node][1].astype(int)
            numPhase = np.where((realFlow[:, 0] == realFlow[node][0]) & (realFlow[:, 2] == realFlow[node][2]))[0]
            if len(switchData) != 0 and len(np.intersect1d(np.where((switchData[:, 1].astype(int) == fromNode)),np.where((switchData[:, 3].astype(int) == realFlow[node][3].astype(int))))) > 0:
                idx = np.where((switchData[:, 1].astype(int) == fromNode) & (switchData[:, 3].astype(int) == realFlow[node][3].astype(int)))[0]
                h_APF[node] = swStates[idx[0]]
                h_RPF[node] = swStates[idx[0] + len(switchData)]
            elif len(switchData) != 0 and len(np.intersect1d(np.where((switchData[:, 3].astype(int) == fromNode)),np.where((switchData[:, 1].astype(int) == realFlow[node][3].astype(int))))) > 0:
                idx = np.where((switchData[:, 3].astype(int) == fromNode) & (switchData[:, 1].astype(int) == realFlow[node][3].astype(int)))[0]
                h_APF[node] = -swStates[idx]
                h_RPF[node] = -swStates[idx + len(switchData)]
            else: 
                for phase in range(len(numPhase)):  ### How many phase are there will be found and change with the number 3
                    ### Real Power Flow
                    dummyfromNode = realFlow[numPhase[phase]][1].astype(int)
                    toNode = realFlow[numPhase[phase]][3].astype(int)
                    
                    Bprime = B[fromNode, toNode] - self.lineSusceptance[fromNode, toNode]
                    Gprime = G[fromNode, toNode] - self.lineConductance[fromNode, toNode]
 
                    
                    h_APF[node] += (voltageState[toNode] * (G[fromNode, toNode] * np.cos(thetaState[fromNode] - thetaState[toNode]) + B[fromNode, toNode] * np.sin(thetaState[fromNode] - thetaState[toNode]))) - (voltageState[dummyfromNode] * (Gprime * np.cos(
                                       thetaState[fromNode] - thetaState[dummyfromNode]) + Bprime * np.sin(thetaState[fromNode] - thetaState[dummyfromNode])))
                    ### Reactive Power Flow
                    h_RPF[node] += (voltageState[toNode] * (G[fromNode, toNode] * np.sin(thetaState[fromNode] - thetaState[toNode]) - B[fromNode, toNode] * np.cos(thetaState[fromNode] - thetaState[toNode]))) - (voltageState[dummyfromNode] * (Gprime * np.sin(
                                       thetaState[fromNode] - thetaState[dummyfromNode]) - Bprime * np.cos(thetaState[fromNode] - thetaState[dummyfromNode])))

                h_APF[node] = h_APF[node] * voltageState[fromNode]
                h_RPF[node] = h_RPF[node] * voltageState[fromNode]

        h_V = np.zeros(len(voltageMeas))
        for node in range(len(voltageMeas)):
            idx = self.nodeOrder.index((voltageMeas[node, 0] + '.' + voltageMeas[node, 1]).upper())
            h_V[node] = voltageState[idx]

        h_V_PMU = np.zeros(len(voltageMagPMU))
        for node in range(len(voltageMagPMU)):
            idx = self.nodeOrder.index((voltageMagPMU[node, 0] + '.' + voltageMagPMU[node, 1]).upper())
            h_V_PMU[node] = voltageState[idx]
        
        h_theta_PMU = np.zeros(len(voltageAnglePMU))
        for node in range(len(voltageAnglePMU)):
            idx = self.nodeOrder.index((voltageAnglePMU[node, 0] + '.' + voltageAnglePMU[node, 1]).upper())
            h_theta_PMU[node] = thetaState[idx]

        if len(switchData) != 0:
            numClosedSwitches = len(np.where(switchData[:, 5].astype(int) == 0)[0])
        else:
            numClosedSwitches = 0

        h_theta_diff_sw = np.zeros(numClosedSwitches)  ### kapali switch modelinde olmayacak bunlar
        counter = 0
        for node in range(len(switchData)):
            if switchData[node, 5].astype(int) == 0:
                fromNode = switchData[node, 1].astype(int)
                toNode = switchData[node, 3].astype(int)
                h_theta_diff_sw[counter] = thetaState[fromNode] - thetaState[toNode]
                counter += 1

        h_v_diff_sw = np.zeros(numClosedSwitches)
        counter = 0
        for node in range(len(switchData)):
            if switchData[node, 5].astype(int) == 0:
                fromNode = switchData[node, 1].astype(int)
                toNode = switchData[node, 3].astype(int)
                h_v_diff_sw[counter] = voltageState[fromNode] - voltageState[toNode]
                counter += 1

        h_P_closed_sw = np.zeros(len(self.instance.openSwitchFlowsIdx))
        counter = 0
        for node in range(len(switchData)):
            if switchData[node, 5].astype(int) == 1:
                fromNode = switchData[node, 1].astype(int)
                toNode = switchData[node, 3].astype(int)
                h_P_closed_sw[counter] = swStates[node]
                counter += 1

        h_Q_closed_sw = np.zeros(len(self.instance.openSwitchFlowsIdx))
        counter = 0
        for node in range(len(switchData)):
            if switchData[node, 5].astype(int) == 1:
                fromNode = switchData[node, 1].astype(int)
                toNode = switchData[node, 3].astype(int)
                h_Q_closed_sw[counter] = swStates[node + len(switchData)]
                counter += 1

        h = np.concatenate(
            (h_API, h_RPI, h_APF, h_RPF, h_V, h_V_PMU, h_theta_PMU, h_theta_diff_sw, h_v_diff_sw, h_P_closed_sw, h_Q_closed_sw))

        return h

    ### Jacobian Functions
    def createJacobian(self, voltageState=None, thetaState=None, swStates=None, voltageMeas=None, realInjection=None,reacInjection=None, realFlow=None, reacFlow=None, voltageMagPMU=None, voltageAnglePMU=None, switchData=None, Ybus=None):
        G = np.real(Ybus)
        B = np.imag(Ybus)
        J_API = np.zeros((len(realInjection), self.systemSize * 2 + len(swStates)))
        J_RPI = np.zeros((len(reacInjection), self.systemSize * 2 + len(swStates)))
        for node in range(len(realInjection)):
            fromNode = self.nodeOrder.index(realInjection[node, 0].upper() + "." + realInjection[node, 1])
            allFromNodes = np.where(realInjection[:, 0] == realInjection[node, 0])[0]
            # connectedLines = np.where(Ybus[node, :] != 0)[0]
            connectedLines = np.where(self.busTobusConnection[fromNode, :] != 0)[0]
            for lineNode in connectedLines:
                if len(switchData) != 0 and len(np.intersect1d(np.where((switchData[:, 1].astype(int) == fromNode)),
                                                               np.where((switchData[:, 3].astype(
                                                                   int) == lineNode)))) > 0:
                    idx = \
                        np.where(
                            (switchData[:, 1].astype(int) == fromNode) & (switchData[:, 3].astype(int) == lineNode))[0]
                    J_API[node, self.systemSize * 2 + idx] = 1
                    J_RPI[node, self.systemSize * 2 + len(switchData) + idx] = 1
                elif len(switchData) != 0 and len(np.intersect1d(np.where((switchData[:, 3].astype(int) == fromNode)),
                                                                 np.where((switchData[:, 1].astype(
                                                                     int) == lineNode)))) > 0:
                    idx = \
                        np.where(
                            (switchData[:, 3].astype(int) == fromNode) & (switchData[:, 1].astype(int) == lineNode))[0]
                    J_API[node, self.systemSize * 2 + idx] = -1
                    J_RPI[node, self.systemSize * 2 + len(switchData) + idx] = -1
                else:
                    Gprime = G[fromNode, lineNode]
                    Bprime = B[fromNode, lineNode]
                    if fromNode == lineNode:
                        ### Real Power Injection derivative Voltage From Bus
                        J_API[node, fromNode + self.systemSize] += 2 * (voltageState[fromNode] * Gprime)
                        ### Reactive Power Injection derivative Voltage From Bus)
                        J_RPI[node, fromNode + self.systemSize] += -2 * (voltageState[fromNode] * Bprime)
                    else:
                        ### Real Power Injection derivative Theta From Bus
                        J_API[node, fromNode] += voltageState[fromNode] * voltageState[lineNode] * (
                                -Gprime * np.sin(thetaState[fromNode] - thetaState[lineNode]) + Bprime * np.cos(
                            thetaState[fromNode] - thetaState[lineNode]))
                        ### Real Power Injection derivative Theta To Bus
                        J_API[node, lineNode] = voltageState[fromNode] * voltageState[lineNode] * (
                                Gprime * np.sin(thetaState[fromNode] - thetaState[lineNode]) - Bprime * np.cos(
                            thetaState[fromNode] - thetaState[lineNode]))

                        ### Reactive Power Injection derivative Theta From Bus
                        J_RPI[node, fromNode] += voltageState[fromNode] * voltageState[lineNode] * (
                                Gprime * np.cos(thetaState[fromNode] - thetaState[lineNode]) + Bprime * np.sin(
                            thetaState[fromNode] - thetaState[lineNode]))
                        ### Reactive Power Injection derivative Theta To Bus
                        J_RPI[node, lineNode] = -voltageState[fromNode] * voltageState[lineNode] * (
                                Gprime * np.cos(thetaState[fromNode] - thetaState[lineNode]) + Bprime * np.sin(
                            thetaState[fromNode] - thetaState[lineNode]))

                        ### Real Power Injection derivative Voltage From Bus
                        J_API[node, fromNode + self.systemSize] += voltageState[lineNode] * (
                                Gprime * np.cos(thetaState[fromNode] - thetaState[lineNode]) + Bprime * np.sin(
                            thetaState[fromNode] - thetaState[lineNode]))
                        ### Real Power Injection derivative Voltage To Bus
                        J_API[node, lineNode + self.systemSize] = voltageState[fromNode] * (
                                Gprime * np.cos(thetaState[fromNode] - thetaState[lineNode]) + Bprime * np.sin(
                            thetaState[fromNode] - thetaState[lineNode]))

                        ### Reactive Power Injection derivative Voltage From Bus
                        J_RPI[node, fromNode + self.systemSize] += voltageState[lineNode] * (
                                Gprime * np.sin(thetaState[fromNode] - thetaState[lineNode]) - Bprime * np.cos(
                            thetaState[fromNode] - thetaState[lineNode]))
                        ### Reactive Power Injection derivative Voltage To Bus
                        J_RPI[node, lineNode + self.systemSize] = voltageState[fromNode] * (
                                Gprime * np.sin(thetaState[fromNode] - thetaState[lineNode]) - Bprime * np.cos(
                            thetaState[fromNode] - thetaState[lineNode]))

        J_APF = np.zeros((len(realFlow), self.systemSize * 2 + len(swStates)))
        J_RPF = np.zeros((len(reacFlow), self.systemSize * 2 + len(swStates)))
        for node in range(len(realFlow)):
            fromNode = realFlow[node][1].astype(int)
            numPhase = np.where((realFlow[:, 0] == realFlow[node][0]) & (realFlow[:, 2] == realFlow[node][2]))[0]
            if len(switchData) != 0 and len(np.intersect1d(np.where((switchData[:, 1].astype(int) == fromNode)),
                                                           np.where((switchData[:, 3].astype(int) == realFlow[node][
                                                               3].astype(int))))) > 0:
                idx = np.where((switchData[:, 1].astype(int) == fromNode) & (
                        switchData[:, 3].astype(int) == realFlow[node][3].astype(int)))[0]
                J_APF[node, self.systemSize * 2 + idx] = 1
                J_RPF[node, self.systemSize * 2 + len(switchData) + idx] = 1
            elif len(switchData) != 0 and len(np.intersect1d(np.where((switchData[:, 3].astype(int) == fromNode)),
                                                             np.where((switchData[:, 1].astype(int) == realFlow[node][
                                                                 3].astype(int))))) > 0:
                idx = np.where((switchData[:, 3].astype(int) == fromNode) & (
                        switchData[:, 1].astype(int) == realFlow[node][3].astype(int)))[0]
                J_APF[node, self.systemSize * 2 + idx] = -1
                J_RPF[node, self.systemSize * 2 + len(switchData) + idx] = -1
            else:
                for phase in range(len(numPhase)):
                    dummyfromNode = realFlow[numPhase[phase]][1].astype(int)
                    toNode = realFlow[numPhase[phase]][3].astype(int)
                    #### Active Power Flow
                    Bprime = B[fromNode, toNode] - self.lineSusceptance[fromNode, toNode]
                    Gprime = G[fromNode, toNode] - self.lineConductance[fromNode, toNode]

                    if dummyfromNode == fromNode:
                        ### Real Power Flow derivative Theta From Self
                        J_APF[node, fromNode] += voltageState[toNode] * (-G[fromNode, toNode] * np.sin(thetaState[fromNode] - thetaState[toNode]) + B[fromNode, toNode] * np.cos(thetaState[fromNode] - thetaState[toNode]))
                        ### Real Power Flow derivative Voltage From Self
                        J_APF[node, fromNode + self.systemSize] += voltageState[toNode] * (G[fromNode, toNode] * np.cos(thetaState[fromNode] - thetaState[toNode]) + B[fromNode, toNode] * np.sin(thetaState[fromNode] - thetaState[toNode])) - 2 * \
                                                                   voltageState[fromNode] * (Gprime * np.cos(thetaState[fromNode] - thetaState[dummyfromNode]) + Bprime * np.sin(thetaState[fromNode] - thetaState[dummyfromNode]))
                    else:
                        ### Real Power Flow derivative Theta From Self
                        J_APF[node, fromNode] += voltageState[toNode] * (-G[fromNode, toNode] * np.sin(thetaState[fromNode] - thetaState[toNode]) + B[fromNode, toNode] * np.cos(thetaState[fromNode] - thetaState[toNode])) - voltageState[
                                                     dummyfromNode] * (-Gprime * np.sin(thetaState[fromNode] - thetaState[dummyfromNode]) + Bprime * np.cos(thetaState[fromNode] - thetaState[dummyfromNode]))
                        ### Real Power Flow derivative Theta From Dummy
                        J_APF[node, dummyfromNode] = -voltageState[fromNode] * voltageState[dummyfromNode] * (Gprime * np.sin(thetaState[fromNode] - thetaState[dummyfromNode]) - Bprime * np.cos(thetaState[fromNode] - thetaState[dummyfromNode]))

                        ### Real Power Flow derivative Voltage From Self
                        J_APF[node, fromNode + self.systemSize] += voltageState[toNode] * (G[fromNode, toNode] * np.cos(thetaState[fromNode] - thetaState[toNode]) + B[fromNode, toNode] * np.sin(thetaState[fromNode] - thetaState[toNode])) - voltageState[
                                                                       dummyfromNode] * (Gprime * np.cos(thetaState[fromNode] - thetaState[dummyfromNode]) + Bprime * np.sin(thetaState[fromNode] - thetaState[dummyfromNode]))
                        ### Real Power Flow derivative Voltage From Dummy
                        J_APF[node, dummyfromNode + self.systemSize] = -voltageState[fromNode] * (Gprime * np.cos(thetaState[fromNode] - thetaState[dummyfromNode]) + Bprime * np.sin(thetaState[fromNode] - thetaState[dummyfromNode]))

                    ### Real Power Flow derivative Theta To Bus
                    J_APF[node, toNode] = voltageState[fromNode] * voltageState[toNode] * (G[fromNode, toNode] * np.sin(thetaState[fromNode] - thetaState[toNode]) - B[fromNode, toNode] * np.cos(thetaState[fromNode] - thetaState[toNode]))
                    ### Real Power Flow derivative Voltage To Bus
                    J_APF[node, toNode + self.systemSize] = voltageState[fromNode] * (G[fromNode, toNode] * np.cos(thetaState[fromNode] - thetaState[toNode]) + B[fromNode, toNode] * np.sin(thetaState[fromNode] - thetaState[toNode]))

                    #### Reactive Power Flow

                    if dummyfromNode == fromNode:
                        ### Reactive Power Flow derivative Theta From Self
                        J_RPF[node, fromNode] += voltageState[toNode] * (
                                G[fromNode, toNode] * np.cos(thetaState[fromNode] - thetaState[toNode]) + B[
                            fromNode, toNode] * np.sin(thetaState[fromNode] - thetaState[toNode]))
                        ### Reactive Power Flow derivative Voltage From Self
                        J_RPF[node, fromNode + self.systemSize] += voltageState[toNode] * (
                                G[fromNode, toNode] * np.sin(thetaState[fromNode] - thetaState[toNode]) - B[
                            fromNode, toNode] * np.cos(thetaState[fromNode] - thetaState[toNode])) - 2 * \
                                                                   voltageState[fromNode] * (Gprime * np.sin(
                            thetaState[fromNode] - thetaState[dummyfromNode]) - Bprime * np.cos(
                            thetaState[fromNode] - thetaState[dummyfromNode]))
                    else:
                        ### Reactive Power Flow derivative Theta From Self
                        J_RPF[node, fromNode] += voltageState[toNode] * (
                                G[fromNode, toNode] * np.cos(thetaState[fromNode] - thetaState[toNode]) + B[
                            fromNode, toNode] * np.sin(thetaState[fromNode] - thetaState[toNode])) - voltageState[
                                                     dummyfromNode] * (Gprime * np.cos(
                            thetaState[fromNode] - thetaState[dummyfromNode]) + Bprime * np.sin(
                            thetaState[fromNode] - thetaState[dummyfromNode]))
                        ### Reactive Power Flow derivative Theta From Dummy
                        J_RPF[node, dummyfromNode] = -voltageState[fromNode] * voltageState[dummyfromNode] * (
                                -Gprime * np.cos(
                            thetaState[fromNode] - thetaState[dummyfromNode]) - Bprime * np.sin(
                            thetaState[fromNode] - thetaState[dummyfromNode]))

                        ### Reactive Power Flow derivative Voltage From Self
                        J_RPF[node, fromNode + self.systemSize] += voltageState[toNode] * (
                                G[fromNode, toNode] * np.sin(thetaState[fromNode] - thetaState[toNode]) - B[
                            fromNode, toNode] * np.cos(thetaState[fromNode] - thetaState[toNode])) - voltageState[
                                                                       dummyfromNode] * (Gprime * np.sin(
                            thetaState[fromNode] - thetaState[dummyfromNode]) - Bprime * np.cos(
                            thetaState[fromNode] - thetaState[dummyfromNode]))
                        ### Reactive Power Flow derivative Voltage From Dummy
                        J_RPF[node, dummyfromNode + self.systemSize] = -voltageState[fromNode] * (
                                Gprime * np.sin(thetaState[fromNode] - thetaState[dummyfromNode]) - Bprime * np.cos(
                            thetaState[fromNode] - thetaState[dummyfromNode]))

                    ### Reactive Power Flow derivative Theta To Bus
                    J_RPF[node, toNode] = voltageState[fromNode] * voltageState[toNode] * (
                            -G[fromNode, toNode] * np.cos(thetaState[fromNode] - thetaState[toNode]) - B[
                        fromNode, toNode] * np.sin(thetaState[fromNode] - thetaState[toNode]))
                    ### Reactive Power Flow derivative Voltage To Bus
                    J_RPF[node, toNode + self.systemSize] = voltageState[fromNode] * (
                            G[fromNode, toNode] * np.sin(thetaState[fromNode] - thetaState[toNode]) - B[
                        fromNode, toNode] * np.cos(thetaState[fromNode] - thetaState[toNode]))

                J_APF[node, fromNode] = J_APF[node, fromNode] * voltageState[fromNode]
                J_RPF[node, fromNode] = J_RPF[node, fromNode] * voltageState[fromNode]

        J_V = np.zeros((len(voltageMeas), self.systemSize))
        for node in range(len(voltageMeas)):
            idx = self.nodeOrder.index((voltageMeas[node, 0] + '.' + voltageMeas[node, 1]).upper())
            J_V[node, idx] = 1

        J_V_PMU = np.zeros((len(voltageMagPMU), self.systemSize * 2 + len(swStates)))
        for node in range(len(voltageMagPMU)):
            idx = self.nodeOrder.index((voltageMagPMU[node, 0] + '.' + voltageMagPMU[node, 1]).upper())
            J_V_PMU[node, self.systemSize+idx] = 1

        J_Angle_PMU = np.zeros((len(voltageAnglePMU), self.systemSize * 2 + len(swStates)))
        for node in range(len(voltageAnglePMU)):
            idx = self.nodeOrder.index((voltageAnglePMU[node, 0] + '.' + voltageAnglePMU[node, 1]).upper())
            J_Angle_PMU[node, idx] = 1

        if len(switchData) != 0:
            numClosedSwitches = len(np.where(switchData[:, 5].astype(int) == 0)[0])
        else:
            numClosedSwitches = 0

        J_theta_diff_sw = np.zeros((numClosedSwitches, self.systemSize * 2 + len(swStates)))
        counter = 0
        for node in range(len(switchData)):
            if switchData[node, 5].astype(int) == 0:
                fromNode = switchData[node, 1].astype(int)
                toNode = switchData[node, 3].astype(int)
                J_theta_diff_sw[counter, fromNode] = 1
                J_theta_diff_sw[counter, toNode] = -1
                counter += 1

        J_v_diff_sw = np.zeros((numClosedSwitches, self.systemSize * 2 + len(swStates)))
        counter = 0
        for node in range(len(switchData)):
            if switchData[node, 5].astype(int) == 0:
                fromNode = switchData[node, 1].astype(int)
                toNode = switchData[node, 3].astype(int)
                J_v_diff_sw[counter, fromNode + self.systemSize] = 1
                J_v_diff_sw[counter, toNode + self.systemSize] = -1
                counter += 1

        J_P_closed_sw = np.zeros((len(self.instance.openSwitchFlowsIdx), self.systemSize * 2 + len(swStates)))
        counter = 0
        for node in range(len(switchData)):
            if switchData[node, 5].astype(int) == 1:
                fromNode = switchData[node, 1].astype(int)
                toNode = switchData[node, 3].astype(int)
                J_P_closed_sw[counter, self.systemSize * 2 + node] = 1
                counter += 1

        J_Q_closed_sw = np.zeros((len(self.instance.openSwitchFlowsIdx), self.systemSize * 2 + len(swStates)))
        counter = 0
        for node in range(len(switchData)):
            if switchData[node, 5].astype(int) == 1:
                fromNode = switchData[node, 1].astype(int)
                toNode = switchData[node, 3].astype(int)
                J_Q_closed_sw[counter, self.systemSize * 2 + len(switchData) + node] = 1
                counter += 1

        J = np.concatenate((J_API, J_RPI, J_APF, J_RPF, np.concatenate((np.zeros((J_V.shape[0], J_V.shape[1])), J_V, np.zeros((J_V.shape[0], len(swStates)))), axis=1), J_V_PMU, J_Angle_PMU,
                            J_theta_diff_sw, J_v_diff_sw, J_P_closed_sw, J_Q_closed_sw))
        return J

    def StateEstimatorSettings(self):
        if self.instance.lagrangeSolver:
            numClosedSwitches = len(np.where(self.instance.switchesIsland[:, 5].astype(int) == 0)[0])
            W = np.concatenate([
                # 2500 * np.ones(len(self.realInjection)),
                # 2500 * np.ones(len(self.reacInjection)),
                # 2500 * np.ones(len(self.realFlow)),
                # 2500 * np.ones(len(self.reacFlow)),
                # 10000 * np.ones(len(self.voltages))
                1/0.01 * np.ones(len(self.realInjection)),
                1/0.01 * np.ones(len(self.reacInjection)),
                1/0.008 * np.ones(len(self.realFlow)),
                1/0.008 * np.ones(len(self.reacFlow)),
                1/0.004 * np.ones(len(self.voltages)),
                1/0.0001 * np.ones(len(self.voltageMagPMU)),
                1/0.0001 * np.ones(len(self.voltageAnglePMU))
            ])
        else:
            W = np.concatenate([
                1/0.01 * np.ones(len(self.realInjection)),
                1/0.01 * np.ones(len(self.reacInjection)),
                1/0.008 * np.ones(len(self.realFlow)),
                1/0.008 * np.ones(len(self.reacFlow)),
                1/0.004 * np.ones(len(self.voltages)),
                1/0.0001 * np.ones(len(self.voltageMagPMU)),
                1/0.0001 * np.ones(len(self.voltageAnglePMU))
            ])

        # Compute the inverse of the elements in W
        inverse_W = 1 / W
        # Create the R matrix as a diagonal matrix with the inverse_W elements
        R = np.diag(inverse_W)
        SEerror = 10
        threshold = 0.0001
        iterationLimit = 50
        # threshold = 0.001
        # threshold = 2
        
        firstBus = self.nodeOrder[0].split(".")[0]
        if len(self.voltageAnglePMU) != 0:
            refBus = []
        else:
            refBus = np.in1d(self.nodeOrder,[firstBus+".1",firstBus+".2",firstBus+".3"]).nonzero()[0]
        ### Island operation icersinde bakilmasi gerekiyor 3 ref bus var mi diye

        return R, SEerror, threshold, refBus, iterationLimit

    ### WLS State Estimation Related Functions
    def StateEstimation_WLS(self, voltageState=None, thetaState=None, swStates=None, voltageMeas=None,realInjection=None, reacInjection=None, realFlow=None, reacFlow=None, voltageMagPMU=None, voltageAnglePMU=None, switchData=None,Ybus=None):
        WLS_counter = 0
        if len(switchData) == 0:
            numClosedSwitches = 0
            self.instance.openSwitchFlowsIdx = []
        else:
            numClosedSwitches = len(np.where(switchData[:, 5].astype(int) == 0)[0])

        R, SEerror, threshold, refBus, iterationLimit = self.StateEstimatorSettings()

        Z = np.concatenate((realInjection[:, 2].astype(float), reacInjection[:, 2].astype(float),
                            realFlow[:, 5].astype(float), reacFlow[:, 5].astype(float), voltageMeas[:, 2].astype(float), voltageMagPMU[:,2].astype(float),
                            np.deg2rad(voltageAnglePMU[:,2].astype(float)), np.zeros(numClosedSwitches * 2), np.zeros(len(self.instance.openSwitchFlowsIdx) * 2)))
        
        # Z = np.concatenate((realInjection[:, 2].astype(float), reacInjection[:, 2].astype(float),
        #                     realFlow[:, 5].astype(float), reacFlow[:, 5].astype(float), voltageMagPMU[:,2].astype(float),
        #                     np.deg2rad(voltageAnglePMU[:,2].astype(float)), np.zeros(numClosedSwitches * 2), np.zeros(len(self.instance.openSwitchFlowsIdx) * 2)))
        
        inv_R = np.linalg.inv(R)
        SEerror_history = []
        iteration = 0

        while SEerror > threshold and WLS_counter < iterationLimit:
            iteration += 1
            h = self.createMeasurementFunction(voltageState=voltageState, thetaState=thetaState, swStates=swStates,
                                               voltageMeas=voltageMeas, realInjection=realInjection,
                                               reacInjection=reacInjection, realFlow=realFlow, reacFlow=reacFlow, voltageMagPMU = voltageMagPMU, voltageAnglePMU = voltageAnglePMU,switchData=switchData, Ybus=Ybus)
            H = self.createJacobian(voltageState=voltageState, thetaState=thetaState, swStates=swStates,
                                    voltageMeas=voltageMeas, realInjection=realInjection, reacInjection=reacInjection,
                                    realFlow=realFlow, reacFlow=reacFlow, voltageMagPMU=voltageMagPMU, voltageAnglePMU=voltageAnglePMU, switchData=switchData, Ybus=Ybus)
            # h = self.createMeasurementFunction(voltageState=self.voltages[:,2].astype(float), thetaState=np.deg2rad(self.angles[:,2].astype(float)), swStates=swStates,
            #                                    voltageMeas=voltageMeas, realInjection=realInjection,
            #                                    reacInjection=reacInjection, realFlow=realFlow, reacFlow=reacFlow, voltageMagPMU = voltageMagPMU, voltageAnglePMU = voltageAnglePMU,
            #                                    switchData=switchData, Ybus=Ybus)
            # H = self.createJacobian(voltageState=self.voltages[:,2].astype(float), thetaState=np.deg2rad(self.angles[:,2].astype(float)), swStates=swStates,
            #                         voltageMeas=voltageMeas, realInjection=realInjection, reacInjection=reacInjection,
            #                         realFlow=realFlow, reacFlow=reacFlow, switchData=switchData, Ybus=Ybus)

            H = np.delete(H,refBus,axis=1)

            residual = Z - h

            if self.instance.lagrangeSolver:
                # Augmented matrix
                aug_matrix = np.block([[np.diag(np.hstack([np.diag(R),np.zeros(numClosedSwitches * 2 + len(self.instance.openSwitchFlowsIdx) * 2)])),H],[H.T, np.zeros((H.shape[1], H.shape[1]))]])

                # Right-hand side vector
                rhs_vector = np.hstack([residual, np.zeros(H.shape[1])])

                # Solve the augmented system
                solution = np.linalg.solve(aug_matrix, rhs_vector)

                # Extract Lagrange multipliers and updated state estimates
                lambda_ = solution[:len(residual)]
                delta_x = solution[len(residual):]
            else:
                G = np.dot(H.T, np.dot(inv_R, H))
                # Q, R = qr(G, mode='economic')
                Q, R = np.linalg.qr(G)
                Qt_residual = np.dot(Q.T, np.dot(H.T, np.dot(inv_R, residual)))
                delta_x = np.linalg.solve(R, Qt_residual)
                # G = np.dot(H.T, np.dot(inv_R, H))
                # inv_G = np.linalg.inv(G)
                # delta_x = np.dot(inv_G, np.dot(H.T, np.dot(inv_R, residual)))
                aug_matrix = []
                lambda_ = []

            if len(switchData) == 0:
                idx = np.setdiff1d(range(0,self.systemSize),refBus)
                # thetaState[refBus:self.systemSize] += delta_x[:self.systemSize - refBus]
                thetaState[idx] += delta_x[:self.systemSize - len(refBus)]
                voltageState += delta_x[self.systemSize - len(refBus):]
            else:
                idx = np.setdiff1d(range(0,self.systemSize),refBus)
                # thetaState[refBus:self.systemSize] += delta_x[:self.systemSize - refBus]
                thetaState[idx] += delta_x[:self.systemSize - len(refBus)]
                voltageState += delta_x[self.systemSize - len(refBus):-len(swStates)]
                swStates += delta_x[-len(swStates):]

            SEerror = max(abs(delta_x))
            SEerror_history.append(SEerror)

            if iteration >= 3:
                avg_diff = np.mean([abs(SEerror_history[-1] - SEerror_history[-2]),
                                    abs(SEerror_history[-2] - SEerror_history[-3])])
                if avg_diff < 1e-7:
                    print(f"Terminated since SEerror does not change WLS!!! Seed Num: {self.instance.seedCounter}")
                    break
            print(SEerror)
            WLS_counter += 1

        if WLS_counter < iterationLimit:
            # print("WLS SE converged!")
            # mse_v = np.mean((voltageState - self.trueVoltages[self.instance.closedStateIdx, 2].astype(float)) ** 2)
            # # print(f"WLS Mean Squared Error between V and Vhat: {mse_v}")
            # mse_angle = np.mean((thetaState - np.deg2rad(self.trueAngles[self.instance.closedStateIdx, 2].astype(float))) ** 2)
            self.instance.overall_mse_WLS = np.mean((voltageState - self.trueVoltages[self.instance.closedStateIdx, 2].astype(float)) ** 2 + (thetaState - np.deg2rad(self.trueAngles[self.instance.closedStateIdx, 2].astype(float))) ** 2)
            print(f"Overall MSE WLS: {self.instance.overall_mse_WLS}")
            # print(f"WLS Mean Squared Error between theta and theta_hat: {mse_angle}")
        else:
            self.instance.lagrangeSolver = False
            voltageState = []
            thetaState = []
            swStates = []
            residual = []

        return voltageState, thetaState, swStates, aug_matrix, lambda_, residual

    def normalizedResidualTest(self, voltageState=None, thetaState=None, swStates=None, voltageMeas=None,realInjection=None, reacInjection=None, realFlow=None, reacFlow=None, voltageMagPMU=None, voltageAnglePMU=None, switchData=None,Ybus=None):
        R, SEerror, threshold, refBus, iterationLimit = self.StateEstimatorSettings()
        # Z = np.concatenate((realInjection[:, 2].astype(float), reacInjection[:, 2].astype(float),
        #                     realFlow[:, 5].astype(float), reacFlow[:, 5].astype(float),
        #                     voltageMeas[:, 2].astype(float)))
        
        Z = np.concatenate((realInjection[:, 2].astype(float), reacInjection[:, 2].astype(float),
                            realFlow[:, 5].astype(float), reacFlow[:, 5].astype(float),
                            voltageMeas[:, 2].astype(float), voltageMagPMU[:,2].astype(float),np.deg2rad(voltageAnglePMU[:,2].astype(float))))


        h = self.createMeasurementFunction(voltageState=voltageState, thetaState=thetaState, swStates=swStates,
                                           voltageMeas=voltageMeas, realInjection=realInjection,
                                           reacInjection=reacInjection, realFlow=realFlow,
                                           reacFlow=reacFlow, voltageMagPMU=voltageMagPMU, voltageAnglePMU=voltageAnglePMU, switchData=switchData, Ybus=Ybus)

        H = self.createJacobian(voltageState=voltageState, thetaState=thetaState, swStates=swStates,
                                voltageMeas=voltageMeas, realInjection=realInjection, reacInjection=reacInjection,
                                realFlow=realFlow, reacFlow=reacFlow, voltageMagPMU=voltageMagPMU, voltageAnglePMU=voltageAnglePMU, switchData=switchData, Ybus=Ybus)
        
        H = np.delete(H,refBus,axis=1)
        

       # Identity matrix of the same size as H*H.T
        I = np.eye(H.shape[0])

        # Compute G using H, inv_R
        G = H.T @ np.linalg.inv(R) @ H

        # Perform QR decomposition on G
        Q_G, R_G = np.linalg.qr(G)

        # Compute inv(R_G) using QR decomposition
        inv_R_G = np.linalg.inv(R_G)
        G_inv = np.dot(inv_R_G, Q_G.T)
                
        # G_inv = np.linalg.inv(G)
        S = I - H @ G_inv @ H.T @ np.linalg.inv(R)

        # # Ensure S is positive definite by adding a small value to the diagonals if necessary
        # S = S + np.eye(S.shape[0]) * 1e-8

        # Compute omega
        # Q_S, R_S = np.linalg.qr(S)
        # omega = np.linalg.solve(Q_S.T, R_S @ R)
        omega = S @ R
        omega_diag = np.diag(omega)

        residual_investigated = Z - h
        norm_res = np.abs(residual_investigated) / np.sqrt(omega_diag)
        
        max_norm_res_idx = np.argmax(norm_res)
        
        if max(norm_res) > 3:
            badDataFlag = True
            if max_norm_res_idx < len(realInjection):
                self.badData_WLS.loc[len(self.badData_WLS)] = [max_norm_res_idx, "P Injection", self.realInjection[max_norm_res_idx, 1], "WLS"]

                idx = max_norm_res_idx
                self.realInjection[idx, 2] = self.realInjection[idx, 2].astype(float) - (
                        R[max_norm_res_idx, max_norm_res_idx] / omega_diag[max_norm_res_idx]) * \
                                             residual_investigated[max_norm_res_idx]
            elif max_norm_res_idx < len(realInjection) + len(reacInjection):
                self.badData_WLS.loc[len(self.badData_WLS)] = [max_norm_res_idx, "Q Injection",
                                                       self.reacInjection[max_norm_res_idx - len(realInjection), 1],
                                                       "WLS"]

                idx = max_norm_res_idx - len(realInjection)

                self.reacInjection[idx, 2] = self.reacInjection[idx, 2].astype(float) - (
                        R[max_norm_res_idx, max_norm_res_idx] / omega_diag[max_norm_res_idx]) * \
                                             residual_investigated[max_norm_res_idx]
            elif max_norm_res_idx < 2 * len(realInjection) + len(realFlow):
                self.badData_WLS.loc[len(self.badData_WLS)] = [max_norm_res_idx, "P Flow",
                    self.realFlow[max_norm_res_idx - len(realInjection) - len(reacInjection), 4], "WLS"]

                idx = max_norm_res_idx - len(realInjection) - len(reacInjection)

                self.realFlow[idx, 5] = self.realFlow[idx, 5].astype(float) - (
                        R[max_norm_res_idx, max_norm_res_idx] / omega_diag[max_norm_res_idx]) * \
                                        residual_investigated[max_norm_res_idx]
            elif max_norm_res_idx < 2 * len(realInjection) + 2 * len(realFlow):
                self.badData_WLS.loc[len(self.badData_WLS)] = [max_norm_res_idx, "Q Flow",
                    self.reacFlow[max_norm_res_idx - 2 * len(realInjection) - len(realFlow), 4], "WLS"]

                idx = max_norm_res_idx - 2 * len(realInjection) - len(realFlow)

                self.reacFlow[idx, 5] = self.reacFlow[idx, 5].astype(float) - (
                        R[max_norm_res_idx, max_norm_res_idx] / omega_diag[max_norm_res_idx]) * \
                                        residual_investigated[max_norm_res_idx]
            elif max_norm_res_idx < 2 * len(realInjection) + 2 * len(realFlow) + len(voltageMeas):
                self.badData_WLS.loc[len(self.badData_WLS)] = [max_norm_res_idx,"Voltage Meas", self.voltages[max_norm_res_idx - 2 * len(realInjection) - 2 * len(realFlow), 1],"WLS"]

                idx = max_norm_res_idx - 2 * len(realInjection) - 2 * len(realFlow)

                self.voltages[idx, 2] = self.voltages[idx, 2].astype(float) - (
                        R[max_norm_res_idx, max_norm_res_idx] / omega_diag[max_norm_res_idx]) * \
                                        residual_investigated[max_norm_res_idx]
                                        
            elif max_norm_res_idx < 2 * len(realInjection) + 2 * len(realFlow) + len(voltageMeas) + len(voltageMagPMU):
                self.badData_WLS.loc[len(self.badData_WLS)] = [max_norm_res_idx,"Voltage Meas PMU", self.voltageMagPMU[max_norm_res_idx - 2 * len(realInjection) - 2 * len(realFlow) - len(voltageMeas), 1],"WLS"]
                idx = max_norm_res_idx - 2 * len(realInjection) - 2 * len(realFlow) - len(voltageMeas)
                self.voltageMagPMU[idx, 2] = self.voltageMagPMU[idx, 2].astype(float) - (
                        R[max_norm_res_idx, max_norm_res_idx] / omega_diag[max_norm_res_idx]) * \
                                        residual_investigated[max_norm_res_idx]
            
            elif max_norm_res_idx < 2 * len(realInjection) + 2 * len(realFlow) + len(voltageMeas) + len(voltageMagPMU) + len(voltageAnglePMU):
                self.badData_WLS.loc[len(self.badData_WLS)] = [max_norm_res_idx,"Voltage Angle PMU", self.voltageAnglePMU[max_norm_res_idx - 2 * len(realInjection) - 2 * len(realFlow) - len(voltageMeas) - len(voltageMagPMU), 1],"WLS"]
                idx = max_norm_res_idx - 2 * len(realInjection) - 2 * len(realFlow) - len(voltageMeas) - len(voltageMagPMU)
                self.voltageAnglePMU[idx, 2] = self.voltageAnglePMU[idx, 2].astype(float) - (
                        R[max_norm_res_idx, max_norm_res_idx] / omega_diag[max_norm_res_idx]) * \
                                        residual_investigated[max_norm_res_idx]
                
        else:
            badDataFlag = False
        return badDataFlag

    ### LAV State Estimation Related Functions
    def StateEstimation_LAV(self, voltageState=None, thetaState=None, swStates=None, voltageMeas=None,realInjection=None, reacInjection=None, realFlow=None, reacFlow=None, voltageMagPMU=None, voltageAnglePMU=None , switchData=None,Ybus=None):
        LAV_counter = 0
        if len(switchData) == 0:
            numClosedSwitches = 0
            self.instance.openSwitchFlowsIdx = []
        else:
            numClosedSwitches = len(np.where(switchData[:, 5].astype(int) == 0)[0])

        R, SEerror, threshold, refBus, iterationLimit = self.StateEstimatorSettings()



        # Z = np.concatenate((realInjection[:, 2].astype(float), reacInjection[:, 2].astype(float),
        #                     realFlow[:, 5].astype(float), reacFlow[:, 5].astype(float),
        #                     voltageMeas[:, 2].astype(float), np.zeros(numClosedSwitches * 2),
        #                     np.zeros(len(self.instance.openSwitchFlowsIdx) * 2)))
        
        Z = np.concatenate((realInjection[:, 2].astype(float), reacInjection[:, 2].astype(float),
                            realFlow[:, 5].astype(float), reacFlow[:, 5].astype(float), voltageMeas[:, 2].astype(float), voltageMagPMU[:,2].astype(float),
                            np.deg2rad(voltageAnglePMU[:,2].astype(float)), np.zeros(numClosedSwitches * 2), np.zeros(len(self.instance.openSwitchFlowsIdx) * 2)))


        iteration = 0
        SEerror_history = []

        while SEerror > threshold and LAV_counter < iterationLimit:
            iteration += 1
            h = self.createMeasurementFunction(voltageState=voltageState, thetaState=thetaState, swStates=swStates,voltageMeas=voltageMeas, realInjection=realInjection,reacInjection=reacInjection, realFlow=realFlow,reacFlow=reacFlow, voltageMagPMU = voltageMagPMU, voltageAnglePMU = voltageAnglePMU, switchData=switchData, Ybus=Ybus)
            H = self.createJacobian(voltageState=voltageState, thetaState=thetaState, swStates=swStates,voltageMeas=voltageMeas,realInjection=realInjection, reacInjection=reacInjection, realFlow=realFlow,reacFlow=reacFlow, voltageMagPMU = voltageMagPMU, voltageAnglePMU = voltageAnglePMU, switchData=switchData, Ybus=Ybus)

            # H = H[:, refBus:]
            H = np.delete(H,refBus,axis=1)

            residual = Z - h
            m, n = H.shape

            C = np.zeros(2 * n + 2 * m)
            # C[2 * n:] = np.concatenate([PSWeights, PSWeights])
            C[2 * n:] = 1

            A_eq = np.hstack([H, -H, np.eye(m), -np.eye(m)])
            b_eq = residual
            bounds = [(0, None)] * (2 * n + 2 * m)
            options = {
                'ipm_optimality_tolerance': 1e-5,  # Increase the maximum number of iterations
                'dual_feasibility_tolerance': 1e-5,  # Increase the maximum number of iterations
                'primal_feasibility_tolerance': 1e-5,  # Increase the maximum number of iterations
            }

            result = linprog(C, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs-ipm", options=options)

            if not result.success:
                self.instance.LAV_converged = False
                raise ValueError("Linear programming did not converge")

            x = result.x
            delta_x_plus = x[:n]
            delta_x_minus = x[n:2 * n]
            delta_x = delta_x_plus - delta_x_minus

            if len(switchData) != 0:
                idx = np.setdiff1d(range(0,self.systemSize),refBus)
                thetaState[idx] += delta_x[:self.systemSize - len(refBus)]
                voltageState += delta_x[self.systemSize - len(refBus):-len(swStates)]
                swStates += delta_x[-len(swStates):]
            else:
                idx = np.setdiff1d(range(0,self.systemSize),refBus)
                thetaState[idx] += delta_x[:self.systemSize - len(refBus)]
                voltageState += delta_x[self.systemSize - len(refBus):]

            SEerror = max(abs(delta_x))
            SEerror_history.append(SEerror)

            if iteration >= 3:
                avg_diff = np.mean([abs(SEerror_history[-1] - SEerror_history[-2]),
                                    abs(SEerror_history[-2] - SEerror_history[-3])])
                if avg_diff < 1e-7:
                    print(f"Terminated since SEerror does not change LAV!!! Seed Num: {self.instance.seedCounter}")
                    break

            print(SEerror)
            LAV_counter += 1
            
        if LAV_counter < iterationLimit:
            res_plus = x[2 * n:2 * n + m]
            res_minus = x[2 * n + m:]
            residual_LAV = res_plus - res_minus
            
            # print("LAV SE converged!")
            # mse_v = np.mean((voltageState - self.trueVoltages[self.instance.closedStateIdx, 2].astype(float)) ** 2)
            # print(f"LAV Mean Squared Error between V and Vhat: {mse_v}")
            # mse_angle = np.mean((thetaState - np.deg2rad(self.trueAngles[self.instance.closedStateIdx, 2].astype(float))) ** 2)
            # print(f"LAV Mean Squared Error between theta and theta_hat: {mse_angle}")
            self.instance.overall_mse_LAV = np.mean((voltageState - self.trueVoltages[self.instance.closedStateIdx, 2].astype(float)) ** 2 + (thetaState - np.deg2rad(self.trueAngles[self.instance.closedStateIdx, 2].astype(float))) ** 2)
            print(f"Overall MSE LAV: {self.instance.overall_mse_LAV}")
        else:
            self.instance.LAV_converged = False
            self.instance.overall_mse_LAV = -99
            raise ValueError("Linear programming did not converge")


        return residual_LAV, Z, H, R, numClosedSwitches

    def LAV_bad_data_detection(self, residuals=None, R=None, alpha=0.05, numClosedSwitches=0):

        measurement_variances = np.diag(R)
        # BadDataDetection = pd.DataFrame(columns=['Measurement Index', 'Hypothesis Testing'])

        # Hypothesis Testing with Residuals
        p_values = 2 * (1 - stats.norm.cdf(np.abs(residuals) / np.sqrt(measurement_variances)))
        hypothesis_testing_indices = np.where(p_values < alpha)[0]

        # # Normalized Residuals Test
        # normalized_residuals_test = np.abs(residuals) / np.sqrt(measurement_variances)
        # normalized_residuals_threshold = stats.norm.ppf(1 - alpha / 2)
        # normalized_residuals_threshold = 3
        # normalized_residuals_indices = np.where(normalized_residuals_test > normalized_residuals_threshold)[0]

        # Combine results into a DataFrame
        all_indices = hypothesis_testing_indices
        results = []

        # for idx in all_indices:
        #     results.append({
        #         'Measurement Index': idx,
        #         'Hypothesis Testing': idx in hypothesis_testing_indices,
        #         'Normalized Residuals': idx in normalized_residuals_indices
        #     })

        # BadDataDetection = pd.concat([BadDataDetection, pd.DataFrame(results)], ignore_index=True)

        # Store bad data information with bus numbers and types
        for idx in all_indices:
            if idx < len(self.realInjection):
                bus_number = self.realInjection[idx, 0]
                value = self.realInjection[idx, 1]
                type = "P Injection"
            elif idx < len(self.realInjection) + len(self.reacInjection):
                bus_number = self.reacInjection[idx - len(self.realInjection), 0]
                value = self.reacInjection[idx - len(self.realInjection), 1]
                type = "Q Injection"
            elif idx < len(self.realInjection) + len(self.reacInjection) + len(self.realFlow):
                bus_number = self.realFlow[idx - len(self.realInjection) - len(self.reacInjection), 0]
                phase = self.realFlow[idx - len(self.realInjection) - len(self.reacInjection), 2]
                value = self.realFlow[idx - len(self.realInjection) - len(self.reacInjection), 4]
                type = "P Flow"
            elif idx < len(self.realInjection) + len(self.reacInjection) + 2 * len(self.realFlow):
                bus_number = self.reacFlow[idx - 2 * len(self.realInjection) - len(self.realFlow), 0]
                phase = self.reacFlow[idx - 2 * len(self.realInjection) - len(self.realFlow), 2]
                value = self.reacFlow[idx - 2 * len(self.realInjection) - len(self.realFlow), 4]
                type = "Q Flow"
            elif idx < len(self.realInjection) + len(self.reacInjection) + 2 * len(self.realFlow) + len(self.voltages):
                bus_number = self.voltages[idx - 2 * len(self.realInjection) - 2 * len(self.realFlow), 0]
                value = self.voltages[idx - 2 * len(self.realInjection) - 2 * len(self.realFlow), 1]
                type = "Voltage Meas"
            elif idx < len(self.realInjection) + len(self.reacInjection) + 2 * len(self.realFlow) + len(self.voltages) + len(self.voltageMagPMU):
                bus_number = self.voltageMagPMU[idx - 2 * len(self.realInjection) - 2 * len(self.realFlow) - len(self.voltages), 0]
                value = self.voltageMagPMU[idx - 2 * len(self.realInjection) - 2 * len(self.realFlow) - len(self.voltages), 1]
                type = "Voltage PMU Meas"
            else:
                bus_number = self.voltageAnglePMU[idx - 2 * len(self.realInjection) - 2 * len(self.realFlow) - len(self.voltages) - len(self.voltageMagPMU), 0]
                value = self.voltageAnglePMU[idx - 2 * len(self.realInjection) - 2 * len(self.realFlow) - len(self.voltages) - len(self.voltageMagPMU), 1]
                type = "Angle PMU Meas"


            self.badData_LAV.loc[len(self.badData_LAV)] = [idx, type, value, 'LAV']
            self.instance.badData_LAV = self.badData_LAV
            # self.instance.badData_LAV[len(self.instance.badData_LAV)] = [idx, type, value, solver]

    ### Bad Data Detection and Identification
    def solve_LAV_badData(self, Ybus=None):

        residual_LAV, Z, H, R, numClosedSwitches = self.StateEstimation_LAV(voltageState=self.voltageInitial.astype(float), thetaState=np.deg2rad(self.thetaInitial),swStates=[], voltageMeas=self.voltages, realInjection=self.realInjection,reacInjection=self.reacInjection, realFlow=self.realFlow, reacFlow=self.reacFlow, voltageMagPMU=self.voltageMagPMU, voltageAnglePMU=self.voltageAnglePMU, switchData=[],Ybus=Ybus)
        self.LAV_bad_data_detection(residuals=residual_LAV, R=R,alpha=0.01)

    def solve_WLS_badData(self, Ybus=None,):
        badDataFlag = True
        while badDataFlag:
            voltageState, thetaState, swStates, aug_matrix, lambda_, residual = self.StateEstimation_WLS(
                voltageState=self.voltageInitial.astype(float), thetaState=np.deg2rad(self.thetaInitial), swStates=[],
                voltageMeas=self.voltages, realInjection=self.realInjection, reacInjection=self.reacInjection,
                realFlow=self.realFlow, reacFlow=self.reacFlow, voltageMagPMU=self.voltageMagPMU, voltageAnglePMU=self.voltageAnglePMU, switchData=[], Ybus=Ybus)
            if len(voltageState) != 0:
                badDataFlag = self.normalizedResidualTest(voltageState=voltageState, thetaState=thetaState, swStates=[],
                                                          voltageMeas=self.voltages, realInjection=self.realInjection,
                                                          reacInjection=self.reacInjection, realFlow=self.realFlow,
                                                          reacFlow=self.reacFlow, voltageMagPMU=self.voltageMagPMU, voltageAnglePMU=self.voltageAnglePMU, switchData=[], Ybus=Ybus)
                if not badDataFlag:
                    self.instance.badData_WLS = self.badData_WLS.drop_duplicates().sort_values(by='Measurement Idx', ascending=True).reset_index(drop=True)
                    self.voltageState = voltageState
                    self.thetaState = thetaState
                    break
            else:
                print('WLS is not converged !!')
                self.instance.overall_mse_WLS = -99
                self.instance.WLS_converged = False
                self.voltageState = voltageState
                self.thetaState = thetaState
                break

    ### Topology Error Identification
    def solve_WLS_topology(self, Ybus=None):
        while True:
            voltageState, thetaState, swStates, aug_matrix, lambda_, residual = self.StateEstimation_WLS(
                voltageState=self.voltageInitial.astype(float), thetaState=np.deg2rad(self.thetaInitial),
                swStates=self.initial_switchStates.astype(float), voltageMeas=self.voltages,
                realInjection=self.realInjection, reacInjection=self.reacInjection, realFlow=self.realFlow,
                reacFlow=self.reacFlow, voltageMagPMU=self.voltageMagPMU, voltageAnglePMU=self.voltageAnglePMU , switchData=self.instance.switchesIsland, Ybus=Ybus)
            if self.instance.lagrangeSolver:
                topology_errors, criticalPairFlag, criticalPairSwitches = self.topologyErrorIdentification(
                    voltageState=voltageState, thetaState=thetaState, swStates=swStates, aug_matrix=aug_matrix,
                    lambda_=lambda_, residual=residual, switchData=self.instance.switchesIsland)
                if criticalPairFlag:
                    states = ["Closed", "Open"]
                    initialSwitchStates = (
                        states[self.instance.switchesIsland[criticalPairSwitches[0] * 3, 5].astype(int)],
                        states[self.instance.switchesIsland[criticalPairSwitches[1] * 3, 5].astype(int)])
                    # Generate all combinations of states for two switches
                    combinations = list(itertools.product(states, repeat=2))
                    del combinations[combinations.index(initialSwitchStates)]

                    for combination in combinations:
                        self.set_switch_status(pd.DataFrame(
                            {"SwitchName": [self.instance.switchesIsland[criticalPairSwitches[0] * 3, 6]],
                             "Status": [combination[0]]}))
                        self.set_switch_status(pd.DataFrame(
                            {"SwitchName": [self.instance.switchesIsland[criticalPairSwitches[1] * 3, 6]],
                             "Status": [combination[1]]}))
                        self.instance.switchesIsland = self.get_switches(self.base_power_kw, Ybus)
                        self.instance.openSwitchFlowsIdx = self.extract_open_switch_flows()
                        voltageState, thetaState, swStates, aug_matrix, lambda_, residual = self.StateEstimation_WLS(
                            voltageState=self.voltageInitial.astype(float), thetaState=np.deg2rad(self.thetaInitial),
                            swStates=self.initial_switchStates.astype(float), voltageMeas=self.voltages,
                            realInjection=self.realInjection, reacInjection=self.reacInjection, realFlow=self.realFlow,
                            reacFlow=self.reacFlow, switchData=self.instance.switchesIsland, Ybus=Ybus)
                        stopBrutForce = self.brutForceLagrangeCheck(aug_matrix=aug_matrix, lambda_=lambda_,
                                                                    residual=residual,
                                                                    switchData=self.instance.switchesIsland,
                                                                    criticalPairSwitches=criticalPairSwitches,
                                                                    Combination=combination)
                        if stopBrutForce:
                            differentValueIdx = [i for i, (a, b) in enumerate(zip(initialSwitchStates, combination)) if
                                                 a != b]
                            for idx in differentValueIdx:
                                self.switchErrors.loc[len(self.switchErrors)] = [
                                    self.instance.switchesIsland[criticalPairSwitches[idx] * 3, 6],
                                    initialSwitchStates[idx], combination[idx]]
                            break
                else:
                    if any(topology_errors):
                        self.correctSwitchStatus(topology_errors=topology_errors,
                                                 switchData=self.instance.switchesIsland)
                        if len(self.switchErrors) == 0:
                            self.instance.lagrangeSolver = False
                    elif len(self.switchErrors) > 0:
                        break
                    else:
                        self.instance.lagrangeSolver = False
                        break
            else:
                self.instance.lagrangeSolver = False
                break

    def topologyErrorIdentification(self, voltageState=None, thetaState=None, swStates=None, aug_matrix=None, lambda_=None, residual=None, switchData=None):
        numClosedSwitches = len(np.where(switchData[:, 5].astype(int) == 0)[0])
        numOpenSwitches = len(self.instance.openSwitchFlowsIdx)

        closedSwitches = np.unique(divmod(np.where(switchData[:, 5].astype(int) == 0)[0], 3)[0])
        openSwitches = np.unique(divmod(np.where(switchData[:, 5].astype(int) == 1)[0], 3)[0])

        criticalPairFlag = False
        criticalPairSwitches = []
        # Normalize the Lagrange multipliers
        V = np.linalg.inv(aug_matrix)
        V = V[:len(residual), :len(residual)]
        V = V + np.eye(V.shape[0]) * 1e-8
        lambda_norm = abs(lambda_) / np.sqrt(V.diagonal())
        # Threshold for identifying topology errors
        threshold = 2.0
        # Identify switches with potential errors
        topology_errors = np.bool_(np.zeros(2 * numClosedSwitches + 2 * numOpenSwitches))
        if max(abs(lambda_norm[-(2 * numClosedSwitches + 2 * numOpenSwitches):])) > threshold and \
                np.where(abs(lambda_norm) == max(abs(lambda_norm)))[0][0] in range(
            len(lambda_norm) - 2 * numClosedSwitches - 2 * numOpenSwitches, len(lambda_norm)):
            lambda_norm_switchStates = lambda_norm[-(2 * numClosedSwitches + 2 * numOpenSwitches):]
            maxValue = max(abs(lambda_norm_switchStates))
            secmaxvalue = max(lambda_norm_switchStates,
                              key=lambda x: min(lambda_norm_switchStates) - 1 if (x == maxValue) else x)
            idx_max = divmod(np.where(abs(lambda_norm_switchStates) == maxValue)[0], 3)[0]
            idx_secmax = divmod(np.where(abs(lambda_norm_switchStates) == secmaxvalue)[0], 3)[0]

            if (idx_max[0] != idx_secmax[0] and abs(maxValue - secmaxvalue) < 0.1):
                idx1 = np.where(abs(lambda_norm_switchStates) == maxValue)[0]
                idx2 = np.where(abs(lambda_norm_switchStates) == secmaxvalue)[0]

                idx1SwitchIdx = divmod(idx1, numClosedSwitches)[1]
                idx1SwitchIdx = divmod(idx1SwitchIdx, 3)[0][0]
                idx2SwitchIdx = divmod(idx2, numClosedSwitches)[1]
                idx2SwitchIdx = divmod(idx2SwitchIdx, 3)[0][0]

                criticalPairFlag = True
                criticalPairSwitches = [closedSwitches[idx1SwitchIdx], closedSwitches[idx2SwitchIdx]]
            elif (len(idx_max) > 1 and idx_max[0] != idx_max[1]):
                idx1SwitchIdx = divmod(idx_max[0], numClosedSwitches)[1]
                idx2SwitchIdx = divmod(idx_max[1], numClosedSwitches)[1]
                criticalPairFlag = True
                criticalPairSwitches = [closedSwitches[idx1SwitchIdx], closedSwitches[idx2SwitchIdx]]
            else:
                idx = np.where(abs(lambda_norm_switchStates) == max(abs(lambda_norm_switchStates)))[0]
                if idx < 2 * numClosedSwitches:
                    if idx < numClosedSwitches:
                        switchNumber = divmod(idx, 3)[0][0]
                        topology_errors[switchNumber * 3:switchNumber * 3 + 3] = True
                        topology_errors[
                        switchNumber * 3 + numClosedSwitches:switchNumber * 3 + 3 + numClosedSwitches] = True
                    else:
                        idx = idx - numClosedSwitches
                        switchNumber = divmod(idx, 3)[0][0]
                        topology_errors[switchNumber * 3:switchNumber * 3 + 3] = True
                        topology_errors[
                        switchNumber * 3 + numClosedSwitches:switchNumber * 3 + 3 + numClosedSwitches] = True
                else:
                    idx = idx - 2 * numClosedSwitches
                    if idx < numOpenSwitches:
                        idx = idx
                        switchNumber = divmod(idx, 3)[0][0]
                        topology_errors[
                        numClosedSwitches * 2 + switchNumber * 3:numClosedSwitches * 2 + switchNumber * 3 + 3] = True
                        topology_errors[
                        switchNumber * 3 + numClosedSwitches * 2 + numOpenSwitches:switchNumber * 3 + 3 + numClosedSwitches * 2 + numOpenSwitches] = True
                    else:
                        idx = idx - numOpenSwitches
                        switchNumber = divmod(idx, 3)[0][0]
                        topology_errors[
                        numClosedSwitches * 2 + switchNumber * 3:numClosedSwitches * 2 + switchNumber * 3 + 3] = True
                        topology_errors[
                        switchNumber * 3 + numClosedSwitches * 2 + numOpenSwitches:switchNumber * 3 + 3 + numClosedSwitches * 2 + numOpenSwitches] = True

            # topology_errors = np.abs(lambda_norm[-(2*numClosedSwitches + 2*numOpenSwitches):]) > threshold
        else:
            topology_errors = np.bool_(np.zeros(2 * numClosedSwitches + 2 * numOpenSwitches))

        return topology_errors, criticalPairFlag, criticalPairSwitches

    def correctSwitchStatus(self, topology_errors=None, switchData=None):
        numClosedSwitches = len(np.where(switchData[:, 5].astype(int) == 0)[0])
        numOpenSwitches = len(self.instance.openSwitchFlowsIdx)

        closedSwitches = switchData[np.where(switchData[:, 5].astype(int) == 0)[0]]
        openSwitches = switchData[np.where(switchData[:, 5].astype(int) == 1)[0]]
        ### Closed Switch Errors
        teClosedSwitches = topology_errors[:numClosedSwitches]
        dummyLineName = '0'
        counter = 0
        for count, switch in enumerate(closedSwitches):
            if switch[6] != dummyLineName:
                if counter >= 2:
                    # print(f"Switch between {dummyLineName} is Actually Open but Assumed as Closed")
                    self.set_switch_status(pd.DataFrame({"SwitchName": [f"{dummyLineName}"], "Status": ["Open"]}))
                    idx = np.where(switchData[:, 6] == dummyLineName)[0]
                    self.instance.switchesIsland[idx, 5] = 1
                    self.switchErrors.loc[len(self.switchErrors)] = [dummyLineName, 'Closed', 'Open']
                counter = 0
            if teClosedSwitches[count]:
                counter += 1
            dummyLineName = switch[6]

        if counter >= 2:
            # print(f"Switch between {dummyLineName} is Actually Open but Assumed as Closed")
            self.set_switch_status(pd.DataFrame({"SwitchName": [f"{dummyLineName}"], "Status": ["Open"]}))
            idx = np.where(switchData[:, 6] == dummyLineName)[0]
            self.instance.switchesIsland[idx, 5] = 1
            self.switchErrors.loc[len(self.switchErrors)] = [dummyLineName, 'Closed', 'Open']

        ### Open Switch Errors
        teOpenSwitches = topology_errors[2 * numClosedSwitches:2 * numClosedSwitches + numOpenSwitches]
        dummyLineName = '0'
        counter = 0
        for count, switch in enumerate(openSwitches):
            if switch[6] != dummyLineName:
                if counter >= 2:
                    # print(f"Switch between {dummyLineName} is Actually Closed but Assumed as Open")
                    self.set_switch_status(pd.DataFrame({"SwitchName": [f"{dummyLineName}"], "Status": ["Closed"]}))
                    idx = np.where(switchData[:, 6] == dummyLineName)[0]
                    self.instance.switchesIsland[idx, 5] = 0
                    self.switchErrors.loc[len(self.switchErrors)] = [dummyLineName, 'Open', 'Closed']
                counter = 0
            if teOpenSwitches[count]:
                counter += 1
            dummyLineName = switch[6]

        if counter >= 2:
            # print(f"Switch between {dummyLineName} is Actually Closed but Assumed as Open")
            self.set_switch_status(pd.DataFrame({"SwitchName": [f"{dummyLineName}"], "Status": ["Closed"]}))
            idx = np.where(switchData[:, 6] == dummyLineName)[0]
            self.instance.switchesIsland[idx, 5] = 0
            self.switchErrors.loc[len(self.switchErrors)] = [dummyLineName, 'Open', 'Closed']

        self.instance.openSwitchFlowsIdx = self.extract_open_switch_flows()
        print('done')

    def brutForceLagrangeCheck(self, aug_matrix=None, lambda_=None, residual=None, switchData=None,criticalPairSwitches=None, Combination=None):
        numClosedSwitches = len(np.where(switchData[:, 5].astype(int) == 0)[0])
        numOpenSwitches = len(self.instance.openSwitchFlowsIdx)

        stopBrutForce = False
        # Normalize the Lagrange multipliers
        V = np.linalg.inv(aug_matrix)
        V = V[:len(residual), :len(residual)]
        V = V + np.eye(V.shape[0]) * 1e-8
        lambda_norm = abs(lambda_) / np.sqrt(V.diagonal())
        # Threshold for identifying topology errors
        threshold = 10.0
        if max(abs(lambda_norm[-(2 * numClosedSwitches + 2 * numOpenSwitches):])) > threshold and \
                np.where(abs(lambda_norm) == max(abs(lambda_norm[-(2 * numClosedSwitches + 2 * numOpenSwitches):])))[0][
                    0] in range(len(lambda_norm) - 2 * numClosedSwitches - 2 * numOpenSwitches, len(lambda_norm)):
            lambda_norm_switchStates = lambda_norm[-(2 * numClosedSwitches + 2 * numOpenSwitches):]
            closedSwitches = np.unique(divmod(np.where(switchData[:, 5].astype(int) == 0)[0], 3)[0])
            idx_closed = np.where(closedSwitches == np.intersect1d(closedSwitches, criticalPairSwitches))[0]
            lambda_closed = []
            for i in idx_closed:
                lambda_closed.append(lambda_norm_switchStates[i * 3:i * 3 + 3])
                lambda_closed.append(lambda_norm_switchStates[i * 3 + numClosedSwitches:i * 3 + 3 + numClosedSwitches])

            openSwitches = np.unique(divmod(np.where(switchData[:, 5].astype(int) == 1)[0], 3)[0])
            idx_open = np.where(openSwitches == np.intersect1d(openSwitches, criticalPairSwitches))[0]
            lambda_open = []
            for i in idx_open:
                lambda_open.append(
                    lambda_norm_switchStates[i * 3 + numClosedSwitches * 2:i * 3 + 3 + numClosedSwitches * 2])
                lambda_open.append(lambda_norm_switchStates[
                                   i * 3 + numClosedSwitches * 2 + numOpenSwitches:i * 3 + numClosedSwitches * 2 + 3 + numOpenSwitches])

            lambda_closed = np.array(lambda_closed).reshape(len(lambda_closed * 3), 1)
            lambda_open = np.array(lambda_open).reshape(len(lambda_open * 3), 1)

            if (len(lambda_closed) != 0 and max(lambda_closed) > threshold) or (
                    len(lambda_open) != 0 and max(lambda_open) > threshold):
                stopBrutForce = False
            else:
                stopBrutForce = True

            # maxValue = max(abs(lambda_norm_switchStates))
            # if np.where(abs(lambda_norm_switchStates) == maxValue)[0][0] < 2 * numClosedSwitches: ## to make it more robust, instead of using idx_max, exact indexes of critical pairs will be checked based on the threshold
            #     idx_max = divmod(np.where(abs(lambda_norm_switchStates) == maxValue)[0],3)[0]
            #     closedSwitches = np.unique(divmod(np.where(switchData[:,5].astype(int) == 0)[0],3)[0])
            #     if switchData[closedSwitches[idx_max]*3,6] in [switchData[criticalPairSwitches[0]*3,6],switchData[criticalPairSwitches[1]*3,6]]:
            #         stopBrutForce = False
            #     else:
            #         stopBrutForce = True
            # else:
            #     idx_max = divmod(np.where(abs(lambda_norm_switchStates) == maxValue)[0]-2*numClosedSwitches,3)[0]
            #     openSwitches = np.unique(divmod(np.where(switchData[:,5].astype(int) == 1)[0],3)[0])
            #     if switchData[openSwitches[idx_max]*3,6] in [switchData[criticalPairSwitches[0]*3,6],switchData[criticalPairSwitches[1]*3,6]]:
            #         stopBrutForce = False
            #     else:
            #         stopBrutForce = True
        else:
            stopBrutForce = True

        return stopBrutForce

    ### Switch Status Functions
    def set_switch_status(self, switchStatus=None):
        for index, row in switchStatus.iterrows():
            switch_name = row['SwitchName']
            status = row['Status'].lower()
            switch = self.instance.dss.ActiveCircuit.CktElements(f"Line.{switch_name}")
            if status == "open":
                switch.Open(2, 0)
            elif status == "closed":
                switch.Close(2, 0)

    def get_switches(self, base_power_kw, Ybus):
        switchData = []
        self.instance.dss.ActiveCircuit.Lines.First
        while True:
            if self.instance.dss.ActiveCircuit.Lines.IsSwitch:
                fromBus = self.instance.dss.ActiveCircuit.Lines.Bus1.split('.')[0]
                toBus = self.instance.dss.ActiveCircuit.Lines.Bus2.split('.')[0]
                num_phases = self.instance.dss.ActiveCircuit.Lines.Phases
                phase_order = self.instance.dss.ActiveCircuit.ActiveCktElement.NodeOrder
                powers = self.instance.dss.ActiveCircuit.CktElements.Powers  # Get real and reactive power in kW and kVAR
                switch = self.instance.dss.ActiveCircuit.CktElements(
                    f"Line.{self.instance.dss.ActiveCircuit.Lines.Name}")
                # self.instance.dss.ActiveCircuit.SetActiveElement(f"Line.{self.instance.dss.ActiveCircuit.Lines.Name}")
                switchStatus = int(switch.IsOpen(2, 0))
                for phase in range(num_phases):
                    if (fromBus + '.' + str(phase_order[phase])).upper() in self.nodeOrder and (
                            toBus + '.' + str(phase_order[phase])).upper() in self.nodeOrder:
                        fromIdx = self.nodeOrder.index((fromBus + '.' + str(phase_order[phase])).upper())
                        toIdx = self.nodeOrder.index((toBus + '.' + str(phase_order[phase])).upper())
                        Ybus[fromIdx, fromIdx] = Ybus[fromIdx, fromIdx] + Ybus[fromIdx, toIdx]
                        Ybus[toIdx, toIdx] = Ybus[toIdx, toIdx] + Ybus[toIdx, fromIdx]
                        Ybus[fromIdx, toIdx] = 0
                        Ybus[toIdx, fromIdx] = 0
                        idx = phase * 2
                        switchData.append([fromBus, fromIdx, toBus, toIdx, phase_order[phase], switchStatus,
                                           self.instance.dss.ActiveCircuit.Lines.Name])

            if not self.instance.dss.ActiveCircuit.Lines.Next > 0:
                break
        return np.array(switchData)

    def extract_open_switch_flows(self):
        open_switch_flows_idx = []
        deleteFlowIdx = []
        for switch in self.instance.switchesIsland:
            if switch[5].astype(int) == 1:
                idx = np.where(((self.realFlow[:, 0] == switch[0]) & ((self.realFlow[:, 2] == switch[2])) & (
                        self.realFlow[:, 4] == switch[4])) | (
                                       (self.realFlow[:, 0] == switch[2]) & ((self.realFlow[:, 2] == switch[0])) & (
                                       self.realFlow[:, 4] == switch[4])))[0]
                open_switch_flows_idx.append(idx)
                # deleteFlowIdx.append(idx)
                # idxInjection = np.where((self.realInjection[:,0] == switch[0]) | (self.realInjection[:,0] == switch[2]))[0]
            # else:
            #     idx = np.where(((self.realFlow[:,0] == switch[0]) & ((self.realFlow[:,2] == switch[2])) & (self.realFlow[:,4] == switch[4])) |  ((self.realFlow[:,0] == switch[2]) & ((self.realFlow[:,2] == switch[0])) & (self.realFlow[:,4] == switch[4])))[0]
            #     deleteFlowIdx.append(idx)

        # self.realFlow = np.delete(self.realFlow, deleteFlowIdx, axis=0)
        # self.reacFlow = np.delete(self.reacFlow, deleteFlowIdx, axis=0)

        # self.realFlow = np.delete(self.realFlow, open_switch_flows_idx, axis=0)
        # self.reacFlow = np.delete(self.reacFlow, open_switch_flows_idx, axis=0)
        # self.realInjection = np.delete(self.realInjection, idxInjection, axis=0)
        # self.reacInjection = np.delete(self.reacInjection, idxInjection, axis=0)

        return open_switch_flows_idx
