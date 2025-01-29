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
from collections import defaultdict


class Observability:
    def __init__(self, obsInstance):
        self.obsInstance = obsInstance
    
    ### Jacobian Functions
    def createJacobian(self, realInjection = None, realFlow = None, voltageAnglePMU = None):
        # realInjection = np.delete(realInjection, np.where((realInjection[:,0] == '684') | (realInjection[:,0] == '611') ), axis=0)
        # realFlow = np.delete(realFlow, np.where((realFlow[:,0] == '684') & (realFlow[:,2] == '611')), axis=0)
        # realInjection = np.delete(realInjection, np.where((realInjection[:,0] == '645') | (realInjection[:,0] == '646') ), axis=0)
        # realFlow = np.delete(realFlow, np.where((realFlow[:,0] == '645') & (realFlow[:,2] == '646')), axis=0)
        # realInjection = np.delete(realInjection, np.where((realInjection[:,0] == '671') | (realInjection[:,0] == '684') ), axis=0)
        # realFlow = np.delete(realFlow, np.where((realFlow[:,0] == '671') & (realFlow[:,2] == '684')), axis=0)
        # realInjection = np.delete(realInjection, np.where((realInjection[:,0] == '684') | (realInjection[:,0] == '652') ), axis=0)
        # realFlow = np.delete(realFlow, np.where((realFlow[:,0] == '684') & (realFlow[:,2] == '652')), axis=0)
        
        # realInjection = np.delete(realInjection, np.where((realInjection[:,0] == '671') | (realInjection[:,0] == '692') ), axis=0)
        # realFlow = np.delete(realFlow, np.where((realFlow[:,0] == '671') & (realFlow[:,2] == '692')), axis=0)
        
        # realInjection = np.delete(realInjection, np.where((realInjection[:,0] == '632') | (realInjection[:,0] == '670') ), axis=0)
        # realFlow = np.delete(realFlow, np.where((realFlow[:,0] == '632') & (realFlow[:,2] == '670')), axis=0)
        
        # realInjection = np.delete(realInjection, np.where((realInjection[:,0] == '633') | (realInjection[:,0] == '634') ), axis=0)
        # realFlow = np.delete(realFlow, np.where((realFlow[:,0] == '633') & (realFlow[:,2] == '634')), axis=0)
        
        # voltageAnglePMU = np.delete(voltageAnglePMU, np.where((voltageAnglePMU[:,0] == '633') | (voltageAnglePMU[:,0] == '634') ), axis=0)
        
        # realInjection = np.delete(realInjection, np.where((realInjection[:,0] == '633') | (realInjection[:,0] == '634')), axis=0)
        # realFlow = np.delete(realFlow, np.where((realFlow[:,0] == '632') & (realFlow[:,2] == '633')), axis=0)
        # realFlow = np.delete(realFlow, np.where((realFlow[:,0] == '633') & (realFlow[:,2] == '634')), axis=0)
        
        # voltageAnglePMU = np.delete(voltageAnglePMU, np.where((voltageAnglePMU[:,0] == '633') | (voltageAnglePMU[:,0] == '634') ), axis=0)
        
        
        injectionConnection = self.brancBusMatrixReduced.T @ self.brancBusMatrixReduced
        # injectionConnection = self.brancBusMatrix.T @ self.brancBusMatrix
        
        J_API = np.zeros((len(realInjection), self.obsInstance.systemSize))
        for node in range(len(realInjection)):
            fromNode = self.obsInstance.nodeOrder.index(realInjection[node, 0].upper() + "." +realInjection[node,1])
            allFromNodes = np.where(realInjection[:,0] == realInjection[node,0])[0]
            J_API[node, :] = injectionConnection[fromNode, :]           
                    
        J_APF = np.zeros((len(realFlow), self.obsInstance.systemSize))
        for node in range(len(realFlow)):
            fromNode = realFlow[node][1].astype(int)
            numPhase = np.where((realFlow[:,0] == realFlow[node][0]) & (realFlow[:,2] == realFlow[node][2]))[0]
            for phase in range(len(numPhase)):
                dummyfromNode = realFlow[numPhase[phase]][1].astype(int)
                toNode = realFlow[numPhase[phase]][3].astype(int)
                #### Active Power Flow
                if dummyfromNode == fromNode:
                    ### Real Power Flow derivative Theta From Self
                    J_APF[node, fromNode] = 1
                else:
                    ### Real Power Flow derivative Theta From Dummy
                    # J_APF[node, dummyfromNode] = -1
                    continue
                
                ### Real Power Flow derivative Theta To Bus
                J_APF[node, toNode] = -1
        
        J_V_Angle = np.zeros((len(voltageAnglePMU), self.obsInstance.systemSize))
        for node in range(len(voltageAnglePMU)):
            fromNode = self.obsInstance.nodeOrder.index(voltageAnglePMU[node, 0].upper() + "." + voltageAnglePMU[node, 1])
            J_V_Angle[node, fromNode] = 1
                            
        J = np.concatenate((J_API, J_APF, J_V_Angle))
        return J
    
    def solve_Obs(self):
        # self.obsInstance.realInjection = np.delete(self.obsInstance.realInjection, np.where((self.obsInstance.realInjection[:,0] == '633') | (self.obsInstance.realInjection[:,0] == '634')), axis=0)
        # self.obsInstance.realFlow = np.delete(self.obsInstance.realFlow, np.where((self.obsInstance.realFlow[:,0] == '632') & (self.obsInstance.realFlow[:,2] == '633')), axis=0)
        # self.obsInstance.realFlow = np.delete(self.obsInstance.realFlow, np.where((self.obsInstance.realFlow[:,0] == '633') & (self.obsInstance.realFlow[:,2] == '634')), axis=0)
        # self.obsInstance.voltageAnglePMU = np.delete(self.obsInstance.voltageAnglePMU, np.where((self.obsInstance.voltageAnglePMU[:,0] == '633') | (self.obsInstance.voltageAnglePMU[:,0] == '634') ), axis=0)

        
        self.branchBusIncidanceMatrix() 
        while True:
            H = self.createJacobian(realInjection=self.obsInstance.realInjection, realFlow=self.obsInstance.realFlow, voltageAnglePMU=self.obsInstance.voltageAnglePMU) 
            # H = self.createJacobian(realInjection=self.obsInstance.realInjection, realFlow=self.obsInstance.realFlow, voltageAnglePMU=[])
            G = np.dot(H.T, H)
            x = np.zeros((self.obsInstance.systemSize,1))
            
            L = np.tril(G)
            n = self.obsInstance.systemSize
    
            counter = 1
            for j in range(n):
                if j > 0:
                    L[j:n, j] = L[j:n, j] - L[j:n, :j] @ L[j, :j].T

                if abs(L[j, j]) < 0.001:
                    L[j, j] = 1
                    x[j, 0] = counter
                    counter += 1
                else:
                    L[j:n, j] = L[j:n, j] / np.sqrt(L[j, j])

            Gprime = L @ L.T    
            theta = np.linalg.solve(Gprime, x) 
            
            Pb = self.brancBusMatrixReduced @ theta
            locations = np.where(np.abs(Pb) > 0.001)[0]
            
            self.brancBusMatrixReduced = np.delete(self.brancBusMatrixReduced, locations, axis=0)
            self.brancBusMatrix = np.delete(self.brancBusMatrix, locations, axis=0)
            
            fromIdx = []
            toIdx = []
            for idx in locations:
                fromBus = self.branchList[idx][0]
                toBus = self.branchList[idx][1]
                fromIdx.append(np.where((self.obsInstance.realInjection[:,0] == fromBus.split('.')[0]) & (self.obsInstance.realInjection[:,1] == fromBus.split('.')[1]))[0])
                toIdx.append(np.where((self.obsInstance.realInjection[:,0] == toBus.split('.')[0]) & (self.obsInstance.realInjection[:,1] == toBus.split('.')[1]))[0])
            
            if len(np.union1d(fromIdx,toIdx)) != 0:
                self.obsInstance.realInjection = np.delete(self.obsInstance.realInjection, np.union1d(fromIdx,toIdx), axis=0)
                self.obsInstance.reacInjection = np.delete(self.obsInstance.reacInjection, np.union1d(fromIdx,toIdx), axis=0)
                
            
            asd = list(map(self.branchList.__getitem__, locations))
            if len(locations) == 0:
                break
            
            # for idx in locations:
        
        adj = self.brancBusMatrix.T @ self.brancBusMatrix
        nn = self.obsInstance.systemSize
        counter_dfs = 0
        self.obsInstance.observableIslands = np.zeros(nn, dtype=int)
        visited = [False] * nn
        self.observableIslands = self.finder_comp(counter_dfs, self.obsInstance.observableIslands, visited, adj)

        
    def dfs_comp(self, at, visited, islands, counter_dfs, adj):
        visited[at] = True
        islands[at] = counter_dfs
        column = np.where(adj[at, :] != 0)[0]
        
        for next_node in column:
            if not visited[next_node]:
                visited, islands, counter_dfs, adj = self.dfs_comp(next_node, visited, islands, counter_dfs, adj)
        
        return visited, islands, counter_dfs, adj

    def finder_comp(self, counter_dfs, islands, visited, adj):
        for ii in range(len(visited)):
            if not visited[ii]:
                counter_dfs += 1
                visited, islands, counter_dfs, adj = self.dfs_comp(ii, visited, islands, counter_dfs, adj)
        return islands
        
    
    def branchBusIncidanceMatrix(self):
        self.branchList = []
        self.brancBusMatrix = np.zeros((self.obsInstance.branchNumber, self.obsInstance.systemSize))
        self.brancBusMatrixReduced = np.zeros((self.obsInstance.branchNumber, self.obsInstance.systemSize))
        self.bus_to_bus_connection = np.zeros((self.obsInstance.systemSize,self.obsInstance.systemSize))
        counter = 0
        self.obsInstance.dss.ActiveCircuit.Lines.First
        while True:
            if not self.obsInstance.dss.ActiveCircuit.Lines.IsSwitch:
                fromBus = self.obsInstance.dss.ActiveCircuit.Lines.Bus1.split('.')[0]
                toBus  = self.obsInstance.dss.ActiveCircuit.Lines.Bus2.split('.')[0]
                num_phases = self.obsInstance.dss.ActiveCircuit.Lines.Phases
                # num_phases = len(self.dss.ActiveCircuit.Lines.Bus1.split('.')[1:])
                phase_order = self.obsInstance.dss.ActiveCircuit.ActiveCktElement.NodeOrder
                for phase1 in range(num_phases):
                    fromIdx = self.obsInstance.nodeOrder.index((fromBus + '.' + str(phase_order[phase1])).upper())
                    self.brancBusMatrix[counter, fromIdx] = 1
                    self.brancBusMatrixReduced[counter, fromIdx] = 1
                    self.brancBusMatrixReduced[counter, self.obsInstance.nodeOrder.index((toBus + '.' + str(phase_order[phase1])).upper())] = -1
                    self.branchList.append(((fromBus + '.' + str(phase_order[phase1])).upper(), (toBus + '.' + str(phase_order[phase1])).upper()))
                    for phase2 in range(num_phases):
                        offDiagFromIdx = self.obsInstance.nodeOrder.index((fromBus + '.' + str(phase_order[phase2])).upper())
                        toIdx = self.obsInstance.nodeOrder.index((toBus + '.' + str(phase_order[phase2])).upper())
                        self.brancBusMatrix[counter, offDiagFromIdx] = 1
                        self.brancBusMatrix[counter, toIdx] = -1
                        
                        self.bus_to_bus_connection[fromIdx, offDiagFromIdx] = 1
                        self.bus_to_bus_connection[fromIdx, toIdx] = 1
                        
                    counter += 1
                    
                for phase1 in range(num_phases):
                    toIdx = self.obsInstance.nodeOrder.index((toBus + '.' + str(phase_order[phase1])).upper())
                    for phase2 in range(num_phases):
                        offDiagToIdx = self.obsInstance.nodeOrder.index((toBus + '.' + str(phase_order[phase2])).upper())
                        fromIdx = self.obsInstance.nodeOrder.index((fromBus + '.' + str(phase_order[phase2])).upper())
                        self.bus_to_bus_connection[toIdx, offDiagToIdx] = 1
                        self.bus_to_bus_connection[toIdx, fromIdx] = 1
                    
            if not self.obsInstance.dss.ActiveCircuit.Lines.Next > 0:
                break
        
        # Get power flows for transformers
        self.obsInstance.dss.ActiveCircuit.Transformers.First
        if self.obsInstance.dss.ActiveCircuit.Transformers.Count > 0:
            while True:
                fromBus = self.obsInstance.dss.ActiveCircuit.ActiveCktElement.BusNames[0].split('.')[0]
                toBus = self.obsInstance.dss.ActiveCircuit.ActiveCktElement.BusNames[1].split('.')[0]
                num_phases = self.obsInstance.dss.ActiveCircuit.ActiveCktElement.NumPhases
                phase_order = self.obsInstance.dss.ActiveCircuit.ActiveElement.NodeOrder
                for phase1 in range(num_phases):
                    fromIdx = self.obsInstance.nodeOrder.index((fromBus + '.' + str(phase_order[phase1])).upper())
                    self.brancBusMatrix[counter, fromIdx] = 1
                    self.brancBusMatrixReduced[counter, fromIdx] = 1
                    self.brancBusMatrixReduced[counter, self.obsInstance.nodeOrder.index((toBus + '.' + str(phase_order[phase1])).upper())] = -1
                    self.branchList.append(((fromBus + '.' + str(phase_order[phase1])).upper(), (toBus + '.' + str(phase_order[phase1])).upper()))
                    for phase2 in range(num_phases):
                        offDiagFromIdx = self.obsInstance.nodeOrder.index((fromBus + '.' + str(phase_order[phase2])).upper())
                        toIdx = self.obsInstance.nodeOrder.index((toBus + '.' + str(phase_order[phase2])).upper())
                        self.brancBusMatrix[counter, offDiagFromIdx] = 1
                        self.brancBusMatrix[counter, toIdx] = -1
                        
                        self.bus_to_bus_connection[fromIdx, offDiagFromIdx] = 1
                        self.bus_to_bus_connection[fromIdx, toIdx] = 1
                    
                    counter += 1
                    
                for phase1 in range(num_phases):
                    toIdx = self.obsInstance.nodeOrder.index((toBus + '.' + str(phase_order[phase1])).upper())
                    for phase2 in range(num_phases):
                        offDiagToIdx = self.obsInstance.nodeOrder.index((toBus + '.' + str(phase_order[phase2])).upper())
                        fromIdx = self.obsInstance.nodeOrder.index((fromBus + '.' + str(phase_order[phase2])).upper())
                        self.bus_to_bus_connection[toIdx, offDiagToIdx] = 1
                        self.bus_to_bus_connection[toIdx, fromIdx] = 1

                if not self.obsInstance.dss.ActiveCircuit.Transformers.Next > 0:
                    break
        
        # Get power flows for switches
        self.obsInstance.dss.ActiveCircuit.Lines.First
        while True:
            if self.obsInstance.dss.ActiveCircuit.Lines.IsSwitch:
                fromBus = self.obsInstance.dss.ActiveCircuit.Lines.Bus1.split('.')[0]
                toBus  = self.obsInstance.dss.ActiveCircuit.Lines.Bus2.split('.')[0]
                num_phases = self.obsInstance.dss.ActiveCircuit.Lines.Phases
                # num_phases = len(self.dss.ActiveCircuit.Lines.Bus1.split('.')[1:])
                phase_order = self.obsInstance.dss.ActiveCircuit.ActiveCktElement.NodeOrder
                switch = self.obsInstance.dss.ActiveCircuit.CktElements(f"Line.{self.obsInstance.dss.ActiveCircuit.Lines.Name}")
                switchStatus = int(switch.IsOpen(2,0))
                for phase1 in range(num_phases):
                    fromIdx = self.obsInstance.nodeOrder.index((fromBus + '.' + str(phase_order[phase1])).upper())
                    toIdx = self.obsInstance.nodeOrder.index((toBus + '.' + str(phase_order[phase1])).upper())
                    if not switchStatus:
                        self.branchList.append(((fromBus + '.' + str(phase_order[phase1])).upper(), (toBus + '.' + str(phase_order[phase1])).upper()))
                        self.brancBusMatrix[counter, fromIdx] = 1
                        self.brancBusMatrix[counter, toIdx] = -1
                        self.brancBusMatrixReduced[counter, fromIdx] = 1
                        self.brancBusMatrixReduced[counter, toIdx] = -1
                        counter += 1
                    
                        self.bus_to_bus_connection[fromIdx, fromIdx] = 1
                        self.bus_to_bus_connection[fromIdx, toIdx] = 1
                        self.bus_to_bus_connection[toIdx, fromIdx] = 1
                        self.bus_to_bus_connection[toIdx, toIdx] = 1
            if not self.obsInstance.dss.ActiveCircuit.Lines.Next > 0:
                break
