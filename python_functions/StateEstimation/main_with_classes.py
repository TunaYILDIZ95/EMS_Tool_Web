import numpy as np
import pandas as pd
from dss import dss
from scipy.sparse import csc_matrix
from scipy.optimize import linprog
from scipy.linalg import qr
import copy
from operator import itemgetter
from cmath import rect

#
import itertools

### Subclasses
from python_functions.StateEstimation.WLS_SE_class import StateEstimation
from python_functions.StateEstimation.Observability import Observability

### Simulation Class
from python_functions.StateEstimation.OpenDSS_sim_v2 import OpenDSS_Sim_v2

class StateEstimator:
    def __init__(self, dss, badDataNumber = 10, seedCounter = 0, fileLocation=None): #20de calismiyor #seed 12
        self.dss = dss
        # fileLocation = input("Enter the path to the master.dss file: ")
        
        # fileLocation = "Networks/IEEE13/IEEE13Nodeckt.dss"
        # fileLocation = "Networks/IEEE13/IEEE13Nodev7_with_deltawye_and_single_phase.dss"
        # fileLocation = "Networks/IEEE14/14Bus.dss"
        # fileLocation = "Networks/IEEE30/Master.dss"
        # fileLocation = "Networks/IEEE123/123Bus_fixed_taps.dss"

        
        # TrueSwitchStatus = pd.DataFrame({"SwitchName": ["671692", "650633" ],"Status": ["Closed" , "Closed"]})
        # ActualSwitchStatus = pd.DataFrame({"SwitchName": ["671692","650633"],"Status": ["Closed", "Closed"]})
        # TrueSwitchStatus = pd.DataFrame({"SwitchName": ["671692", "675670", "650633" ],"Status": ["Open", "Closed", "Closed"]})
        # ActualSwitchStatus = pd.DataFrame({"SwitchName": ["671692", "675670","650633"],"Status": ["Closed", "Closed", "Open"]})
        # TrueSwitchStatus = pd.DataFrame({"SwitchName": ["671692", "675670", "650633", "671680", "632680"],"Status": ["Closed", "Closed", "Closed", "Closed", "Closed"]})
        # ActualSwitchStatus = pd.DataFrame({"SwitchName": ["671692", "675670","650633", "671680", "632680"],"Status": ["Closed", "Closed", "Closed", "Closed", "Closed"]})
        # TrueSwitchStatus = pd.DataFrame({"SwitchName": ["671692"],"Status": ["Closed"]})
        # ActualSwitchStatus = pd.DataFrame({"SwitchName": ["671692"],"Status": ["Closed"]})
        # TrueSwitchStatus = pd.DataFrame({"SwitchName": ["Sw1","Sw2","Sw3","Sw4","Sw5","Sw6"],"Status": ["Closed","Closed","Closed","Closed","Closed","Closed"]})
        # ActualSwitchStatus = pd.DataFrame({"SwitchName": ["Sw1","Sw2","Sw3","Sw4","Sw5","Sw6"],"Status": ["Closed","Closed","Closed","Closed","Closed","Closed"]})
        # TrueSwitchStatus = pd.DataFrame({"SwitchName": ["Sw2","Sw3","Sw4","Sw5","Sw6"],"Status": ["Closed","Closed","Closed","Closed","Closed"]})
        # ActualSwitchStatus = pd.DataFrame({"SwitchName": ["Sw2","Sw3","Sw4","Sw5","Sw6"],"Status": ["Closed","Closed","Closed","Closed","Closed"]})

        # fileLocation = "Networks/IEEE13/IEEE13Nodev7_no_sw.dss"
        TrueSwitchStatus = pd.DataFrame({"SwitchName": [], "Status": []})
        ActualSwitchStatus = pd.DataFrame({"SwitchName": [], "Status": []})

        self.externalSimulationTrue = OpenDSS_Sim_v2(dss, fileLocation, IsBadData = False, IsNoise=False, switchStatus = TrueSwitchStatus, badDataNumber = badDataNumber, seedCounter = seedCounter)
        self.externalSimulationActual = OpenDSS_Sim_v2(dss, fileLocation, IsBadData = False, IsNoise=False, switchStatus = TrueSwitchStatus, badDataNumber = badDataNumber, seedCounter = seedCounter)

        self.dss('redirect ' + fileLocation)
        self.dss.ActiveCircuit.Solution.Solve()

        # self.open_close_switches(switchLinesNames)
        self.set_switch_status(switchStatus=ActualSwitchStatus)

        # Initialize System with only lines and transformers
        self.loads_vsource_and_solve(option = 1)  
        
        # Montecarlo Simulation
        self.seedCounter = seedCounter
        
        # Initialize the variables      
        self.initialize_variables()
 
    def initialize_variables(self):
        ### SE Settings        
        self.systemSize = len(self.dss.ActiveCircuit.YNodeOrder)
        self.branchNumber = 0
        self.branchListThetaInitialization = []
        
        self.base_power_mva = 1  # Example: 10 MVA
        self.base_power_kw = self.base_power_mva * 1000  # Convert MVA to kW for calculations
        self.systemFreq = 60 # Hz
        self.voltageInitial = np.ones(self.systemSize)
        
        self.nodeOrder = self.dss.ActiveCircuit.YNodeOrder
        nodeList = [i.split('.')[1] for i in self.nodeOrder]


        self.initialLineSusceptance = True

        self.voltageBases = self.get_bus_base_voltages()
        self.lineSusceptance = self.get_line_charging_susceptances()
        self.lineConductance = np.zeros((self.systemSize,self.systemSize))
        self.busTobusConnection = self.get_bus_connection()
        
        self.Ybus, self.YMatrix = self.get_YMatrix(self.voltageBases, self.base_power_mva)
        self.Ybus_original = self.Ybus.copy()
                
        self.loads_vsource_and_solve(option=2)
        self.getTransformerTap()
        # self.YbusLoad,_ = self.get_YMatrixLoad(self.voltageBases, self.base_power_mva)
        self.loads_vsource_and_solve(option=1)
        self.correctTransformerLineSusceptance(self.Ybus, self.voltageBases, self.base_power_mva)
        # self.Ybus, self.YMatrix = self.get_YMatrix(self.voltageBases, self.base_power_mva)
        
        # self.thetaInitial = self.assign_initial_angle_states()
        
        self.get_transformer_connection_types()
        ### Observability Analysis
        self.observableIslands = []
        
        #### Enable Vsource and Loads to get the measurements
        # self.loads_vsource_and_solve(option=2)

        ## Get True System Values
        self.trueVoltages = self.externalSimulationTrue.voltages
        self.trueAngles = self.externalSimulationTrue.angles
        self.trueBadDataLocation = self.externalSimulationActual.trueBadDataLocation.sort_values(by='Measurement Idx', ascending=True).reset_index(drop=True)

        ### Get Actual System Measurements
        self.voltages = self.externalSimulationActual.voltages
        self.angles = self.externalSimulationActual.angles
        self.realInjection = self.externalSimulationActual.realInjection
        self.reacInjection = self.externalSimulationActual.reacInjection
        self.realFlow = self.externalSimulationActual.realFlow
        self.reacFlow = self.externalSimulationActual.reacFlow

        ### PMU Data
        self.voltageMagPMU = self.externalSimulationActual.voltageMagPMU
        self.voltageAnglePMU = self.externalSimulationActual.voltageAnglePMU
        self.currentMagPMU = self.externalSimulationActual.currentMagPMU
        self.currentAnglePMU = self.externalSimulationActual.currentAnglePMU

        ### Get Switch Measurements
        self.switches = self.get_switches(self.base_power_kw)
        ### Delete Open Switch Flows
        self.openSwitchFlowsIdx = self.extract_open_switch_flows()
        
        if len(self.switches) > 0:
            self.initial_switchStates = np.zeros(2*len(self.switches))          
            self.lagrangeSolver = True
        else:
            self.initial_switchStates = []
            self.lagrangeSolver = False
            
        # self.openBus_idx, self.closedBus_idx = self.get_open_Buses()
    
        ### Corrected Switch Errors
        self.switchErrors = pd.DataFrame(columns = ["SwitchName", "Initial Status", "Corrected Status"])
        ### Deleted Bad Data
        # self.badData = pd.DataFrame(columns = ["FromBus", "ToBus", "Type", "Phase", "Solver"])
        self.badData_WLS = pd.DataFrame(columns=["Measurement Idx", "Type", "Phase", "Solver"])
        self.badData_LAV = pd.DataFrame(columns=["Measurement Idx", "Type", "Phase", "Solver"])

        ### Montecarlo Variables
        self.WLS_converged = True
        self.LAV_converged = True
        self.overall_mse_WLS = -99
        self.overall_mse_LAV = -99

        self.connection_types = self.get_transformer_connection_types()
        self.thetaInitial = self.assign_initial_angle_states(initial_angle=0)
    
    def assign_initial_angle_states(self, initial_angle=None):

        # Deep copy of realFlow data to preserve the original data
        branch_data = copy.deepcopy(self.realFlow)
        theta_initial = np.zeros(self.systemSize)
        theta_initial[:3] = 0

        # Number of branches
        num_branches = branch_data.shape[0]
        bus_to_branch_data = np.zeros((num_branches, self.systemSize))

        # Populate bus_to_branch_data with connections
        for i in range(num_branches):
            from_bus_idx = int(branch_data[i, 1])
            to_bus_idx = int(branch_data[i, 3])
            bus_to_branch_data[i, from_bus_idx] = 1
            bus_to_branch_data[i, to_bus_idx] = 1

        bus_to_bus = np.dot(bus_to_branch_data.T, bus_to_branch_data)
        transformer_bus_names = {conn['From Bus'].upper() for conn in self.connection_types}.union(
            {conn['To Bus'].upper() for conn in self.connection_types})
        transformer_bus_indices = [
            idx for idx, bus in enumerate(self.nodeOrder)
            if bus.split('.')[0].upper() in transformer_bus_names
        ]

        # DFS function to traverse the graph
        def dfs(graph, start, transformer_bus_indices):
            visited = set()
            stack = [start]

            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    # If a transformer bus is encountered, stop the search
                    if node in transformer_bus_indices:
                        if len(stack) == 0:
                            return visited
                        else:
                            node = stack.pop()
                            continue
                    # Add all adjacent nodes to the stack
                    for neighbor in range(len(graph[node])):
                        if graph[node][neighbor] != 0 and neighbor not in visited:
                            stack.append(neighbor)
            return visited

        # Initialize a set to keep track of all visited nodes
        all_visited_nodes = set()

        # Initialize a list to keep track of the connections and their types
        connection_info = []

        # Perform DFS from every unvisited node
        for node in range(self.systemSize):
            if node not in all_visited_nodes:
                visited_nodes = dfs(bus_to_bus, node, transformer_bus_indices)
                all_visited_nodes.update(visited_nodes)

                # Get the actual names of the buses using nodeOrder
                visited_bus_names = [self.nodeOrder[idx] for idx in visited_nodes]

                # Find the connection types for the visited bus names
                for bus_name in visited_bus_names:
                    for connection in self.connection_types:
                        if bus_name.split('.')[0].upper() == connection['From Bus'].upper():
                            for bus_name_temp in visited_bus_names:
                                if connection['Connections'][0] == 'Delta':
                                    deltawye = 1
                                else:
                                    deltawye = 0
                                connection_info.append({
                                    'Bus Name': bus_name_temp,  # Convert list to tuple
                                    'Connection Type': connection['Connections'][0],
                                    'Bus Index': self.nodeOrder.index(bus_name_temp),
                                    'DeltawyeFlag': deltawye
                                })
                        elif bus_name.split('.')[0].upper() == connection['To Bus'].upper():
                            for bus_name_temp in visited_bus_names:
                                if connection['Connections'][1] == 'Delta':
                                    deltawye = 1
                                else:
                                    deltawye = 0
                                connection_info.append({
                                    'Bus Name': bus_name_temp,  # Convert list to tuple
                                    'Connection Type': connection['Connections'][1],
                                    'Bus Index': self.nodeOrder.index(bus_name_temp),
                                    'DeltawyeFlag': deltawye
                                })
            if len(all_visited_nodes) == self.systemSize:
                break
        # Create a DataFrame for bus connection data and remove duplicates
        bus_connection_data = pd.DataFrame(connection_info).drop_duplicates()

        # intial_angle = self.dss.Circuits.
        def assign_angle(graph, start, bus_connection_data, theta_initial, phase_angle):
            visited = set()
            stack = [start]
            theta_initial[start] = phase_angle
            while stack:
                node = stack.pop()
                if node not in visited:
                    frombusind = int(np.where(node == bus_connection_data['Bus Index'].values)[0][0])
                    visited.add(node)
                    # Add all adjacent nodes to the stack
                    for neighbor in range(len(graph[node])):
                        if graph[node][neighbor] != 0 and neighbor not in visited:
                            tobusind = int(np.where(neighbor == bus_connection_data['Bus Index'].values)[0][0])
                            if bus_connection_data['DeltawyeFlag'].iloc[frombusind] == 1 and bus_connection_data['DeltawyeFlag'].iloc[tobusind] == 1:
                                theta_initial[neighbor] = theta_initial[node]
                            elif bus_connection_data['DeltawyeFlag'].iloc[frombusind] == 0 and bus_connection_data['DeltawyeFlag'].iloc[tobusind] == 0:
                                theta_initial[neighbor] = theta_initial[node]
                            elif bus_connection_data['DeltawyeFlag'].iloc[frombusind] == 1 and bus_connection_data['DeltawyeFlag'].iloc[tobusind] == 0:
                                theta_initial[neighbor] = theta_initial[node] - 30
                            elif bus_connection_data['DeltawyeFlag'].iloc[frombusind] == 0 and bus_connection_data['DeltawyeFlag'].iloc[tobusind] == 1:
                                theta_initial[neighbor] = theta_initial[node] - 30
                            stack.append(neighbor)
            return theta_initial

        bus1 = self.nodeOrder[0].split('.')[0]
        bus1a = self.nodeOrder.index(f'{bus1}.1')
        bus1b = self.nodeOrder.index(f'{bus1}.2')
        bus1c = self.nodeOrder.index(f'{bus1}.3')

        theta_initial = assign_angle(bus_to_bus, bus1a, bus_connection_data, theta_initial, 0 + initial_angle)
        theta_initial = assign_angle(bus_to_bus, bus1b, bus_connection_data, theta_initial, -120 + initial_angle)
        theta_initial = assign_angle(bus_to_bus, bus1c, bus_connection_data, theta_initial, 120 + initial_angle)

        return theta_initial
    def loads_vsource_and_solve(self, option = None):
        if option == 1:
            # Set the Vsource as active and disable it
            vsource_name = "source"  # Replace with the name of your Vsource
            self.dss(f"Disable Vsource.{vsource_name}")
            
            self.originalLoads_P = []
            self.originalLoads_Q = []
            self.dss.ActiveCircuit.Loads.First
            while True:
                self.originalLoads_P.append(self.dss.ActiveCircuit.Loads.kW)
                self.originalLoads_Q.append(self.dss.ActiveCircuit.Loads.kvar)
                self.dss.ActiveCircuit.Loads.kW = 0
                self.dss.ActiveCircuit.Loads.kvar = 0
                if not self.dss.ActiveCircuit.Loads.Next > 0:
                    break
            self.originalLoads_P = np.array(self.originalLoads_P)
            self.originalLoads_Q = np.array(self.originalLoads_Q)
        elif option == 2:
            # Set the Vsource as active and disable it
            vsource_name = "source"  # Replace with the name of your Vsource
            self.dss(f"Enable Vsource.{vsource_name}")
            
            self.dss.ActiveCircuit.Loads.First
            counter = 0
            while True:
                self.dss.ActiveCircuit.Loads.kW = self.originalLoads_P[counter]
                self.dss.ActiveCircuit.Loads.kvar = self.originalLoads_Q[counter]
                counter += 1
                if not self.dss.ActiveCircuit.Loads.Next > 0:
                    break
        
        self.dss.ActiveCircuit.Solution.Solve()

    def solve(self):
        # self.voltages = np.delete(self.voltages, range(0,18), axis=0)
        
        self.Obs = Observability(self)
        self.Obs.solve_Obs()

        self.setIslandVariables(1)
        self.SE = StateEstimation(self)
        self.lagrangeSolver = False
        if self.lagrangeSolver:
            self.SE.solve_WLS_topology(Ybus=self.Ybus[np.ix_(self.closedStateIdx, self.closedStateIdx)])
            print(self.switchErrors)
        if not self.lagrangeSolver:
            # self.Ybus, self.YMatrix = self.get_YMatrix(self.voltageBases, self.base_power_mva)
            self.Ybus = self.Ybus_original.copy()

            self.SE = StateEstimation(self)
            self.SE.solve_WLS_badData(Ybus=self.Ybus[np.ix_(self.closedStateIdx, self.closedStateIdx)])

            # try:
            #     self.SE = StateEstimation(self)
            #     self.SE.solve_LAV_badData(Ybus=self.Ybus[np.ix_(self.closedStateIdx, self.closedStateIdx)])
            #     finalBadData = self.badData_LAV['Measurement Idx']
            # except:
            finalBadData = self.badData_WLS['Measurement Idx']

            # print(self.badData_WLS)
            # print(self.badData_LAV)
            # print(self.trueBadDataLocation)
        return finalBadData

    def setIslandVariables(self, island):
        self.closedStateIdx = np.where(self.observableIslands == island)[0]
        
        self.voltageInitialIsland = self.voltageInitial[self.closedStateIdx]
        self.thetaInitialIsland = list(map(self.thetaInitial.__getitem__,self.closedStateIdx))
        self.nodeOrderIsland = list(map(self.nodeOrder.__getitem__,self.closedStateIdx))
        
        ### Measurement Data Island
        self.realFlowIsland = self.realFlow.copy()
        self.reacFlowIsland = self.reacFlow.copy()
        self.realInjectionIsland = self.realInjection.copy()
        self.reacInjectionIsland = self.reacInjection.copy()
        self.voltagesIsland = self.voltages.copy()
        
        ### PMU Measurements Island
        self.voltageMagPMUIsland = self.voltageMagPMU.copy()
        self.voltageAnglePMUIsland = self.voltageAnglePMU.copy()
        
        ### Switch Data Island
        self.switchesIsland = self.switches.copy()
        
        removeFlowIdx = []
        for i in range(len(self.realFlowIsland)):
            if (self.realFlowIsland[i,0] + '.' + str(self.realFlowIsland[i,4])).upper() in self.nodeOrderIsland and (self.realFlowIsland[i,2] + '.' + str(self.realFlowIsland[i,4])).upper() in self.nodeOrderIsland:
                self.realFlowIsland[i,1] = self.nodeOrderIsland.index((self.realFlowIsland[i,0] + '.' + str(self.realFlowIsland[i,4])).upper())
                self.realFlowIsland[i,3] = self.nodeOrderIsland.index((self.realFlowIsland[i,2] + '.' + str(self.realFlowIsland[i,4])).upper())
                self.reacFlowIsland[i,1] = self.nodeOrderIsland.index((self.reacFlowIsland[i,0] + '.' + str(self.reacFlowIsland[i,4])).upper())
                self.reacFlowIsland[i,3] = self.nodeOrderIsland.index((self.reacFlowIsland[i,2] + '.' + str(self.reacFlowIsland[i,4])).upper())
            else:
                removeFlowIdx.append(i)
     
        self.realFlowIsland = np.delete(self.realFlowIsland, removeFlowIdx, axis=0) 
        self.reacFlowIsland = np.delete(self.reacFlowIsland, removeFlowIdx, axis=0)
        
        removeInjIdx = []
        for i in range(len(self.realInjectionIsland)):
            if (self.realInjectionIsland[i,0] + '.' + str(self.realInjectionIsland[i,1])).upper() not in self.nodeOrderIsland:
                removeInjIdx.append(i)
        
        self.realInjectionIsland = np.delete(self.realInjectionIsland, removeInjIdx, axis=0)
        self.reacInjectionIsland = np.delete(self.reacInjectionIsland, removeInjIdx, axis=0)
        
        removeVoltIdx = []
        for i in range(len(self.voltagesIsland)):
            if (self.voltagesIsland[i,0] + '.' + str(self.voltagesIsland[i,1])).upper() not in self.nodeOrderIsland:
                removeVoltIdx.append(i)
        
        self.voltagesIsland = np.delete(self.voltagesIsland, removeVoltIdx, axis=0)
        
        removeVoltPMUMagIdx = []
        for i in range(len(self.voltageMagPMUIsland)):
            if (self.voltageMagPMUIsland[i,0] + '.' + str(self.voltageMagPMUIsland[i,1])).upper() not in self.nodeOrderIsland:
                removeVoltPMUMagIdx.append(i)
                
        self.voltageMagPMUIsland = np.delete(self.voltageMagPMUIsland, removeVoltPMUMagIdx, axis=0)
        
        removeVoltPMUAngleIdx = []
        for i in range(len(self.voltageMagPMUIsland)):
            if (self.voltageAnglePMUIsland[i,0] + '.' + str(self.voltageAnglePMUIsland[i,1])).upper() not in self.nodeOrderIsland:
                removeVoltPMUAngleIdx.append(i)
                
        self.voltageAnglePMUIsland = np.delete(self.voltageAnglePMUIsland, removeVoltPMUAngleIdx, axis=0)
           
        removeSwitchIdx = [] 
        for i in range(self.switchesIsland.shape[0]):
            if (self.switchesIsland[i,0] + '.' + str(self.switchesIsland[i,4])).upper() in self.nodeOrderIsland and (self.switchesIsland[i,2] + '.' + str(self.switchesIsland[i,4])).upper() in self.nodeOrderIsland:
                self.switchesIsland[i,1] = self.nodeOrderIsland.index((self.switchesIsland[i,0] + '.' + str(self.switchesIsland[i,4])).upper())
                self.switchesIsland[i,3] = self.nodeOrderIsland.index((self.switchesIsland[i,2] + '.' + str(self.switchesIsland[i,4])).upper())
            else:
                removeSwitchIdx.append(i)
        
        self.switchesIsland = np.delete(self.switchesIsland, removeSwitchIdx, axis=0)
        self.initial_switchStates = np.zeros(2*len(self.switchesIsland)) 
        
        self.systemSizeIsland = len(self.closedStateIdx)
        self.busTobusConnectionIsland = self.busTobusConnection[np.ix_(self.closedStateIdx, self.closedStateIdx)]
        self.lineSusceptanceIsland = self.lineSusceptance[np.ix_(self.closedStateIdx, self.closedStateIdx)]
        self.lineConductanceIsland = self.lineConductance[np.ix_(self.closedStateIdx, self.closedStateIdx)]
                
    def get_YMatrix(self, V_base_vector = None, S_base = None):
        YMatrix = csc_matrix(self.dss.YMatrix.GetCompressedYMatrix()).toarray()
        V_base = V_base_vector[:,2].astype(float) * 1000
        Z_base = V_base**2/(S_base * 1000000)        
        
        self.lineG = np.zeros((self.systemSize,self.systemSize))
  
        # Normalize Ybus to per-unit and consider transformer connections
        Ybus_pu = YMatrix.copy()
        for i in range(len(Ybus_pu)):
            for j in range(len(Ybus_pu[i])):
                if V_base[i] != V_base[j] and Ybus_pu[i, j] != 0:
                    Ybus_pu[i, j] *= (V_base[i] / V_base[j])
                    if self.initialLineSusceptance:
                        self.lineSusceptance[i, j] *= (V_base[i] / V_base[j])
                Ybus_pu[i, j] *= Z_base[j]
                if self.initialLineSusceptance:
                    self.lineSusceptance[i, j] *= Z_base[j]
        
        return Ybus_pu, YMatrix            
                    
    def correctTransformerLineSusceptance(self, Ybus_pu, V_base_vector, S_base):
        V_base = V_base_vector[:,2].astype(float) * 1000
        Z_base = V_base**2/(S_base * 1000000)
        if self.initialLineSusceptance:
            self.dss.ActiveCircuit.Transformers.First
            if self.dss.ActiveCircuit.Transformers.Count > 0:
                while True:
                    fromBusList = []
                    toBusList = []
                    fromBus = self.dss.ActiveCircuit.ActiveCktElement.BusNames[0].split('.')[0]
                    toBus = self.dss.ActiveCircuit.ActiveCktElement.BusNames[1].split('.')[0]
                    num_phases = self.dss.ActiveCircuit.ActiveCktElement.NumPhases
                    phase_order = self.dss.ActiveCircuit.ActiveElement.NodeOrder
                    num_windings = self.dss.ActiveCircuit.Transformers.NumWindings
                    
                    self.dss.ActiveCircuit.Transformers.Wdg = 1
                    tap_1 = self.dss.ActiveCircuit.Transformers.Tap
                    delta = self.dss.ActiveCircuit.Transformers.IsDelta

                    self.dss.ActiveCircuit.Transformers.Wdg = 2
                    tap_2 = self.dss.ActiveCircuit.Transformers.Tap
                    if not delta:
                        delta = self.dss.ActiveCircuit.Transformers.IsDelta
                    
                    if tap_1 != 1:
                        tap = tap_1
                    else:
                        tap = tap_2

                    if num_windings == 2:
                        noLoadTap = tap
                        vreg_flag = False
                        for phase in range(num_phases):
                            if tap != float(self.transformerTap[np.where((self.transformerTap[:,0] == fromBus) & (self.transformerTap[:,4] == str(phase_order[phase])))[0],5][0]):
                                tap = float(self.transformerTap[np.where((self.transformerTap[:,0] == fromBus) & (self.transformerTap[:,4] == str(phase_order[phase])))[0],5][0])
                                vreg_flag = True
                            
                            fromIdx = self.nodeOrder.index((fromBus + '.' + str(phase_order[phase])).upper())
                            toIdx = self.nodeOrder.index((toBus + '.' + str(phase_order[phase])).upper())
                            
                            fromBusList.append(fromIdx)
                            toBusList.append(toIdx)
                            
                        num_windings = self.dss.ActiveCircuit.Transformers.NumWindings
                        winding_connections = ""
                        for i in range(1, num_windings + 1):
                            self.dss.ActiveCircuit.Transformers.Wdg = i
                            conn = self.dss.ActiveCircuit.Transformers.IsDelta
                            winding_connections += ('Delta' if conn else 'Wye')
                            
                        Yprim = self.dss.ActiveCircuit.ActiveCktElement.Yprim   
                        # Convert the Yprim to a complex numpy array
                        Yprim_real = np.array(Yprim[::2])  
                        Yprim_imag = np.array(Yprim[1::2])    
                        Yprim_complex = Yprim_real + 1j * Yprim_imag
                        # Reshape the Yprim matrix
                        num_conductors = self.dss.ActiveCircuit.ActiveCktElement.NumConductors 
                        num_terminals = self.dss.ActiveCircuit.ActiveCktElement.NumTerminals 
                        Yprim_matrix = Yprim_complex.reshape((num_conductors * num_terminals, num_conductors * num_terminals))
                        GrounPhases = np.where(phase_order == 0)
                        nonGroundPhases = np.delete(phase_order, GrounPhases)
                        Yprim_matrix = np.delete(Yprim_matrix, GrounPhases, axis=0)
                        Yprim_matrix = np.delete(Yprim_matrix, GrounPhases, axis=1)
                        array,index = np.unique([fromBusList, toBusList], return_index=True)
                        V_base_prim = V_base[array[index]]
                        
                        # V_base_prim = V_base[np.union1d(fromBusList, toBusList)]
                        Z_base_prim = V_base_prim**2/(S_base * 1000000)
                        for i in range(len(Yprim_matrix)):
                            for j in range(len(Yprim_matrix[i])):
                                if V_base_prim[i] != V_base_prim[j] and Yprim_matrix[i, j] != 0:
                                    Yprim_matrix[i, j] *= (V_base_prim[i] / V_base_prim[j])
                                Yprim_matrix[i, j] *= Z_base_prim[j]

                        if tap == 1:
                            YprimfromBusList = range(0,int(len(nonGroundPhases)/2))
                            YprimtoBusList = range(int(len(nonGroundPhases)/2),len(nonGroundPhases))
                            trafoSusceptanceFromSide = Yprim_matrix[np.ix_(YprimfromBusList, YprimfromBusList)] - (-Yprim_matrix[np.ix_(YprimfromBusList, YprimtoBusList)])
                            self.lineSusceptance[np.ix_(fromBusList, toBusList)] = trafoSusceptanceFromSide.imag
                            self.lineConductance[np.ix_(fromBusList, toBusList)] = trafoSusceptanceFromSide.real
                            
                            trafoSusceptanceToSide = Yprim_matrix[np.ix_(YprimtoBusList, YprimtoBusList)] - (-Yprim_matrix[np.ix_(YprimtoBusList, YprimfromBusList)])
                            self.lineSusceptance[np.ix_(toBusList, fromBusList)] = trafoSusceptanceToSide.imag
                            self.lineConductance[np.ix_(toBusList, fromBusList)] = trafoSusceptanceToSide.real
                                                
                        elif tap_1 != 1:
                            # trafoSusceptanceFromSide = ((-Ybus_pu[np.ix_(fromBusList, toBusList)]) * tap)*(1-tap)/tap**2
                            # self.lineSusceptance[np.ix_(fromBusList, toBusList)] = trafoSusceptanceFromSide.imag
                            # self.lineConductance[np.ix_(fromBusList, toBusList)] = trafoSusceptanceFromSide.real

                            # trafoSusceptanceToSide = ((-Ybus_pu[np.ix_(fromBusList, toBusList)]) * tap)*(tap-1)/tap
                            # self.lineSusceptance[np.ix_(toBusList, fromBusList)] = trafoSusceptanceToSide.imag
                            # self.lineConductance[np.ix_(toBusList, fromBusList)] = trafoSusceptanceToSide.real
                            YprimfromBusList = range(0,int(len(nonGroundPhases)/2))
                            YprimtoBusList = range(int(len(nonGroundPhases)/2),len(nonGroundPhases))
                            trafoSusceptanceFromSide = Yprim_matrix[np.ix_(YprimfromBusList, YprimfromBusList)] - (-Yprim_matrix[np.ix_(YprimfromBusList, YprimtoBusList)])
                            self.lineSusceptance[np.ix_(fromBusList, toBusList)] = trafoSusceptanceFromSide.imag
                            self.lineConductance[np.ix_(fromBusList, toBusList)] = trafoSusceptanceFromSide.real
                            
                            trafoSusceptanceToSide = Yprim_matrix[np.ix_(YprimtoBusList, YprimtoBusList)] - (-Yprim_matrix[np.ix_(YprimtoBusList, YprimfromBusList)])
                            self.lineSusceptance[np.ix_(toBusList, fromBusList)] = trafoSusceptanceToSide.imag
                            self.lineConductance[np.ix_(toBusList, fromBusList)] = trafoSusceptanceToSide.real
                            
                        elif tap_2 != 1:
                            if vreg_flag:
                                Ybus_pu[np.ix_(fromBusList, toBusList)] = Ybus_pu[np.ix_(fromBusList, toBusList)] * noLoadTap/tap
                                Ybus_pu[np.ix_(toBusList, fromBusList)] = Ybus_pu[np.ix_(toBusList, fromBusList)] * noLoadTap/tap
                                Ybus_pu[np.ix_(toBusList, toBusList)] = Ybus_pu[np.ix_(toBusList, toBusList)] * (noLoadTap/tap) ** 2
                            
                            trafoSusceptanceFromSide = ((-Ybus_pu[np.ix_(fromBusList, toBusList)]) * tap)*(tap-1)/tap
                            self.lineSusceptance[np.ix_(fromBusList, toBusList)] = trafoSusceptanceFromSide.imag * noLoadTap/tap
                            self.lineConductance[np.ix_(fromBusList, toBusList)] = trafoSusceptanceFromSide.real * noLoadTap/tap
                            
                            trafoSusceptanceToSide = ((-Ybus_pu[np.ix_(toBusList, fromBusList)]) * tap)*(1-tap)/tap**2
                            self.lineSusceptance[np.ix_(toBusList, fromBusList)] = trafoSusceptanceToSide.imag * noLoadTap/tap
                            self.lineConductance[np.ix_(toBusList, fromBusList)] = trafoSusceptanceToSide.real * noLoadTap/tap
                        
                    elif num_windings == 3:
                        noLoadTap = tap
                        phase_length = int(len(phase_order)/num_windings)
                        phase_order_total = [phase_order[np.union1d(range(0,phase_length),range(phase_length,2*phase_length))],phase_order[np.union1d(range(0,phase_length),range(phase_length*2,3*phase_length))]]
                        for wdg, phase_order in enumerate(phase_order_total):
                            fromBusList = []
                            toBusList = []
                            noLoadTap = tap
                            phase_order_nonzeros = np.delete(phase_order, np.where(phase_order == 0))
                            for phase in range(num_phases):
                                fromIdx = self.nodeOrder.index((fromBus + '.' + str(phase_order_nonzeros[phase])).upper())
                                toIdx = self.nodeOrder.index((toBus + '.' + str(phase_order_nonzeros[phase+phase_length-1])).upper())
                                fromBusList.append(fromIdx)
                                toBusList.append(toIdx)
                                
                            Yprim = self.dss.ActiveCircuit.ActiveCktElement.Yprim   
                            # Convert the Yprim to a complex numpy array
                            Yprim_real = np.array(Yprim[::2])  
                            Yprim_imag = np.array(Yprim[1::2])    
                            Yprim_complex = Yprim_real + 1j * Yprim_imag
                            # Reshape the Yprim matrix
                            num_conductors = self.dss.ActiveCircuit.ActiveCktElement.NumConductors 
                            num_terminals = self.dss.ActiveCircuit.ActiveCktElement.NumTerminals 
                            Yprim_matrix = Yprim_complex.reshape((num_conductors * num_terminals, num_conductors * num_terminals))
                            if wdg == 0:
                                idx = [0,1,2,3]
                            else:
                                idx = [0,1,4,5]
                                
                            Yprim_matrix = Yprim_matrix[np.ix_(idx, idx)]
                            GrounPhases = np.where(phase_order == 0)
                            nonGroundPhases = np.delete(phase_order, GrounPhases)
                            Yprim_matrix = np.delete(Yprim_matrix, GrounPhases, axis=0)
                            Yprim_matrix = np.delete(Yprim_matrix, GrounPhases, axis=1)
                            
                            array,index = np.unique([fromBusList, toBusList], return_index=True)
                            V_base_prim = V_base[array[index]]
                            # V_base_prim = V_base[np.union1d(fromBusList, toBusList)]
                            Z_base_prim = V_base_prim**2/(S_base * 1000000)
                            for i in range(len(Yprim_matrix)):
                                for j in range(len(Yprim_matrix[i])):
                                    if V_base_prim[i] != V_base_prim[j] and Yprim_matrix[i, j] != 0:
                                        Yprim_matrix[i, j] *= (V_base_prim[i] / V_base_prim[j])
                                    Yprim_matrix[i, j] *= Z_base_prim[j]

                            if tap == 1:
                                YprimfromBusList = range(0,int(len(nonGroundPhases)/2))
                                YprimtoBusList = range(int(len(nonGroundPhases)/2),len(nonGroundPhases))
                                trafoSusceptanceFromSide = Yprim_matrix[np.ix_(YprimfromBusList, YprimfromBusList)] - (-Yprim_matrix[np.ix_(YprimfromBusList, YprimtoBusList)])
                                self.lineSusceptance[np.ix_(fromBusList, toBusList)] = trafoSusceptanceFromSide.imag
                                self.lineConductance[np.ix_(fromBusList, toBusList)] = trafoSusceptanceFromSide.real
                                
                                trafoSusceptanceToSide = Yprim_matrix[np.ix_(YprimtoBusList, YprimtoBusList)] - (-Yprim_matrix[np.ix_(YprimtoBusList, YprimfromBusList)])
                                self.lineSusceptance[np.ix_(toBusList, fromBusList)] = trafoSusceptanceToSide.imag
                                self.lineConductance[np.ix_(toBusList, fromBusList)] = trafoSusceptanceToSide.real
                                                    
                            elif tap_1 != 1:
                                trafoSusceptanceFromSide = ((-Ybus_pu[np.ix_(fromBusList, toBusList)]) * tap)*(1-tap)/tap**2
                                self.lineSusceptance[np.ix_(fromBusList, toBusList)] = trafoSusceptanceFromSide.imag
                                self.lineConductance[np.ix_(fromBusList, toBusList)] = trafoSusceptanceFromSide.real

                                trafoSusceptanceToSide = ((-Ybus_pu[np.ix_(fromBusList, toBusList)]) * tap)*(tap-1)/tap
                                self.lineSusceptance[np.ix_(toBusList, fromBusList)] = trafoSusceptanceToSide.imag
                                self.lineConductance[np.ix_(toBusList, fromBusList)] = trafoSusceptanceToSide.real
                                
                            elif tap_2 != 1:
                                trafoSusceptanceFromSide = ((-Ybus_pu[np.ix_(fromBusList, toBusList)]) * tap)*(tap-1)/tap
                                self.lineSusceptance[np.ix_(fromBusList, toBusList)] = trafoSusceptanceFromSide.imag * noLoadTap/tap
                                self.lineConductance[np.ix_(fromBusList, toBusList)] = trafoSusceptanceFromSide.real * noLoadTap/tap
                                
                                trafoSusceptanceToSide = ((-Ybus_pu[np.ix_(toBusList, fromBusList)]) * tap)*(1-tap)/tap**2
                                self.lineSusceptance[np.ix_(toBusList, fromBusList)] = trafoSusceptanceToSide.imag * noLoadTap/tap
                                self.lineConductance[np.ix_(toBusList, fromBusList)] = trafoSusceptanceToSide.real * noLoadTap/tap
                    
                    if not self.dss.ActiveCircuit.Transformers.Next > 0:
                        break
        
        self.initialLineSusceptance = False
        # return Ybus_pu, YMatrix  

    def get_line_charging_susceptances(self):
        # List of LineCodes
        lineCodes = self.dss.ActiveCircuit.LineCodes.AllNames
        # List to store susceptances for each line
        self.dss.ActiveCircuit.Lines.First
        # Loop over all lines
        susceptance_matrix = np.zeros((self.systemSize,self.systemSize))
        while True:         
            # Get the line name
            fromBus = self.dss.ActiveCircuit.Lines.Bus1.split('.')[0]
            toBus = self.dss.ActiveCircuit.Lines.Bus2.split('.')[0]            
            # Get length of the line
            length = self.dss.ActiveCircuit.Lines.Length # Convert feet to m    
                
            num_phases = self.dss.ActiveCircuit.Lines.Phases
            # num_phases = len(self.dss.ActiveCircuit.Lines.Bus1.split('.')[1:])
            phase_order = self.dss.ActiveCircuit.ActiveCktElement.NodeOrder  
            # Append to the list
            for phase in range(num_phases):
                fromIdx = self.nodeOrder.index((fromBus + '.' + str(phase_order[phase])).upper())
                for phase2 in range(num_phases):
                    toIdx = self.nodeOrder.index((toBus + '.' + str(phase_order[phase2])).upper())
                    susceptance_matrix[fromIdx, toIdx] = self.dss.ActiveCircuit.Lines.Cmatrix[phase*num_phases + phase2] * 1e-9 * np.pi * self.systemFreq * length #/ lineCode_length
            
            if not self.dss.ActiveCircuit.Lines.Next > 0:
                break
        
        return susceptance_matrix
        
    def get_bus_base_voltages(self):
        voltageBases = []
        for bus in self.dss.ActiveCircuit.ActiveBus:
            phases = bus.Nodes
            for phase in range(len(phases)):
                voltageBases.append([bus.Name,phases[phase],bus.kVBase])
        mapIndex = [self.nodeOrder.index(voltageBases[i][0].upper() + '.' + str(voltageBases[i][1])) for i in range(len(voltageBases))]
        reordered_bases = [0]*len(voltageBases)
        for idx, position in enumerate(mapIndex):
            reordered_bases[position] = voltageBases[idx]

        return np.array(reordered_bases)
    
    def get_transformer_connection_types(self):
        self.dss.ActiveCircuit.Transformers.First
        connection_types = []
        while True:

            fromBus = self.dss.ActiveCircuit.ActiveCktElement.BusNames[0].split('.')[0]
            toBus = self.dss.ActiveCircuit.ActiveCktElement.BusNames[1].split('.')[0]
            transformer_name = self.dss.ActiveCircuit.Transformers.Name
            self.dss.ActiveCircuit.SetActiveElement(f"Transformer.{transformer_name}")
            num_windings = self.dss.ActiveCircuit.Transformers.NumWindings
            winding_connections = []

            for i in range(1, num_windings + 1):
                self.dss.ActiveCircuit.Transformers.Wdg = i
                conn = self.dss.ActiveCircuit.Transformers.IsDelta
                winding_connections.append('Delta' if conn else 'Wye')

            connection_types.append({
                'From Bus': fromBus,
                'To Bus': toBus,
                'Transformer': transformer_name,
                'Connections': winding_connections
            })

            if not self.dss.ActiveCircuit.Transformers.Next > 0:
                break

        return connection_types
    def get_switches(self, base_power_kw):
        switchData = []
        self.dss.ActiveCircuit.Lines.First
        while True:
            if self.dss.ActiveCircuit.Lines.IsSwitch:
                fromBus = self.dss.ActiveCircuit.Lines.Bus1.split('.')[0]
                toBus  = self.dss.ActiveCircuit.Lines.Bus2.split('.')[0]
                num_phases = self.dss.ActiveCircuit.Lines.Phases
                phase_order = self.dss.ActiveCircuit.ActiveCktElement.NodeOrder
                powers = self.dss.ActiveCircuit.CktElements.Powers  # Get real and reactive power in kW and kVAR
                switch = self.dss.ActiveCircuit.CktElements(f"Line.{self.dss.ActiveCircuit.Lines.Name}")
                # self.dss.ActiveCircuit.SetActiveElement(f"Line.{self.dss.ActiveCircuit.Lines.Name}")
                switchStatus = int(switch.IsOpen(2,0))
                for phase in range(num_phases):
                    fromIdx = self.nodeOrder.index((fromBus + '.' + str(phase_order[phase])).upper())
                    toIdx = self.nodeOrder.index((toBus + '.' + str(phase_order[phase])).upper())
                    self.Ybus[fromIdx, fromIdx] = self.Ybus[fromIdx, fromIdx] + self.Ybus[fromIdx, toIdx]
                    self.Ybus[toIdx, toIdx] = self.Ybus[toIdx, toIdx] + self.Ybus[toIdx, fromIdx]
                    self.Ybus[fromIdx, toIdx] = 0
                    self.Ybus[toIdx, fromIdx] = 0
                    idx = phase * 2
                    # p_kw = powers[idx]
                    # q_kvar = powers[idx + 1]
                    # p_pu = p_kw / base_power_kw
                    # q_pu = q_kvar / base_power_kw
                    switchData.append([fromBus, fromIdx, toBus, toIdx, phase_order[phase], switchStatus, self.dss.ActiveCircuit.Lines.Name])
                    # switchData.append([fromBus, fromIdx, toBus, toIdx, phase_order[phase], switchStatus, p_pu, q_pu])
            if not self.dss.ActiveCircuit.Lines.Next > 0:
                break
        return np.array(switchData)
    
    def open_close_switches(self,switchLinesNames):
        for name in switchLinesNames:
            switch = self.dss.ActiveCircuit.CktElements(f"Line.{name}")
            switch.Open(2,0)
        # self.dss.ActiveCircuit.Lines.First
        # while True:
        #     if self.dss.ActiveCircuit.Lines.IsSwitch:
        #         switch = self.dss.ActiveCircuit.CktElements(f"Line.{self.dss.ActiveCircuit.Lines.Name}")
        #         switch.Open(2,0)
        #     if not self.dss.ActiveCircuit.Lines.Next > 0:
        #         break
    
    def get_open_Buses(self):
        openBus_idx = np.where(self.voltages[:,2].astype(float) == 0)[0]
        closedBus_idx = np.where(self.voltages[:,2].astype(float) != 0)[0]
        # return np.concatenate((openBus_idx,openBus_idx+self.systemSize))
        return np.array([openBus_idx,openBus_idx+self.systemSize]), closedBus_idx

    def set_switch_status(self,switchStatus = None):
        for index, row in switchStatus.iterrows():
            switch_name = row['SwitchName']
            status = row['Status'].lower()
            switch = self.dss.ActiveCircuit.CktElements(f"Line.{switch_name}")
            if status == "open":
                switch.Open(2, 0)
            elif status == "closed":
                switch.Close(2, 0)
                
    def get_bus_connection(self):
        bus_to_bus_connection = np.zeros((self.systemSize,self.systemSize))
        self.dss.ActiveCircuit.Lines.First
        while True:
            if not self.dss.ActiveCircuit.Lines.IsSwitch:
                fromBus = self.dss.ActiveCircuit.Lines.Bus1.split('.')[0]
                toBus  = self.dss.ActiveCircuit.Lines.Bus2.split('.')[0]
                num_phases = self.dss.ActiveCircuit.Lines.Phases
                # num_phases = len(self.dss.ActiveCircuit.Lines.Bus1.split('.')[1:])
                phase_order = self.dss.ActiveCircuit.ActiveCktElement.NodeOrder
                
                for phase1 in range(num_phases):
                    self.branchNumber += 1
                    fromIdx = self.nodeOrder.index((fromBus + '.' + str(phase_order[phase1])).upper())
                    self.branchListThetaInitialization.append([(fromBus).upper(),fromIdx,(toBus).upper(), self.nodeOrder.index((toBus + '.' + str(phase_order[phase1])).upper()), str(phase_order[phase1]), 'Line'])
                    for phase2 in range(num_phases):
                        offDiagFromIdx = self.nodeOrder.index((fromBus + '.' + str(phase_order[phase2])).upper())
                        toIdx = self.nodeOrder.index((toBus + '.' + str(phase_order[phase2])).upper())
                        bus_to_bus_connection[fromIdx, offDiagFromIdx] = 1
                        bus_to_bus_connection[fromIdx, toIdx] = 1
                        
                for phase1 in range(num_phases):
                    toIdx = self.nodeOrder.index((toBus + '.' + str(phase_order[phase1])).upper())
                    for phase2 in range(num_phases):
                        offDiagToIdx = self.nodeOrder.index((toBus + '.' + str(phase_order[phase2])).upper())
                        fromIdx = self.nodeOrder.index((fromBus + '.' + str(phase_order[phase2])).upper())
                        bus_to_bus_connection[toIdx, offDiagToIdx] = 1
                        bus_to_bus_connection[toIdx, fromIdx] = 1
                    
            if not self.dss.ActiveCircuit.Lines.Next > 0:
                break
        
        # Get power flows for transformers
        self.dss.ActiveCircuit.Transformers.First
        if self.dss.ActiveCircuit.Transformers.Count > 0:
            while True:
                fromBus = self.dss.ActiveCircuit.ActiveCktElement.BusNames[0].split('.')[0]
                toBus = self.dss.ActiveCircuit.ActiveCktElement.BusNames[1].split('.')[0]
                num_phases = self.dss.ActiveCircuit.ActiveCktElement.NumPhases
                phase_order = self.dss.ActiveCircuit.ActiveElement.NodeOrder
                
                num_windings = self.dss.ActiveCircuit.Transformers.NumWindings
                winding_connections = ""
                for i in range(1, num_windings + 1):
                    self.dss.ActiveCircuit.Transformers.Wdg = i
                    conn = self.dss.ActiveCircuit.Transformers.IsDelta
                    winding_connections =winding_connections +  ('Delta' if conn else 'Wye')
                
                for phase1 in range(num_phases):
                    self.branchNumber += 1
                    fromIdx = self.nodeOrder.index((fromBus + '.' + str(phase_order[phase1])).upper())
                    self.branchListThetaInitialization.append([(fromBus).upper(),fromIdx,(toBus).upper(), self.nodeOrder.index((toBus + '.' + str(phase_order[phase1])).upper()), str(phase_order[phase1]), winding_connections])
                    for phase2 in range(num_phases):
                        offDiagFromIdx = self.nodeOrder.index((fromBus + '.' + str(phase_order[phase2])).upper())
                        toIdx = self.nodeOrder.index((toBus + '.' + str(phase_order[phase2])).upper())
                        bus_to_bus_connection[fromIdx, offDiagFromIdx] = 1
                        bus_to_bus_connection[fromIdx, toIdx] = 1
                        
                for phase1 in range(num_phases):
                    toIdx = self.nodeOrder.index((toBus + '.' + str(phase_order[phase1])).upper())
                    for phase2 in range(num_phases):
                        offDiagToIdx = self.nodeOrder.index((toBus + '.' + str(phase_order[phase2])).upper())
                        fromIdx = self.nodeOrder.index((fromBus + '.' + str(phase_order[phase2])).upper())
                        bus_to_bus_connection[toIdx, offDiagToIdx] = 1
                        bus_to_bus_connection[toIdx, fromIdx] = 1

                if not self.dss.ActiveCircuit.Transformers.Next > 0:
                    break
        
        # Get power flows for switches
        self.dss.ActiveCircuit.Lines.First
        while True:
            if self.dss.ActiveCircuit.Lines.IsSwitch:
                fromBus = self.dss.ActiveCircuit.Lines.Bus1.split('.')[0]
                toBus  = self.dss.ActiveCircuit.Lines.Bus2.split('.')[0]
                num_phases = self.dss.ActiveCircuit.Lines.Phases
                # num_phases = len(self.dss.ActiveCircuit.Lines.Bus1.split('.')[1:])
                phase_order = self.dss.ActiveCircuit.ActiveCktElement.NodeOrder
                switch = self.dss.ActiveCircuit.CktElements(f"Line.{self.dss.ActiveCircuit.Lines.Name}")
                switchStatus = int(switch.IsOpen(2,0))
                for phase1 in range(num_phases):
                    if not switchStatus:
                        self.branchNumber += 1
                    fromIdx = self.nodeOrder.index((fromBus + '.' + str(phase_order[phase1])).upper())
                    toIdx = self.nodeOrder.index((toBus + '.' + str(phase_order[phase1])).upper())
                    self.branchListThetaInitialization.append([(fromBus).upper(),fromIdx,(toBus).upper(), self.nodeOrder.index((toBus + '.' + str(phase_order[phase1])).upper()), str(phase_order[phase1]), 'Switch'])
                    bus_to_bus_connection[fromIdx, fromIdx] = 1
                    bus_to_bus_connection[fromIdx, toIdx] = 1
                    bus_to_bus_connection[toIdx, fromIdx] = 1
                    bus_to_bus_connection[toIdx, toIdx] = 1
            if not self.dss.ActiveCircuit.Lines.Next > 0:
                break
        return bus_to_bus_connection

    def extract_open_switch_flows(self):
        open_switch_flows_idx = []
        deleteFlowIdx = []
        for switch in self.switches:
            if switch[5].astype(int) == 1:
                idx = np.where(((self.realFlow[:,0] == switch[0]) & ((self.realFlow[:,2] == switch[2])) & (self.realFlow[:,4] == switch[4])) |  ((self.realFlow[:,0] == switch[2]) & ((self.realFlow[:,2] == switch[0])) & (self.realFlow[:,4] == switch[4])))[0]
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

    def getTransformerTap(self):
        self.transformerTap = []
        self.dss.ActiveCircuit.Transformers.First
        if self.dss.ActiveCircuit.Transformers.Count > 0:
            while True:
                fromBus = self.dss.ActiveCircuit.ActiveCktElement.BusNames[0].split('.')[0]
                toBus = self.dss.ActiveCircuit.ActiveCktElement.BusNames[1].split('.')[0]
                num_phases = self.dss.ActiveCircuit.ActiveCktElement.NumPhases
                phase_order = self.dss.ActiveCircuit.ActiveElement.NodeOrder
                
                self.dss.ActiveCircuit.Transformers.Wdg = 1
                tap_1 = self.dss.ActiveCircuit.Transformers.Tap

                self.dss.ActiveCircuit.Transformers.Wdg = 2
                tap_2 = self.dss.ActiveCircuit.Transformers.Tap
                
                if tap_1 != 1:
                    tap = tap_1
                else:
                    tap = tap_2

                for phase in range(num_phases):
                    fromIdx = self.nodeOrder.index((fromBus + '.' + str(phase_order[phase])).upper())
                    toIdx = self.nodeOrder.index((toBus + '.' + str(phase_order[phase])).upper())
                    idx = phase * 2     
                    self.transformerTap.append([fromBus, fromIdx, toBus, toIdx, phase_order[phase],tap])

                if not self.dss.ActiveCircuit.Transformers.Next > 0:
                    self.transformerTap = np.array(self.transformerTap)
                    break

    def monte_carlo_simulation(self, run_size=100):
        missed_bad_data_counts_WLS = []
        false_bad_data_counts_WLS = []
        true_bad_data_counts_WLS = []

        missed_bad_data_counts_LAV = []
        false_bad_data_counts_LAV = []
        true_bad_data_counts_LAV = []

        baddata_seed = []
        noise_seed = []
        
        lav_converged = []
        wls_converged = []
        
        overall_mse_WLS = []
        overall_mse_LAV = []
        # results = []

        bad_data_number = [5,10,15,20]
        bad_dataCounter = []
        for run in range(run_size):            
            if run % int(run_size/10) == 0:
                print(run)
            bad_data_idx = np.divmod(run, int(run_size/len(bad_data_number)))[0]
            bad_data = bad_data_number[bad_data_idx]
 
            self.__init__(dss, badDataNumber = bad_data, seedCounter=run)
            # Run the solve method to perform state estimation and bad data detection
            self.solve()
            
            lav_converged.append(self.LAV_converged)
            wls_converged.append(self.WLS_converged)
            
            overall_mse_WLS.append(self.overall_mse_WLS)
            overall_mse_LAV.append(self.overall_mse_LAV)
            
            bad_dataCounter.append(bad_data)
                        
            baddata_seed.append(self.externalSimulationActual.baddata_seed)
            noise_seed.append(self.externalSimulationActual.noise_seed)
            # Extract the detected and true bad data indices
            detected_bad_data_WLS = self.badData_WLS['Measurement Idx'].tolist()
            detected_bad_data_LAV = self.badData_LAV['Measurement Idx'].tolist()
            true_bad_data = self.trueBadDataLocation['Measurement Idx'].tolist()

            # Initialize counters for missed, false, and true bad data for WLS and LAV
            missed_bad_data_WLS = 0
            false_bad_data_WLS = 0
            true_bad_data_detected_WLS = 0

            missed_bad_data_LAV = 0
            false_bad_data_LAV = 0
            true_bad_data_detected_LAV = 0

            # Check for missed bad data in WLS
            for true_data in true_bad_data:
                if true_data not in detected_bad_data_WLS:
                    missed_bad_data_WLS += 1

            # Check for false bad data and correctly detected bad data in WLS
            for detected_data in detected_bad_data_WLS:
                if detected_data in true_bad_data:
                    true_bad_data_detected_WLS += 1
                else:
                    false_bad_data_WLS += 1

            # Check for missed bad data in LAV
            for true_data in true_bad_data:
                if true_data not in detected_bad_data_LAV:
                    missed_bad_data_LAV += 1

            # Check for false bad data and correctly detected bad data in LAV
            for detected_data in detected_bad_data_LAV:
                if detected_data in true_bad_data:
                    true_bad_data_detected_LAV += 1
                else:
                    false_bad_data_LAV += 1

            missed_bad_data_counts_WLS.append(missed_bad_data_WLS)
            false_bad_data_counts_WLS.append(false_bad_data_WLS)
            true_bad_data_counts_WLS.append(true_bad_data_detected_WLS)

            missed_bad_data_counts_LAV.append(missed_bad_data_LAV)
            false_bad_data_counts_LAV.append(false_bad_data_LAV)
            true_bad_data_counts_LAV.append(true_bad_data_detected_LAV)

        # Create a DataFrame to store the results
        results = pd.DataFrame({
            'Noise Seed' : noise_seed,
            'Bad Data Seed': baddata_seed,
            'Run': range(1, run_size + 1),
            'Missed Bad Data WLS': missed_bad_data_counts_WLS,
            'False Bad Data WLS': false_bad_data_counts_WLS,
            'True Bad Data WLS': true_bad_data_counts_WLS,
            'Missed Bad Data LAV': missed_bad_data_counts_LAV,
            'False Bad Data LAV': false_bad_data_counts_LAV,
            'True Bad Data LAV': true_bad_data_counts_LAV,
            'WLS Convergence': wls_converged,
            'WLS MSE' : overall_mse_WLS,
            'LAV Convergence': lav_converged,
            'LAV MSE' : overall_mse_LAV,
            'Bad Data Number': bad_dataCounter
        })
        summary = {
            'Mean Missed Bad Data WLS': np.mean(missed_bad_data_counts_WLS),
            'Mean False Bad Data WLS': np.mean(false_bad_data_counts_WLS),
            'Mean True Bad Data WLS': np.mean(true_bad_data_counts_WLS),
            'Mean Missed Bad Data LAV': np.mean(missed_bad_data_counts_LAV),
            'Mean False Bad Data LAV': np.mean(false_bad_data_counts_LAV),
            'Mean True Bad Data LAV': np.mean(true_bad_data_counts_LAV),
        }

        print("\nSummary Statistics:")
        print(summary)

        return results

    def monte_carlo_simulation_crossing_badData(self, run_size=None):
        missed_bad_data_counts = []
        false_bad_data_counts = []
        true_bad_data_counts = []

        baddata_seed = []
        noise_seed = []
        
        lav_converged = []
        wls_converged = []
        
        overall_mse_WLS = []
        overall_mse_LAV = []
        # results = []
        counter_badData = 0
        bad_data_number = [10,10,20]
        bad_dataCounter = []
        for run in range(run_size):
            if run == 0:
                bad_data = bad_data_number[counter_badData]
                counter_badData += 1
            elif run == (run_size/3):
                bad_data = bad_data_number[counter_badData]
                counter_badData += 1
            elif run == 2*(run_size/3):
                bad_data = bad_data_number[counter_badData]
                counter_badData += 1
            
            self.__init__(dss, badDataNumber = bad_data)
            # Run the solve method to perform state estimation and bad data detection
            finalBadData = self.solve()
            
            lav_converged.append(self.LAV_converged)
            wls_converged.append(self.WLS_converged)
            
            overall_mse_WLS.append(self.overall_mse_WLS)
            overall_mse_LAV.append(self.overall_mse_LAV)
            
            bad_dataCounter.append(bad_data)
                        
            baddata_seed.append(self.externalSimulationActual.baddata_seed)
            noise_seed.append(self.externalSimulationActual.noise_seed)
            # Extract the detected and true bad data indices
            detected_bad_data_WLS = finalBadData.tolist()
            true_bad_data = self.trueBadDataLocation['Measurement Idx'].tolist()

            # Initialize counters for missed, false, and true bad data for WLS and LAV
            missed_bad_data = 0
            false_bad_data = 0
            true_bad_data_detected = 0

            # Check for missed bad data in WLS
            for true_data in true_bad_data:
                if true_data not in detected_bad_data_WLS:
                    missed_bad_data += 1

            # Check for false bad data and correctly detected bad data in WLS
            for detected_data in detected_bad_data_WLS:
                if detected_data in true_bad_data:
                    true_bad_data_detected += 1
                else:
                    false_bad_data += 1

            missed_bad_data_counts.append(missed_bad_data)
            false_bad_data_counts.append(false_bad_data)
            true_bad_data_counts.append(true_bad_data_detected)

        # Create a DataFrame to store the results
        results = pd.DataFrame({
            'Noise Seed' : noise_seed,
            'Bad Data Seed': baddata_seed,
            'Run': range(1, run_size + 1),
            'Missed Bad Data WLS': missed_bad_data_counts,
            'False Bad Data WLS': false_bad_data_counts,
            'True Bad Data WLS': true_bad_data_counts,
            'WLS Convergence': wls_converged,
            'WLS MSE' : overall_mse_WLS,
            'LAV Convergence': lav_converged,
            'LAV MSE' : overall_mse_LAV,
            'Bad Data Number': bad_dataCounter
        })

        summary = {
            'Mean Missed Bad Data WLS': np.mean(missed_bad_data_counts),
            'Mean False Bad Data WLS': np.mean(false_bad_data_counts),
            'Mean True Bad Data WLS': np.mean(true_bad_data_counts),
        }


        # print("Monte Carlo Simulation Results:")
        # print(results)
        print("\nSummary Statistics:")
        print(summary)
        # for key, value in summary.items():
        #     print(f"{key}: {value}")

        return results

# main = StateEstimator(dss)
# sol = main.solve()

