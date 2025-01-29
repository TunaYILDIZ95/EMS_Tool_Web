import numpy as np
import pandas as pd
from dss import dss
from scipy.sparse import csc_matrix
from scipy.optimize import linprog
import copy
from operator import itemgetter
from cmath import rect


class OpenDSS_Sim_v2:
    def __init__(self,dss, fileLocation = None, switchStatus = None, IsBadData = None, IsNoise = None, badDataNumber = None, seedCounter = None):
        
        self.badDataNumber = badDataNumber
        self.seedCounter = seedCounter
        self.dss = dss
        self.dss('redirect ' + fileLocation)
        self.dss.ActiveCircuit.Solution.Solve()

        if switchStatus is not None:
            self.set_switch_status(switchStatus=switchStatus)

        self.loads_vsource_and_solve(option = 1)  
        
        # Initialize the variables      
        self.initialize_variables(IsBadData = IsBadData, IsNoise = IsNoise)
        
    def initialize_variables(self,IsTrue = None, IsBadData = None, IsNoise = None):
        self.systemSize = len(self.dss.ActiveCircuit.YNodeOrder)
        self.base_power_mva = 1  # Example: 10 MVA
        self.base_power_kw = self.base_power_mva * 1000  # Convert MVA to kW for calculations
        self.systemFreq = 60 # Hz
        self.nodeOrder = self.dss.ActiveCircuit.YNodeOrder
        
        self.voltageBases = self.get_bus_base_voltages()     
        
        self.Ybus, self.YMatrix = self.get_YMatrix(self.voltageBases, self.base_power_mva)  
        #### Enable Vsource and Loads to get the measurements
        self.loads_vsource_and_solve(option=2)
         
        ### Get System Measurements
        self.voltages, self.angles  = self.get_bus_voltages()      
        self.realInjection, self.reacInjection, self.voltageMagPMU, self.voltageAnglePMU = self.get_powerInjections_in_pu(self.voltages[:,2].astype(float), np.deg2rad(self.angles[:,2].astype(float)), self.Ybus)
        self.realFlow, self.reacFlow, self.currentMagPMU, self.currentAnglePMU = self.get_powers_in_pu(self.base_power_kw)
        
        ### Get Switch Measurements
        self.switches = self.get_switches(self.base_power_kw)

        ### Add noise to the measurements
        if IsNoise:
            self.add_noise()
        ### Add bad data to measurements
        # self.trueBadDataLocation = pd.DataFrame(columns = ["FromBus", "ToBus", "Type", "Phase", "Before Bad Data", "After Bad Data"])
        self.trueBadDataLocation = pd.DataFrame(columns=["Measurement Idx", "Type", "Phase", "Before Bad Data", "After Bad Data"])
        if IsBadData:
            self.add_baddata()

    def set_switch_status(self,switchStatus = None):
        for index, row in switchStatus.iterrows():
            switch_name = row['SwitchName']
            status = row['Status'].lower()
            switch = self.dss.ActiveCircuit.CktElements(f"Line.{switch_name}")
            if status == "open":
                switch.Open(2, 0)
            elif status == "closed":
                switch.Close(2, 0)


    def add_noise(self):
        # self.noise_seed = np.random.randint(0, 1000)
        self.noise_seed = self.seedCounter
        # self.noise_seed = 598259
        # self.noise_seed = 929555
        # self.noise_seed = 254461
        # print("noise seed is " + str(self.noise_seed))
        np.random.seed(self.noise_seed)  # Set seed for reproducibility - will be deleted later
        # np.random.seed(253213)
        # np.random.seed(745784)

        #SCADA 
        std_dev_power_flow_percentage = 0.008  # 1%
        std_dev_power_injection_percentage = 0.01  # 1%
        std_dev_voltage_percentage = 0.004  # 0.2%

        #PMU
        std_dev_pmu_percentage = 0.0001

        # std_dev_power_flow_percentage = 0.001  # 1%
        # std_dev_power_injection_percentage = 0.005  # 1%
        # std_dev_voltage_percentage = 0.0002  # 0.2%

        # Add noise to realInjection
        real_injection_col = self.realInjection[:, 2].astype(float)
        real_injection_col += np.random.normal(0, std_dev_power_injection_percentage, len(real_injection_col))
        self.realInjection[:, 2] = real_injection_col

        # Add noise to reacInjection
        reac_injection_col = self.reacInjection[:, 2].astype(float)
        reac_injection_col += np.random.normal(0, std_dev_power_injection_percentage, len(reac_injection_col))
        self.reacInjection[:, 2] = reac_injection_col

        # Add noise to realFlow
        real_flow_col = self.realFlow[:, 5].astype(float)
        real_flow_col += np.random.normal(0, std_dev_power_flow_percentage, len(real_flow_col))
        self.realFlow[:, 5] = real_flow_col

        # Add noise to reacFlow
        reac_flow_col = self.reacFlow[:, 5].astype(float)
        reac_flow_col += np.random.normal(0, std_dev_power_flow_percentage, len(reac_flow_col))
        self.reacFlow[:, 5] = reac_flow_col

        # Add noise to voltages
        voltage_col = self.voltages[:, 2].astype(float)
        voltage_col += np.random.normal(0, std_dev_voltage_percentage, len(voltage_col))
        self.voltages[:, 2] = voltage_col

        # Add noise to PMU
        voltage_PMU_col = self.voltageMagPMU[:,2].astype(float)
        voltage_PMU_col += np.random.normal(0, std_dev_pmu_percentage, len(voltage_PMU_col))
        angle_col = self.voltageAnglePMU[:,2].astype(float)
        angle_col += np.random.normal(0, std_dev_pmu_percentage, len(angle_col))
        self.voltageMagPMU[:,2] = voltage_PMU_col
        self.voltageAnglePMU[:,2] = angle_col


    def add_baddata(self):
        # total_bad_data = 10  # Total number of bad data points
        total_bad_data = self.badDataNumber  # Total number of bad data points
        self.baddata_seed = self.seedCounter
        # self.baddata_seed = np.random.randint(0, 1000)
        # self.baddata_seed = 920664
        # self.baddata_seed = 741761
        # self.baddata_seed = 463910
        # print("bad data seed is " + str(self.baddata_seed))
        np.random.seed(self.baddata_seed)  # Set seed for reproducibility - will be deleted later
        # np.random.seed(36351)
        # np.random.seed(48506)

        # Generate random distribution of bad data points summing to total_bad_data
        num_bad_data_per_array = np.random.multinomial(total_bad_data, [1 / 7] * 7)

        # Define a function to insert bad data into a given array
        def insert_bad_data(array, num_bad_data, array_type, type, location):
            if num_bad_data > len(array):
                num_bad_data = len(array)  # Ensure we don't exceed the array length
            indices = np.random.choice(len(array), num_bad_data, replace=False)
            for idx in indices:
                if type == 'Injection':
                    # self.trueBadDataLocation.loc[len(self.trueBadDataLocation)] = [array_type[idx, 0], "~", "Injection" , array_type[idx, 1], array[idx], -array[idx]]
                    self.trueBadDataLocation.loc[len(self.trueBadDataLocation)] = [idx + location, "Injection", array_type[idx, 1], array[idx], -array[idx]]
                elif type == 'Flow':
                    # self.trueBadDataLocation.loc[len(self.trueBadDataLocation)] = [array_type[idx, 0], array_type[idx, 2], "Flow" , array_type[idx, 4], array[idx], -array[idx]]
                    self.trueBadDataLocation.loc[len(self.trueBadDataLocation)] = [idx + location, "Flow", array_type[idx, 4], array[idx], -array[idx]]
                elif type == 'Voltage':
                    # self.trueBadDataLocation.loc[len(self.trueBadDataLocation)] = [array_type[idx, 0], "~", "Voltage", array_type[idx, 1], array[idx], -array[idx]]
                    self.trueBadDataLocation.loc[len(self.trueBadDataLocation)] = [idx + location, "Voltage", array_type[idx, 1], array[idx], -array[idx]]
                elif type == 'Voltage PMU Mag':
                    self.trueBadDataLocation.loc[len(self.trueBadDataLocation)] = [idx + location, "Voltage PMU Mag", array_type[idx, 1], array[idx], array[idx]*0.9]
                elif type == 'Voltage PMU Angle':
                    self.trueBadDataLocation.loc[len(self.trueBadDataLocation)] = [idx + location, "Voltage PMU Angle", array_type[idx, 1], array[idx], array[idx]*0.9]
                # array[idx] += 1
                # array[idx] = 5
                if abs(array[idx]<0.2):
                    if type == 'Voltage PMU Mag' or type == 'Voltage PMU Angle':
                        array[idx] *= 0.9
                    else:
                        array[idx] += 2
                else:
                    if type == 'Voltage PMU Mag' or type == 'Voltage PMU Angle':
                        array[idx] *= 0.9
                    else:
                        array[idx] = -array[idx]
                # array[idx] += 0.5
            return array


        # Insert bad data into realFlow
        self.realFlow[:, 5] = insert_bad_data(self.realFlow[:, 5].astype(float), num_bad_data_per_array[0],self.realFlow,'Flow', location = 2 * len(self.realInjection))

        # Insert bad data into realInjection
        self.realInjection[:, 2] = insert_bad_data(self.realInjection[:, 2].astype(float), num_bad_data_per_array[1], self.realInjection,'Injection', location = 0)

        # Insert bad data into reacFlow
        self.reacFlow[:, 5] = insert_bad_data(self.reacFlow[:, 5].astype(float), num_bad_data_per_array[2], self.reacFlow,'Flow', location = len(self.realInjection)*2 + len(self.realFlow))

        # Insert bad data into reacInjection
        self.reacInjection[:, 2] = insert_bad_data(self.reacInjection[:, 2].astype(float), num_bad_data_per_array[3], self.reacInjection,'Injection', location = len(self.realInjection))

        # Insert bad data into voltages
        self.voltages[:, 2] = insert_bad_data(self.voltages[:, 2].astype(float), num_bad_data_per_array[4], self.voltages, 'Voltage', location = 2 * len(self.realInjection) + 2 * len(self.realFlow))
        
        # Insert bad data into voltages mag PMU
        asdf = self.voltageMagPMU.copy()
        self.voltageMagPMU[:, 2] = insert_bad_data(self.voltageMagPMU[:, 2].astype(float), num_bad_data_per_array[4], self.voltageMagPMU, 'Voltage PMU Mag', location = 2 * len(self.realInjection) + 2 * len(self.realFlow)+ len(self.voltages))
        
        # Insert bad data into voltages angle PMU
        self.voltageAnglePMU[:, 2] = insert_bad_data(self.voltageAnglePMU[:, 2].astype(float), num_bad_data_per_array[4], self.voltageAnglePMU, 'Voltage PMU Angle', location = 2 * len(self.realInjection) + 2 * len(self.realFlow)+ len(self.voltages)+len(self.voltageMagPMU))

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
    
    def get_bus_voltages(self):        
        busVoltages = []
        busAngles = []
        for bus in self.dss.ActiveCircuit.ActiveBus:
            phases = bus.Nodes
            for phase in range(len(phases)):
                busVoltages.append([bus.Name, phases[phase], bus.puVmagAngle[phase*2]])
                busAngles.append([bus.Name, phases[phase], bus.puVmagAngle[phase*2 + 1]])
                # busVoltages.append([bus.Name, phases[phase], bus.VMagAngle[phase*2]])
                # busAngles.append([bus.Name, phases[phase], bus.VMagAngle[phase*2 + 1]])
        
        mapIndex = [self.nodeOrder.index(busVoltages[i][0].upper() + '.' + str(busVoltages[i][1])) for i in range(len(busVoltages))]
        # Reorder your voltage_values to match YNodeOrder
        reordered_voltages = [0]*len(busVoltages)
        reordered_angles = [0]*len(busAngles)
        for idx, position in enumerate(mapIndex):
            reordered_voltages[position] = busVoltages[idx]
            reordered_angles[position] = busAngles[idx]

        return np.array(reordered_voltages), np.array(reordered_angles)
    
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
        
    def get_powers_in_pu(self, base_power_kw):
        real_flows_pu = []
        reac_flows_pu = []

        currentMagPMU = []
        currentAnglePMU = []
        V_base = self.voltageBases[:,2].astype(float) * 1000
        Z_base = V_base**2/(self.base_power_mva * 1000000)
        I_base = V_base/Z_base
        # Get power flows for lines        
        self.dss.ActiveCircuit.Lines.First
        while True:
            if not self.dss.ActiveCircuit.Lines.IsSwitch:
                fromBus = self.dss.ActiveCircuit.Lines.Bus1.split('.')[0]
                toBus  = self.dss.ActiveCircuit.Lines.Bus2.split('.')[0]
                powers = self.dss.ActiveCircuit.CktElements.Powers  # Get real and reactive power in kW and kVAR
                num_phases = self.dss.ActiveCircuit.Lines.Phases
                # num_phases = len(self.dss.ActiveCircuit.Lines.Bus1.split('.')[1:])
                phase_order = self.dss.ActiveCircuit.ActiveCktElement.NodeOrder

                currents = self.dss.ActiveCircuit.ActiveCktElement.CurrentsMagAng
                
                
                for phase in range(num_phases):
                    fromIdx = self.nodeOrder.index((fromBus + '.' + str(phase_order[phase])).upper())
                    toIdx = self.nodeOrder.index((toBus + '.' + str(phase_order[phase])).upper())
                    idx = phase * 2
                    p_kw = powers[idx]
                    q_kvar = powers[idx + 1]
                    p_pu = p_kw / base_power_kw
                    q_pu = q_kvar / base_power_kw
                    real_flows_pu.append([fromBus, fromIdx, toBus, toIdx, phase_order[phase], p_pu,1])
                    reac_flows_pu.append([fromBus, fromIdx, toBus, toIdx, phase_order[phase], q_pu,1])

                    currentMagPMU.append([fromBus,fromIdx, toBus, toIdx, phase_order[phase], currents[idx]/I_base[fromIdx]])
                    currentAnglePMU.append([fromBus,fromIdx, toBus, toIdx, phase_order[phase], currents[idx+1]])

            if not self.dss.ActiveCircuit.Lines.Next > 0:
                break
            
        # Get power flows for transformers
        self.dss.ActiveCircuit.Transformers.First
        if self.dss.ActiveCircuit.Transformers.Count > 0:
            while True:
                fromBus = self.dss.ActiveCircuit.ActiveCktElement.BusNames[0].split('.')[0]
                toBus = self.dss.ActiveCircuit.ActiveCktElement.BusNames[1].split('.')[0]
                powers = self.dss.ActiveCircuit.ActiveCktElement.Powers
                num_phases = self.dss.ActiveCircuit.ActiveCktElement.NumPhases
                phase_order = self.dss.ActiveCircuit.ActiveElement.NodeOrder
                
                self.dss.ActiveCircuit.Transformers.Wdg = 1
                tap_1 = self.dss.ActiveCircuit.Transformers.Tap

                self.dss.ActiveCircuit.Transformers.Wdg = 2
                tap_2 = self.dss.ActiveCircuit.Transformers.Tap
                
                currents = self.dss.ActiveCircuit.ActiveCktElement.CurrentsMagAng
                
                if tap_1 != 1:
                    tap = tap_1
                else:
                    tap = tap_2

                for phase in range(num_phases):
                    fromIdx = self.nodeOrder.index((fromBus + '.' + str(phase_order[phase])).upper())
                    toIdx = self.nodeOrder.index((toBus + '.' + str(phase_order[phase])).upper())
                    idx = phase * 2
                    p_kw = powers[idx]
                    q_kvar = powers[idx + 1]
                    p_pu = p_kw / base_power_kw
                    q_pu = q_kvar / base_power_kw
                    real_flows_pu.append([fromBus, fromIdx, toBus, toIdx, phase_order[phase], p_pu,tap])
                    reac_flows_pu.append([fromBus, fromIdx, toBus, toIdx, phase_order[phase], q_pu,tap])

                    currentMagPMU.append([fromBus,fromIdx, toBus, toIdx, phase_order[phase], currents[idx]/I_base[fromIdx]])
                    currentAnglePMU.append([fromBus,fromIdx, toBus, toIdx, phase_order[phase], currents[idx+1]])

                if not self.dss.ActiveCircuit.Transformers.Next > 0:
                    break
                 
        # Get power flows for switches
        self.dss.ActiveCircuit.Lines.First
        while True:
            if self.dss.ActiveCircuit.Lines.IsSwitch:
                fromBus = self.dss.ActiveCircuit.Lines.Bus1.split('.')[0]
                toBus  = self.dss.ActiveCircuit.Lines.Bus2.split('.')[0]
                powers = self.dss.ActiveCircuit.CktElements.Powers  # Get real and reactive power in kW and kVAR
                num_phases = self.dss.ActiveCircuit.Lines.Phases
                # num_phases = len(self.dss.ActiveCircuit.Lines.Bus1.split('.')[1:])
                phase_order = self.dss.ActiveCircuit.ActiveCktElement.NodeOrder
                switch = self.dss.ActiveCircuit.CktElements(f"Line.{self.dss.ActiveCircuit.Lines.Name}")
                # self.dss.ActiveCircuit.SetActiveElement(f"Line.{self.dss.ActiveCircuit.Lines.Name}")
                switchStatus = int(switch.IsOpen(2,0))
                # if not switchStatus:
                for phase in range(num_phases):
                    fromIdx = self.nodeOrder.index((fromBus + '.' + str(phase_order[phase])).upper())
                    toIdx = self.nodeOrder.index((toBus + '.' + str(phase_order[phase])).upper())
                    idx = phase * 2
                    p_kw = powers[idx]
                    q_kvar = powers[idx + 1]
                    p_pu = p_kw / base_power_kw
                    q_pu = q_kvar / base_power_kw
                    real_flows_pu.append([fromBus, fromIdx, toBus, toIdx, phase_order[phase], p_pu,1])
                    reac_flows_pu.append([fromBus, fromIdx, toBus, toIdx, phase_order[phase], q_pu,1])

                    currentMagPMU.append([fromBus,fromIdx, toBus, toIdx, phase_order[phase], currents[idx]/I_base[fromIdx]])
                    currentAnglePMU.append([fromBus,fromIdx, toBus, toIdx, phase_order[phase], currents[idx+1]])

            if not self.dss.ActiveCircuit.Lines.Next > 0:
                break
            
        return np.array(real_flows_pu), np.array(reac_flows_pu), np.array(currentMagPMU), np.array(currentAnglePMU)
    
    def get_powerInjections_in_pu(self, voltageMeas, thetaMeas, Ybus):
        real_injections_pu = []
        reac_injections_pu = []
        nprect = np.vectorize(rect)
        complexV = nprect(voltageMeas, thetaMeas)
        I_pu = np.array(Ybus.dot(complexV))
        S_pu = np.multiply(complexV, np.conj(I_pu))
        real_injections_pu = np.transpose([self.voltages[:,0], self.voltages[:,1], np.real(S_pu)])
        reac_injections_pu = np.transpose([self.voltages[:,0], self.voltages[:,1], np.imag(S_pu)])

        voltageMagPMU = np.transpose([self.voltages[:,0], self.voltages[:,1], abs(complexV)])
        voltageAnglePMU = np.transpose([self.voltages[:,0], self.voltages[:,1], np.rad2deg(np.angle(complexV))])
        
        # voltageMagPMU = voltageMagPMU[[0,1,2,3,4,5,23,24,25]]
        # voltageAnglePMU = voltageAnglePMU[[0,1,2,3,4,5,23,24,25]]
        
        return real_injections_pu, reac_injections_pu, voltageMagPMU, voltageAnglePMU
    
    def get_YMatrix(self, V_base_vector = None, S_base = None):
        YMatrix = csc_matrix(self.dss.YMatrix.GetCompressedYMatrix()).toarray()
        V_base = V_base_vector[:,2].astype(float) * 1000
        Z_base = V_base**2/(S_base * 1000000)

        # Normalize Ybus to per-unit and consider transformer connections
        Ybus_pu = YMatrix.copy()
        for i in range(len(Ybus_pu)):
            for j in range(len(Ybus_pu[i])):
                if V_base[i] != V_base[j] and Ybus_pu[i, j] != 0:
                    Ybus_pu[i, j] *= (V_base[i] / V_base[j])
                Ybus_pu[i, j] *= Z_base[j]
                
        # self.dss.ActiveCircuit.Transformers.First
        # if self.dss.ActiveCircuit.Transformers.Count > 0:
        #     while True:
        #         fromBusList = []
        #         toBusList = []
        #         fromBus = self.dss.ActiveCircuit.ActiveCktElement.BusNames[0].split('.')[0]
        #         toBus = self.dss.ActiveCircuit.ActiveCktElement.BusNames[1].split('.')[0]
        #         num_phases = self.dss.ActiveCircuit.ActiveCktElement.NumPhases
        #         phase_order = self.dss.ActiveCircuit.ActiveElement.NodeOrder
                
        #         self.dss.ActiveCircuit.Transformers.Wdg = 1
        #         tap_1 = self.dss.ActiveCircuit.Transformers.Tap
        #         delta = self.dss.ActiveCircuit.Transformers.IsDelta

        #         self.dss.ActiveCircuit.Transformers.Wdg = 2
        #         tap_2 = self.dss.ActiveCircuit.Transformers.Tap
        #         if not delta:
        #             delta = self.dss.ActiveCircuit.Transformers.IsDelta
                
        #         if tap_1 != 1:
        #             tap = tap_1
        #         else:
        #             tap = tap_2

        #         noLoadTap = tap
        #         for phase in range(num_phases):
        #             fromIdx = self.nodeOrder.index((fromBus + '.' + str(phase_order[phase])).upper())
        #             toIdx = self.nodeOrder.index((toBus + '.' + str(phase_order[phase])).upper())
        #             fromBusList.append(fromIdx)
        #             toBusList.append(toIdx)

        #         # if fromBusList[0] in [3,4,5]:
        #         if fromBusList[0] in [0,2,4,6,8,10,12,14,16]:
        #             # idx_list = [1.08125,1.0562500000000001,1.08125]
        #             # tap = idx_list[fromBusList[0]-3]
        #             idx_list = [1.03125, 1.0124999999999997, 1.0187499999999998, 0.98125, 0.9937499999999999, 0.9999999999999999, 1.0187500000000003, 1.0062499999999999,1.0250000000000004] # With switch
        #             # idx_list = [1.03125, 1.0124999999999997, 1.0187499999999998, 0.9874999999999999, 0.9999999999999999, 0.9999999999999999, 1.0250000000000004, 1.0062499999999999,1.0312500000000002] # first switch out
        #             # idx_list = [1.03125, 1.0124999999999997, 1.0187499999999998, 0.9874999999999999, 0.9999999999999999, 0.9999999999999999, 1.0312500000000004, 1.0062499999999999, 1.0312500000000002]
        #             tap = idx_list[np.divmod(fromBusList[0],2)[0]]

        #         if tap_1 != 1:
        #             y = -Ybus_pu[np.ix_(fromBusList, toBusList)]*noLoadTap
        #             Ybus_pu[fromBusList,fromBusList] = Ybus_pu[toBusList,toBusList] - y/noLoadTap**2
        #             Ybus_pu[fromBusList, fromBusList] = Ybus_pu[toBusList,toBusList] + y/tap**2
                    
        #             Ybus_pu[np.ix_(fromBusList, toBusList)] = Ybus_pu[np.ix_(fromBusList, toBusList)]*noLoadTap/tap
        #             Ybus_pu[np.ix_(toBusList, fromBusList)] = Ybus_pu[np.ix_(toBusList, fromBusList)]*noLoadTap/tap
        #         elif tap_2 != 1: 
        #             y = -Ybus_pu[np.ix_(fromBusList, toBusList)]*noLoadTap
        #             Ybus_pu[toBusList,toBusList] = Ybus_pu[toBusList,toBusList] - y/noLoadTap**2
        #             Ybus_pu[toBusList, toBusList] = Ybus_pu[toBusList,toBusList] + y/tap**2
                    
        #             Ybus_pu[np.ix_(fromBusList, toBusList)] = Ybus_pu[np.ix_(fromBusList, toBusList)]*noLoadTap/tap
        #             Ybus_pu[np.ix_(toBusList, fromBusList)] = Ybus_pu[np.ix_(toBusList, fromBusList)]*noLoadTap/tap

        #         if not self.dss.ActiveCircuit.Transformers.Next > 0:
        #             break

        return Ybus_pu, YMatrix
    
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
                    idx = phase * 2
                    p_kw = powers[idx]
                    q_kvar = powers[idx + 1]
                    p_pu = p_kw / base_power_kw
                    q_pu = q_kvar / base_power_kw
                    switchData.append([fromBus, fromIdx, toBus, toIdx, phase_order[phase], switchStatus, p_pu, q_pu])
            if not self.dss.ActiveCircuit.Lines.Next > 0:
                break
        return np.array(switchData)
    
    def open_close_switches(self, switchLinesNames = None):
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
