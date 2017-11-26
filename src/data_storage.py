import json
import os
import datetime as dt

class DataStorage:
    def __init__(self, dataStoragePath=f"../data"):
        runTimestamp = dt.datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
        self.runStoragePath = os.path.join(dataStoragePath, f"{runTimestamp}")
        if not os.path.exists(self.runStoragePath):
            os.makedirs(self.runStoragePath)

    def start_new_seed(self, seedIndex, netConfiguration):
        self.currentSeedStoragePath = os.path.join(self.runStoragePath, f"seed{seedIndex}")
        if not os.path.exists(self.currentSeedStoragePath):
            os.makedirs(self.currentSeedStoragePath)
        with open(os.path.join(self.currentSeedStoragePath, f"net-configuration"), 'x') as configFile:
            configFile.write(json.dumps(netConfiguration))

    def save_data_point(self, netState, dataPointIndex):
        with open(os.path.join(self.currentSeedStoragePath, f"data-point{dataPointIndex}"), 'x') as dataFile:
            dataFile.write(json.dumps(netState))

    def get_net_configuration(self, seedIndex):
        with open(os.path.join(self.runStoragePath, f"seed{seedIndex}/net-configuration")) as configFile:
            return json.loads(configFile.read())

    def get_data_point(self, seedIndex, dataPointIndex):
        with open(os.path.join(self.runStoragePath, f"seed{seedIndex}/data-point{dataPointIndex}"), 'r') as dataFile:
            return json.loads(dataFile.read())