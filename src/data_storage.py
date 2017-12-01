import json
import os
import datetime as dt
import sys
from plotter import Plotter

class RunStoreHandle:
    def __init__(self, parentStorage, runDesc):
        self.runStorePath = runDesc["runStorePath"]
        self.configFilePath = os.path.join(self.runStorePath, "net-config")
        self.dataFilePath = os.path.join(self.runStorePath, "data-points")
        self.imagesStorePath = os.path.join(self.runStorePath, "images")
        if not os.path.exists(self.runStorePath):
            os.makedirs(self.runStorePath)
        if not os.path.exists(self.imagesStorePath):
            os.makedirs(self.imagesStorePath)
        self.runData = self.load_run_data()

    def __len__(self):
        return len(self.runData)

    def add_data_point(self, dataPoint):
        self.runData.append(dataPoint)

    def commit_data(self):
        with open(self.dataFilePath, 'w') as dataFile:
            dataFile.write(json.dumps(self.runData))

    def get_data_point(self, dataPointIndex):
        return self.runData[dataPointIndex]

    def load_run_data(self):
        if os.path.exists(self.dataFilePath):
            with open(self.dataFilePath, 'r') as dataFile:
                return json.loads(dataFile.read())
        else:
            return []

    def store_net_config(self, netConfig):
        with open(self.configFilePath, 'w') as configFile:
            configFile.write(json.dumps(netConfig))

    def get_net_config(self):
        if os.path.exists(self.configFilePath):
            with open(self.configFilePath, 'r') as configFile:
                return json.loads(configFile.read())

    def get_image_path_regexp(self):
        return os.path.join(self.imagesStorePath, "img%d.png")

    def get_image_path(self, dataPointIndex):
        return os.path.join(self.imagesStorePath, f"img{dataPointIndex}.png")

    def get_video_path(self):
        return os.path.join(self.runStorePath, "run.mp4")

class DataStorage:
    def __init__(self, dataStoragePath=f"../data"):
        seriesTimestamp = dt.datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
        self.seriesStoragePath = os.path.join(dataStoragePath, f"{seriesTimestamp}")
        self.recordedRuns = {}
        if not os.path.exists(self.seriesStoragePath):
            os.makedirs(self.seriesStoragePath)

    def open_run_store(self, runIndex):
        if runIndex not in self.recordedRuns:
            runDesc = {
                "index": runIndex,
                "runStorePath": self.get_run_store_path(runIndex)
            }
            self.recordedRuns.update({runIndex: runDesc})
            return RunStoreHandle(self, runDesc)
        else:
            return RunStoreHandle(self, self.recordedRuns[runIndex])

    def get_run_store_path(self, runIndex):
        return os.path.join(self.seriesStoragePath, f"run{runIndex}")
