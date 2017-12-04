import datetime as dt
import os

from storage.run_store_handle import RunStoreHandle


class DataStorage:
    def __init__(self, data_storage_path, tag):
        seriesTimestamp = dt.datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
        self.seriesStoragePath = os.path.join(data_storage_path, f"{seriesTimestamp}--{tag}")
        self.recordedRuns = {}
        if not os.path.exists(self.seriesStoragePath):
            os.makedirs(self.seriesStoragePath)

    def open_run_store(self, run_index):
        if run_index not in self.recordedRuns:
            run_desc = {
                "index": run_index,
                "runStorePath": self.get_run_store_path(run_index)
            }
            self.recordedRuns.update({run_index: run_desc})
            return RunStoreHandle(run_desc)
        else:
            return RunStoreHandle(self.recordedRuns[run_index])

    def get_run_store_path(self, run_index):
        return os.path.join(self.seriesStoragePath, f"running{run_index}")
