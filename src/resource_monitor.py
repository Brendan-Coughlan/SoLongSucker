import time
import psutil
import threading
import os
import numpy as np

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

class ResourceMonitor:
    def __init__(self, interval=1.0, gpu_index=0):
        self.interval = interval
        self.gpu_index = gpu_index

        self.running = False
        self.thread = None

        self.history = []

        self.process = psutil.Process(os.getpid())

        if GPU_AVAILABLE:
            self.gpu = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    def start(self):
        if self.running:
            return

        self.running = True
        self.history.clear()

        psutil.cpu_percent(interval=None)

        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.running:
            return

        self.running = False

        if self.thread is not None:
            self.thread.join()

        return self.history

    def get_resource_metrics(self):
        cpu_percent = psutil.cpu_percent()

        memory = psutil.virtual_memory()
        process_memory = self.process.memory_info().rss / (1024 ** 3)

        resource_metrics = {
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "ram_percent": memory.percent,
            "ram_used_gb": memory.used / (1024 ** 3),
            "ram_available_gb": memory.available / (1024 ** 3),
            "process_ram_gb": process_memory
        }

        if GPU_AVAILABLE:
            gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu)

            resource_metrics.update({
                "gpu_percent": gpu_utilization.gpu,
                "gpu_memory_used_gb": memory_info.used / (1024 ** 3),
                "gpu_memory_total_gb": memory_info.total / (1024 ** 3),
                "gpu_memory_percent": memory_info.used / memory_info.total * 100,
                "gpu_temperature": pynvml.nvmlDeviceGetTemperature(self.gpu, pynvml.NVML_TEMPERATURE_GPU),
                "gpu_power_draw_watts": pynvml.nvmlDeviceGetPowerUsage(self.gpu) / 1000
            })
        else:
            resource_metrics.update({
                "gpu_percent": None,
                "gpu_memory_used_gb": None,
                "gpu_memory_total_gb": None,
                "gpu_memory_percent": None,
                "gpu_temperature": None,
                "gpu_power_draw_watts": None
            })

        return resource_metrics

    def _monitor(self):
        while self.running:
            resource_metrics = self.get_resource_metrics()
            self.history.append(resource_metrics)
            time.sleep(self.interval)

    def get_summary(self):
        if not self.history:
            return {}

        summary = {}
        keys = self.history[0].keys()
        for key in keys:
            if key == "timestamp":
                continue
            values = [entry[key] for entry in self.history if entry[key] is not None]
            if values:
                summary[key] = {
                    "mean": np.mean(values),
                    "max": np.max(values),
                    "min": np.min(values)
                }
            else:
                summary[key] = {
                    "mean": None,
                    "max": None,
                    "min": None
                }

        return summary