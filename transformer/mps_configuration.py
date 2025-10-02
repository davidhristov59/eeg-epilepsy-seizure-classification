import psutil
import torch

class DeviceManager:
    """Handles device detection and optimization for Mac/MPS"""

    @staticmethod
    def get_optimal_device():
        """Get the best available device with Mac optimization"""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            device_name = "Apple Silicon GPU (MPS)"
            print(f"Using {device_name}")

            # Set optimal CPU threads for MPS
            cpu_count = psutil.cpu_count()
            torch.set_num_threads(min(8, cpu_count))
            print(f"CPU threads: {torch.get_num_threads()}")

        else: # if no MPS available, fallback to CPU
            device = torch.device("cpu")
            device_name = "CPU"

            # Optimize for ARM CPU
            cpu_count = psutil.cpu_count()
            optimal_threads = min(8, cpu_count)
            torch.set_num_threads(optimal_threads)
            print(f"Using {device_name} with {optimal_threads} threads")

        return device, device_name

    @staticmethod
    def clear_memory(device):
        """Clear memory cache based on device"""
        if device.type == "mps":
            torch.mps.empty_cache()

    @staticmethod
    def get_memory_info(device):
        """Get memory information if available"""
        if device.type == "mps":
            return "MPS memory management handled automatically"

        else:
            memory = psutil.virtual_memory()
            return f"RAM: {memory.used / 1024 ** 3:.1f}GB / {memory.total / 1024 ** 3:.1f}GB ({memory.percent:.1f}% used"