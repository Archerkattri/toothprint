import sys
import unittest
from unittest.mock import MagicMock, patch


class GpuRuntimeTests(unittest.TestCase):
    def test_runtime_defaults_expose_primary_cuda_device(self):
        from dcc.runtime import DEFAULT_ENV, gpu_runtime_env

        self.assertEqual(DEFAULT_ENV["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(gpu_runtime_env({"CUDA_VISIBLE_DEVICES": "1"})["CUDA_VISIBLE_DEVICES"], "0")

    def test_gpu_device_returns_cpu_when_torch_unavailable(self):
        """gpu_device() returns 'cpu' when torch cannot be imported."""
        from dcc.runtime import gpu_device

        with patch.dict(sys.modules, {"torch": None}):
            result = gpu_device()
        self.assertEqual(result, "cpu")

    def test_gpu_device_returns_cpu_when_cuda_not_available(self):
        """gpu_device() returns 'cpu' when torch.cuda.is_available() is False."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            from dcc import runtime
            import importlib
            importlib.reload(runtime)
            result = runtime.gpu_device()

        self.assertEqual(result, "cpu")

    def test_gpu_device_returns_cuda_when_available(self):
        """gpu_device() returns 'cuda:N' when CUDA is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.current_device.return_value = 0

        with patch.dict(sys.modules, {"torch": mock_torch}):
            from dcc.runtime import gpu_device
            result = gpu_device()

        self.assertEqual(result, "cuda:0")


if __name__ == "__main__":
    unittest.main()
