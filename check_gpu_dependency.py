import torch
print("PyTorch CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())

import onnxruntime as ort

sess_options = ort.SessionOptions()
try:
    ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"], sess_options=sess_options)
    print("CUDA Execution Provider loaded successfully.")
except Exception as e:
    print(f"Failed to load CUDA Execution Provider: {e}")
