import numba
import numba.cuda
import os

os.environ['CUDA_HOME'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2'    

numba.cuda.api.detect()

numba.cuda.is_available()