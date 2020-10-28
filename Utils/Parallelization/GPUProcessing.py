import numpy as np
from timeit import default_timer as timer
from numba import vectorize
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2'    
os.environ['CUDA_PATH_V10_2'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2'    
os.environ['NUMBAPRO_CUDALIB'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin'        
os.environ['NUMBAPRO_LIBDEVICE'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\nvvm\libdevice'
os.environ['NUMBAPRO_NVVM'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\nvvm\bin\nvvm64_31_0.dll'

os.environ['CUDA_HOME'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2'    

def pow(a, b, c):
    for i in range(a.size):
         c[i] = a[i] ** b[i]


#def main():
#    vec_size = 100000000

#    a = b = np.array(np.random.sample(vec_size), dtype=np.float32)
#    c = np.zeros(vec_size, dtype=np.float32)

#    start = timer()
#    pow(a, b, c)
#    duration = timer() - start

#    print(duration)



@vectorize(['float32(float32, float32)'], target='cuda')
def pow(a, b):
    return a ** b

def main2():
    vec_size = 100000000

    a = b = np.array(np.random.sample(vec_size), dtype=np.float32)
    c = np.zeros(vec_size, dtype=np.float32)

    start = timer()
    c = pow(a, b)
    duration = timer() - start

    print(duration)



if __name__ == '__main__':
    #main()
    main2()
