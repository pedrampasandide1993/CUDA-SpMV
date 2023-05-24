# CUDA-SpMV




<div style = "text-align:center;font-size:20px">
CAS781 - Code Description: Sparse matrix multiplication (A*x) where A is a sparse matrix and x is a dense matrix on GPU and comparing the results with OpenMP on CPU
</div>

<br>

<div style = "text-align:center;font-size:15px">
Pedram Pasandide
</div>

<br>
<br>
<br>
<br>


**<span style="font-size:16px"> Compiler and CUDA setup </span>**

**<span style="font-size:16px"> a) Removing CUDA toolkit if necessary </span>**

Use `nvidia-smi` to see the version of CUDA on your device. Use `nvcc --version` to check the CUDA toolkit installed on your device. If you have a CUDA toolkit that the version is not compatible with device follow the steps:

1. Type the following command to remove the toolkit and all its dependencies:

s`udo apt-get remove --autoremove nvidia-cuda-toolkit`

2. If you also want to remove any configuration files associated with the package, you can use the following command:

`sudo apt-get purge nvidia-cuda-toolkit`

if you have installed another version of cuda lets say X.Y and you need to removed it before installing a specific version:
3. Remove the CUDA Toolkit and its dependencies:

`sudo apt-get --purge remove cuda`

`sudo apt-get autoremove`

4. Remove the NVIDIA drivers installed by the CUDA Toolkit:

`sudo apt-get --purge remove "*cublas*" "*cufft*" "*curand*" "*cusolver*" "*cusparse*" "cuda*" "nsight*"`

`sudo apt-get autoremove`

5. Remove the CUDA directories:

`sudo rm -rf /usr/local/cuda-X.Y/`

6. Remove any references to CUDA from your environment variables, by editing the .bashrc file:

`nano ~/.bashrc`

<br>
<br>

**<span style="font-size:16px"> b) Installing CUDA toolkit </span>**

Download CUDA from (CUDA 12.0)

https://developer.nvidia.com/cuda-12-0-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network

Follow the step to install CUDA toolkit. if you don't have permission use sudo before wget:

1. `wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb`

2. `sudo dpkg -i cuda-keyring_1.0-1_all.deb`

3. `sudo apt-get update`

4. `sudo apt-get -y install cuda`

5. Post installation: add the following variable to `nano ~/.bashrc`

`export PATH=/usr/local/cuda-12.0/bin${PATH:+:${PATH}}`

<br>
<br>

**<span style="font-size:16px"> c) Installing HPC SDK 23.3 </span>**


1. `wget https://developer.download.nvidia.com/hpc-sdk/23.3/nvhpc_2023_233_Linux_x86_64_cuda_12.0.tar.gz`

2. `tar xpzf nvhpc_2023_233_Linux_x86_64_cuda_12.0.tar.gz`

run this one with sudo if u have permission error

3. `nvhpc_2023_233_Linux_x86_64_cuda_12.0/install`

4. Post installation: add the following variables to `nano ~/.bashrc`

`export NVHPC_HOME=/opt/nvidia/hpc_sdk`

`export MANPATH=$MANPATH:$NVHPC_HOME/Linux_x86_64/23.3/compilers/man`

`export PATH=$NVHPC_HOME/Linux_x86_64/23.3/compilers/bin:$PATH`

`export LD_LIBRARY_PATH=$NVHPC_HOME/Linux_x86_64/23.3/compilers/lib:$LD_LIBRARY_PATH`


Use `nvcc --version` to check if it is installed.

<br>
<br>

**<span style="font-size:16px"> Checking the Device capability </span>**

Use `nvidia-smi -L` to check the GPU model:

NVIDIA GeForce GTX 1050 Ti

Use `info.cu` to check the detailed information. To compile the code use the following command:

`nvcc -o info info.cu`


Based on the compute capability reported by GPU (6.1), the maximum number of threads per block supported by the GPU is 1024. This is the maximum number of threads per block for all devices with compute capability 6.x.

The maximum number of blocks allowed by the GPU's compute capability depends on the maximum grid size in each dimension. For devices with compute capability 6.x, the maximum grid size in each dimension is 2^31 - 1, which corresponds to the maximum value of a signed 32-bit integer. Therefore, the maximum number of blocks in each dimension of the grid is 2^31 - 1. The total number of blocks in the grid is limited by the product of the maximum grid size in each dimension. For example, the maximum grid size for a 1D grid would be 2^31 - 1, and the maximum grid size for a 2D grid would be (2^31 - 1, 2^31 - 1).

<br>
<br>

**<span style="font-size:16px"> Compiling the Code </span>**

The whole consists of 

1. `read_csr.h`: Reading the sparse matrix using GSL CPU based library 

2. `spmv.h`: Computing A*x with mixed precision using OpenMP for parallelization on CPU

3. `spmv_csr_adaptive_GPU.h`: Adaptive-CSR on GPU using CUDA

4. `spmv_light_GPU.h`: light SpMV on GPU using CUDA

5. `main.cu`: including above mentioned codes. At the top of this code optimal Block Size for the specific problem should be changed. Here it is defined 512: `#define BlockDim 512`. Also at the top this code based on the GPU capability the maximum thread per block must be changed. On my GPU it was 1024: `#define MAX_NUM_THREADS_PER_BLOCK 1024`.


Use the `makefile` to compile the code with `nvcc` compiler and making the `main` object. Run the code by the following command:

`OMP_NUM_THREADS=<n> ./main <SparseMatrix.mtx> <repeats>`

where `n` is the number of available threads for CPU (it can be find on the device using `nproc`), `SparseMatrix.mtx` is the sparse matrix in Market format, and `repeats` is the number of repeats each algorithm will be run to find a better average efficiency. For example, in the following command `8` thread for CPU and matrix `nlpkkt80.mtx` with `100` repeat has been used:

`OMP_NUM_THREADS=8 ./main nlpkkt80.mtx 100`

<br>
<br>

**<span style="font-size:16px"> Code Description </span>**

The function `num_diff_elems()` in the `main.cu` checks the accuracy of the results compared with SpMV on CPU with mixed precision (`smpv.h`) since the accuracy of this code was correct in the last assignment. If the result is no accurate it returns the maximum difference in the result. The code run the methods for both Single and Double Precisions. 

Note that the total number of threads launched is equal to the product of the number of threads per block and the number of blocks in the grid. For instance, in:

`spmv_csr_adaptive_kernel<T><<<(countRowBlocks - 1), BlockDim>>>`

if `(countRowBlocks - 1) = 64` and `BlockDim = 256`, the total number of threads launched would be 64 * 256 = 16384.


