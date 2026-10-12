@echo off
rem A Gaussian splat of an object scan on hekate: train_splat.py with what gsplat needs to build its
rem CUDA kernels the first time (into %%USERPROFILE%%\AppData\Local\torch_extensions, ~10 min, then
rem cached): the VS 2022 C++ Build Tools' environment and the CUDA 12.8 toolkit, for the 5090 only.
rem MSVC 14.44 is newer than CUDA 12.8 lists, hence -allow-unsupported-compiler; and windows.h's
rem rpcndr.h makes "small" a macro (char), which breaks torch's own CUDACachingAllocator.h ("bool
rem small"), hence WIN32_LEAN_AND_MEAN for nvcc and cl (CL) both.
rem     splat\splat.bat datasets\scans\NAME [--steps 7000]          (from train\)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "NVCC_APPEND_FLAGS=-allow-unsupported-compiler -DWIN32_LEAN_AND_MEAN -DNOMINMAX"
set "CL=/DWIN32_LEAN_AND_MEAN /DNOMINMAX"
set "PYTHONUTF8=1"
"%~dp0..\.venv\Scripts\python.exe" "%~dp0train_splat.py" %*
