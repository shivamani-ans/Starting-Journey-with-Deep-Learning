# Issues Encountered and Resolutions

## Issue: ImportError: libcudnn.so.7: cannot open shared object file: No such file or directory (12/03/2019)
    **Resolution:
        create env variable export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH