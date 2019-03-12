# Issues Encountered and Resolutions

## Issue: ImportError: libcudnn.so.7: cannot open shared object file: No such file or directory (12/03/2019)
### Resolution:
> create env variable export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


## Issue: Error in loading audiofile pkg-audiofile (12/03/2019)
### Resolution:
**url: https://linuxize.com/post/how-to-install-ffmpeg-on-centos-7/**
> sudo yum install epel-release

> sudo rpm -v --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro

> sudo rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm

> sudo yum install ffmpeg ffmpeg-devel

> ffmpeg -version