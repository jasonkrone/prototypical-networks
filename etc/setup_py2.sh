# download anaconda for python2
curl -O https://repo.continuum.io/archive/Anaconda2-4.4.0-Linux-x86_64.sh
# run installer 
bash Anaconda2-4.4.0-Linux-x86_64.sh
# activate installation 
source ~/.bashrc
# verify install
conda list

# ^ taken from https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04

# create tf env
conda create -n tensorflow python=2

# to activate: source activate tensorflow
# to deactivate: source deactivate tensorflow

# activate env 
source activate tensorflow
# install tensorflow
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp27-none-linux_x86_64.whl

# ^ taken from https://www.tensorflow.org/install/install_linux#InstallingAnaconda

# set environment variables
cd /home/jason/anaconda2/envs/tensorflow
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh
# edit activate
echo "#!/bin/sh" > ./etc/conda/activate.d/env_vars.sh
echo "export LD_LIBRARY_PATH='$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64'" > ./etc/conda/activate.d/env_vars.sh
echo "export CUDA_HOME=/usr/local/cuda-8.0" > ./etc/conda/activate.d/env_vars.sh
# edit deactivate
echo "#!/bin/sh" > ./etc/conda/deactivate.d/env_vars.sh
echo "unset LD_LIBRARY_PATH" > ./etc/conda/deactivate.d/env_vars.sh
echo "unset CUDA_HOME" > ./etc/conda/deactivate.d/env_vars.sh

# ^ taken from https://stackoverflow.com/questions/41991101/importerror-libcudnn-when-running-a-tensorflow-program
# and https://conda.io/docs/user-guide/tasks/manage-environments.html#saving-environment-variables
