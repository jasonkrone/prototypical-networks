# install cuda 9.0 for ubuntu 16.04
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-key-adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda-libraries-9-0

# download anaconda for python3
curl -O https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
# run installer 
bash Anaconda3-5.1.0-Linux-x86_64.sh
# activate installation 
source ~/.bashrc
# verify install
conda list
# ^ taken from https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04

# create tf env
conda create -n tensorflow python=3.5.2

# to activate: source activate tensorflow
# to deactivate: source deactivate tensorflow

# activate env 
source activate tensorflow

# install tensorflow
# taken from https://www.tensorflow.org/install/install_linux#InstallingAnaconda
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.5.0-cp35-cp35m-linux_x86_64.whl

# set environment variables
cd /home/jason/anaconda3/envs/tensorflow
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh
# edit activate
echo "#!/bin/sh" >> ./etc/conda/activate.d/env_vars.sh
echo "export LD_LIBRARY_PATH='$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64:/usr/local/cuda-9.0/extras/CUPTI/lib64'" >> ./etc/conda/activate.d/env_vars.sh
echo "export CUDA_HOME=/usr/local/cuda-9.0" >> ./etc/conda/activate.d/env_vars.sh
# edit deactivate
echo "#!/bin/sh" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset LD_LIBRARY_PATH" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset CUDA_HOME" >> ./etc/conda/deactivate.d/env_vars.sh

# ^ taken from https://stackoverflow.com/questions/41991101/importerror-libcudnn-when-running-a-tensorflow-program
# and https://conda.io/docs/user-guide/tasks/manage-environments.html#saving-environment-variables

# TODO: install cuDNN
# goto nvidia-website and download installer
# move files to location listed on webpage

