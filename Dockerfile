FROM robonet-base:latest

COPY requirements.txt /tmp/requirements.txt
RUN ~/myenv/bin/pip install -r /tmp/requirements.txt
RUN ~/myenv/bin/pip install --upgrade "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# to avoid orbax checkpoint error, downgrade flax 
RUN ~/myenv/bin/pip install flax==0.6.11

# avoid git safe directory errors
RUN git config --global --add safe.directory /home/robonet/code/bridge_data_v2

WORKDIR /home/robonet/code/bridge_data_v2
