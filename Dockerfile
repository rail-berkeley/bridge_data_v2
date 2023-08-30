FROM robonet-base:latest

COPY requirements.txt /tmp/requirements.txt
RUN ~/myenv/bin/pip install -r /tmp/requirements.txt
RUN ~/myenv/bin/pip install --upgrade "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# to avoid orbax checkpoint error, downgrade flax 
RUN ~/myenv/bin/pip install flax==0.6.11

ENV PYTHONPATH=${PYTHONPATH}:/home/robonet/code/bridge_data_v2:/home/robonet/code/denoising-diffusion-flax

# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN sudo mkdir -p /usr/local/gcloud \
  && sudo tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && sudo /usr/local/gcloud/google-cloud-sdk/install.sh --quiet

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

# avoid git safe directory errors
RUN git config --global --add safe.directory /home/robonet/code/bridge_data_v2

# activate gcloud credentials (requires them to be mounted at /tmp/gcloud_key.json through docker-compose)
RUN echo "gcloud auth activate-service-account --key-file /tmp/gcloud_key.json" >> ~/.bashrc
ENV GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcloud_key.json

WORKDIR /home/robonet/code/bridge_data_v2
