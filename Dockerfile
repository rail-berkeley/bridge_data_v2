FROM robonet-base:latest

RUN ~/myenv/bin/pip install tensorflow jax[cuda11_cudnn82] flax distrax ml_collections h5py wandb einops \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN sudo mkdir -p /usr/local/gcloud \
  && sudo tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && sudo /usr/local/gcloud/google-cloud-sdk/install.sh --quiet

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

# avoid git safe directory errors
RUN git config --global --add safe.directory /home/robonet/code/BridgeData-V2

# activate gcloud credentials (requires them to be mounted at /tmp/gcloud_key.json through docker-compose)
RUN echo "gcloud auth activate-service-account --key-file /tmp/gcloud_key.json" >> ~/.bashrc
ENV GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcloud_key.json

WORKDIR /home/robonet/code/BridgeData-V2