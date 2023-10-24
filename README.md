# Jax BC/RL Implementations for BridgeData V2

This repository provides code for training on [BridgeData V2](https://rail-berkeley.github.io/bridgedata/).

We provide implementations for the following subset of methods described in the paper:

- Goal-conditioned BC
- Goal-conditioned BC with a diffusion policy 
- Goal-condtioned IQL
- Goal-conditioned contrastive RL 
- Language-conditioned BC

The official implementations and papers for all the methods can be found here:
- [IDQL](https://github.com/philippe-eecs/IDQL) (IQL + diffusion policy) [[Hansen-Estruch et al.](https://github.com/philippe-eecs/IDQL)] and [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) [[Chi et al.](https://diffusion-policy.cs.columbia.edu/)]
- [IQL](https://github.com/ikostrikov/implicit_q_learning) [[Kostrikov et al.](https://arxiv.org/abs/2110.06169)]
- [Contrastive RL](https://chongyi-zheng.github.io/stable_contrastive_rl/) [[Zheng et al.](https://arxiv.org/abs/2306.03346), [Eysenbach et al.](https://arxiv.org/abs/2206.07568)]
- [RT-1](https://github.com/google-research/robotics_transformer) [[Brohan et al.](https://arxiv.org/abs/2212.06817)]
- [ACT](https://github.com/tonyzhaozh/act) [[Zhao et al.](https://arxiv.org/abs/2304.13705)]

Please open a GitHub issue if you encounter problems with this code. 

## Data 
The raw dataset (comprised of JPEGs, PNGs, and pkl files) can be downloaded [here](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/). `demos*.zip` file contains the demonstration data, and `scripted*.zip` contains the data collected with a scripted policy. For training, the raw data needs to be converted into a format that is compatible with a data loader. We offer two options:

- A custom `tf.data` loader. This data loader is implemented in `jaxrl_m/data/bridge_dataset.py` and is used by the training script in this repo. The scripts in the `data_processing` folder convert the raw data into the format required by this data loader. First, use `bridgedata_raw_to_numpy.py` to convert the raw data into NumPy files. Then, use `bridgedata_numpy_to_tfrecord.py` to convert the NumPy files into TFRecord files. 
- A [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/overview) loader. Tensorflow Datasets is a high level wrapper around `tf.data`. We offer a pre-processed TFDS version of the dataset (downsampled to 256x256) in the `tfds` folder here [here](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/). In the TFDS dataset, the trajectories are structured using the [RLDS](https://github.com/google-research/rlds) format. Kevin Black's [DLIMP](https://github.com/kvablack/dlimp) repo provides useful tools for loading and applying transformations to RLDS formatted TFDS datasets (see the `from_rlds` function).

## Training

To start training run the command below. Replace `METHOD` with one of `gc_bc`, `gc_ddpm_bc`, `gc_iql`, or `contrastive_rl_td`, and replace `NAME` with a name for the run. 

```
python experiments/train.py \
    --config experiments/configs/train_config.py:METHOD \
    --bridgedata_config experiments/configs/data_config.py:all \
    --name NAME
```

Training hyperparameters can be modified in `experiments/configs/data_config.py` and data parameters (e.g. subsets to include/exclude) can be modified in `experiments/configs/train_config.py`. 

## Evaluation

First, set up the robot hardware according to our [guide](https://docs.google.com/document/d/1si-6cTElTWTgflwcZRPfgHU7-UwfCUkEztkH3ge5CGc/edit?usp=sharing). Install our WidowX robot controller stack from [this repo](https://github.com/rail-berkeley/bridge_data_robot).

There are two ways to interface a policy with the robot controller: the docker compose service method or the server-client method. Refer to the [bridge_data_robot](https://github.com/rail-berkeley/bridge_data_robot) docs for an explanation of how to set up each method. In general, we recommend the server-client method.

For the server-client method, start the server on the robot. Then run the following commands on the client. You can specify the IP of the remote server via the `--ip` flag. The default IP is `localhost` (i.e the server and client are the same machine). 

```bash
# Specify the path to the downloaded checkpoints directory
export CHECKPOINT_DIR=/path/to/checkpoint_dir

# For GCBC
python experiments/eval.py \
  --checkpoint_weights_path $CHECKPOINT_DIR/checkpoint_300000 \
  --checkpoint_config_path $CHECKPOINT_DIR/gcbc_256_config.json \
  --im_size 256 --goal_type gc --show_image --blocking

# For LCBC
python experiments/eval.py \
  --checkpoint_weights_path $CHECKPOINT_DIR/checkpoint_145000 \
  --checkpoint_config_path $CHECKPOINT_DIR/lcbc_256_config.json \
  --im_size 256 --goal_type lc --show_image --blocking
```

You can also specify an initial position for the end effector with the flag `--initial_eep`. Similarly, use the flag `--goal_eep` to specify the position of the end effector when taking a goal image.

To evaluate image-conditioned or language-conditioned methods with the docker compose service method, run `eval_gc.py` or `eval_lc.py` respectively in the `bridge_data_v2` docker container.

## Provided Checkpoints

Checkpoints for GCBC, LCBC, D-GCBC, GCIQL, and CRL are available [here](https://rail.eecs.berkeley.edu/datasets/bridge_release/checkpoints/). Each checkpoint has an associated JSON file with its configuration information. The name of each checkpoint indicates whether it was trained with 128x128 images or 256x256 images.

We don't currently have a checkpoints for ACT or RT-1 available but may release them soon. 

## Environment

The dependencies for this codebase can be installed in a conda environment:

```bash
conda create -n jaxrl python=3.10
conda activate jaxrl
pip install -e . 
pip install -r requirements.txt
```
For GPU:
```bash
pip install --upgrade "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For TPU
```
pip install --upgrade "jax[tpu]==0.4.13" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```
See the [Jax Github page](https://github.com/google/jax) for more details on installing Jax. 

## Cite

This code is based on [jaxrl_m](https://github.com/dibyaghosh/jaxrl_m) from Dibya Ghosh.

If you use this code and/or BridgeData V2 in your work, please cite the paper with:

```
@inproceedings{walke2023bridgedata,
  title={BridgeData V2: A Dataset for Robot Learning at Scale},
  author={Walke, Homer and Black, Kevin and Lee, Abraham and Kim, Moo Jin and Du, Max and Zheng, Chongyi and Zhao, Tony and Hansen-Estruch, Philippe and Vuong, Quan and He, Andre and Myers, Vivek and Fang, Kuan and Finn, Chelsea and Levine, Sergey},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2023}
}
```
