# Jax BC/RL Implementations for BridgeData V2

This repository provides code for training on [BridgeData V2](https://rail-berkeley.github.io/bridgedata/).

We provide implementations for the following subset of methods described in the paper:

- Goal-conditioned BC
- Goal-conditioned BC with a diffusion policy 
- Goal-condtioned IQL
- Goal-conditioned contrastive RL 

The code for the language-conditioned BC method may be released soon.

The official implementations and papers for all the methods can be found here:
- [IQDL](https://github.com/philippe-eecs/IDQL) (IQL + diffusion policy) [[Hansen-Estruch et al.](https://github.com/philippe-eecs/IDQL)] and [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) [[Chi et al.](https://diffusion-policy.cs.columbia.edu/)]
- [IQL](https://github.com/ikostrikov/implicit_q_learning) [[Kostrikov et al.](https://arxiv.org/abs/2110.06169)]
- [Contrastive RL](https://chongyi-zheng.github.io/stable_contrastive_rl/) [[Zheng et al.](https://arxiv.org/abs/2306.03346), [Eysenbach et al.](https://arxiv.org/abs/2206.07568)]
- [RT-1](https://github.com/google-research/robotics_transformer) [[Brohan et al.](https://arxiv.org/abs/2212.06817)]
- [ACT](https://github.com/tonyzhaozh/act) [[Zhao et al.](https://arxiv.org/abs/2304.13705)]

Please open a GitHub issue if you encounter problems with this code. 

## Data 

The raw dataset (comprised of JPEGs, PNGs, and pkl files) can be downloaded from the [website](https://rail-berkeley.github.io/bridgedata/). For training, the raw data needs to be converted into a TFRecord format that is compatible with the data loader. First, use `data_processing/bridgedata_raw_to_numpy.py` to convert the raw data into numpy files. Then, use `data_processing/bridgedata_numpy_to_tfrecord.py` to convert the numpy files into TFRecord files. 

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

First, set up the robot hardware according to our [guide](https://docs.google.com/document/d/1si-6cTElTWTgflwcZRPfgHU7-UwfCUkEztkH3ge5CGc/edit?usp=sharing). Install our WidowX robot controller stack from [this repo](https://github.com/rail-berkeley/bridge_data_robot). Then, run the command:

```
python experiments/eval.py \
    --num_timesteps NUM_TIMESTEPS \
    --video_save_path VIDEO_DIR \
    --checkpoint_path CHECKPOINT_PATH \
    --wandb_run_name WANDB_RUN_NAME \
    --blocking
```

The script loads some information about the checkpoint from its corresponding WandB run.

## Provided Checkpoints

Checkpoints for GCBC, D-GCBC, GCIQL, CRL, and RT-1 are available [here](https://rail.eecs.berkeley.edu/datasets/bridge_release/checkpoints/). Each checkpoint (except RT-1) has an associated JSON file with its configuration information. To evaluate these checkpoints with the above evaluation script, modify the references to the wandb run configuration to use the dictionary provided in the JSON file instead.

An evaluation script for the RT-1 checkpoint is available in this separate repo (TODO).

We don't currently have checkpoints for ACT or LCBC available but may release them soon. 

## Environment

The dependencies for this codebase can be installed in a conda environment:

```
conda create -n jaxrl python=3.10
conda activate jaxrl
pip install -e . 
pip install -r requirements.txt
```
For GPU:
```
pip install --upgrade "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For TPU
```
pip install --upgrade "jax[tpu]==0.4.13" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```
See the [Jax Github page](https://github.com/google/jax) for more details on installing Jax. 

## Cite

This code is based on [dibyaghosh/jaxrl_m](https://github.com/dibyaghosh/jaxrl_m).

If you use this code and/or BridgeData V2 in your work, please cite the paper with:

```
@inproceedings{walke2023bridgedata,
  title={BridgeData V2: A Dataset for Robot Learning at Scale},
  author={Walke, Homer and Black, Kevin and Lee, Abraham and Kim, Moo Jin and Du, Max and Zheng, Chongyi and Zhao, Tony and Hansen-Estruch, Philippe and Vuong, Quan and He, Andre and Myers, Vivek and Fang, Kuan and Finn, Chelsea and Levine, Sergey},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2023}
}
```
