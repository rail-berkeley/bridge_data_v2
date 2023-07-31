# Jax BC/RL Implementations for BridgeData V2

This repository provides code for training on [BridgeData V2](https://rail-berkeley.github.io/bridgedata/).

We provide implementations for the following methods (described in the paper):

- Goal-conditioned BC
- Goal-conditioned BC with a diffusion policy [[Chi et al.]](https://diffusion-policy.cs.columbia.edu/)
- Goal-condtioned IQL [[Kostrikov et al.]](https://arxiv.org/abs/2110.06169)
- Goal-conditioned contrastive RL [[Zheng et al., Eysenbach et al.]](https://chongyi-zheng.github.io/stable_contrastive_rl/)

The code for RT-1 can be found [here](https://github.com/google-research/robotics_transformer) and the code for the language-conditioned BC method will be released soon. 

## Data 

The raw dataset (comprised of JPEGs, PNGs, and pkl files) can be downloaded from the [website](https://rail-berkeley.github.io/bridgedata/). For training, the raw data needs to be converted into a TFRecord format that is compatible with the data loader. First, use `scripts/bridgedata_raw_to_numpy.py` to convert the raw data into numpy files. Then, use `scripts/bridgedata_numpy_to_tfrecord.py` to convert the numpy files into TFRecord files. 

## Training

To start training run the command below. Replace `METHOD` with one of `gc_bc`, `gc_ddpm_bc`, `gc_iql`, or `contrastive_rl_td`, and replace `NAME` with a name for the run. 

```
python experiments/bridgedata_offline_gc.py \
    --config experiments/configs/train_config.py:METHOD \
    --bridgedata_config experiments/configs/data_config.py:all \
    --name NAME
```

Training hyperparameters can be modified in `experiments/configs/data_config.py` and data parameters (e.g. subsets to include/exclude) can be modified in `experiments/configs/train_config.py`. 

## Evaluation

First, install our WidowX robot controller stack from this repo (TODO). Then, run the command:

```
python experiments/eval_policy.py \
    --num_timesteps NUM_TIMESTEPS \
    --video_save_path VIDEO_DIR \
    --checkpoint_path CHECKPOINT_PATH \
    --wandb_run_name WANDB_RUN_NAME \
    --blocking
```

The script loads some information about the checkpoint from its corresponding WandB run.

Checkpoints for each of the methods evaluated in the paper are available [here](https://rail.eecs.berkeley.edu/datasets/bridge_release/checkpoints/). Each checkpoint has an associated JSON file with its configuration information. To evaluate these checkpoints with the above script, modify the references to the wandb run configuration to use the dictionary provided in the JSON file instead.

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
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For TPU
```
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```
See the [Jax Github page](https://github.com/google/jax) for more details on installing Jax. 

## Cite

This code is based on [dibyaghosh/jaxrl_m](https://github.com/dibyaghosh/jaxrl_m).

If you use this code and/or BridgeData V2 in your work, please cite the paper with:

```
@article{walke2023bridgedata,
  title={BridgeData V2: A Dataset for Robot Learning at Scale},
  author={Walke, Homer and Black, Kevin and Zhao, Tony and Vuong, Quan and Zheng, Chongyi and Hansen-Estruch, Philippe and He, Andre and Myers, Vivek and Kim, Moo Jin and Du, Max and Lee, Abraham and Fang, Kuan and Finn, Chelsea and Levine, Sergey},
  year={2023}
}
```
