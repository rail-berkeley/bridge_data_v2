from ml_collections import ConfigDict


def get_config(config_string):
    base_real_config = dict(
        batch_size=256,
        num_val_batches=8,
        num_steps=int(2e6),
        log_interval=1000,
        eval_interval=2000,
        save_interval=2000,
        save_dir="<path to save dir>",
        data_path="<path_to_language_conditioned_dataset>",
        resume_path=None,
        seed=42,
    )

    base_data_config = dict(
        shuffle_buffer_size=25000,
        prefetch_num_batches=20,
        augment=True,
        augment_next_obs_goal_differently=False,
        augment_kwargs=dict(
            random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
            random_brightness=[0.2],
            random_contrast=[0.8, 1.2],
            random_saturation=[0.8, 1.2],
            random_hue=[0.1],
            augment_order=[
                "random_resized_crop",
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
            ],
        ),
    )

    # params that need to be specified multiple places
    normalization_type = "normal"

    possible_structures = {
        "lc_ddpm_bc": ConfigDict(
            dict(
                agent="gc_ddpm_bc",
                agent_kwargs=dict(
                    score_network_kwargs=dict(
                        time_dim=32,
                        num_blocks=3,
                        dropout_rate=0.1,
                        hidden_dim=256,
                        use_layer_norm=True,
                    ),
                    language_conditioned=True,
                    early_goal_concat=None,
                    shared_goal_encoder=None,
                    use_proprio=False,
                    beta_schedule="cosine",
                    diffusion_steps=20,
                    action_samples=1,
                    repeat_last_step=0,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    actor_decay_steps=int(2e6),
                ),
                dataset_kwargs=dict(
                    goal_relabeling_strategy="delta_goals",
                    goal_relabeling_kwargs=dict(goal_delta=[0, 20]),
                    #goal_relabeling_strategy="uniform",
                    #goal_relabeling_kwargs=dict(reached_proportion=0.0),
                    load_language=True,
                    skip_unlabeled=True,
                    relabel_actions=False,
                    act_pred_horizon=4,
                    obs_horizon=1,
                    **base_data_config,
                ),
                text_processor="muse_embedding",
                text_processor_kwargs=dict(),
                encoder="resnetv1-34-bridge-film",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                **base_real_config,
            )
        ),
    }

    return possible_structures[config_string]
