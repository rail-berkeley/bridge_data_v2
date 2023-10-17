import ml_collections

ACT_MEAN = [
    2.9842544e-04,
    -2.6099570e-04,
    -1.5863389e-04,
    5.8916201e-05,
    -4.4560504e-05,
    8.2349771e-04,
    9.4075650e-02,
]

ACT_STD = [
    0.27278143,
    0.23548537,
    0.2196189,
    0.15881406,
    0.17537235,
    0.27875036,
    1.0049515,
]

PROPRIO_MEAN = [ # We don't actually use proprio so we're using dummy values for this
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]

PROPRIO_STD = [ # We don't actually use proprio so we're using dummy values for this
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]

ACTION_PROPRIO_METADATA = {
    "action": {
        "mean": ACT_MEAN,
        "std": ACT_STD,
        # TODO compute these
        "min": ACT_MEAN,
        "max": ACT_STD
    },
    # TODO compute these
    "proprio": {
        "mean": PROPRIO_MEAN,
        "std": PROPRIO_STD,
        "min": PROPRIO_MEAN,
        "max": PROPRIO_STD
    }
}


def get_config(config_string):
    possible_structures = {
        "all": ml_collections.ConfigDict(
            {
                "include": [
                    [
                        "training/A/?*",
                        "training/B/?*",
                        "training/C/?*"
                    ],
                    [
                        "validation/D/?*",
                    ]
                ],
                "exclude": [],
                "sample_weights": None,
                "action_proprio_metadata": ACTION_PROPRIO_METADATA
            }
        ),
    }
    return possible_structures[config_string]
