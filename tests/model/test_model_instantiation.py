"""
instantiate models with different configurations as a first possible point of failure
"""

import pytest

import torch
import os
from ..common import distributed_test, model_setup, clear_test_dirs, parametrize, binary

PARAMS_TO_TEST = {
    "pipe_parallel_size,model_parallel_size,world_size": [
        [0, 1, 1],
        [1, 2, 2],
        [0, 2, 2],
    ],
    "no_weight_tying": binary,
    "operator_config": [
        [[["global"], "all"]],
        [[["local"], "all"]],
        [[["sparse_variable"], "all"]],
        [[["sparse_fixed"], "all"]],
    ],
    "scaled_upper_triang_masked_softmax_fusion,bias_gelu_fusion": [
        [True, False],
        [False, True],
    ],
    "fp16,fp32_allreduce": [
        [
            {
                "enabled": True,
                "type": "bfloat16",
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            True,
        ],
        [
            {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            False,
        ],
    ],
}

parameters, names = parametrize(
    PARAMS_TO_TEST, max_tests=int(os.getenv("MAX_TESTCASES", 50)), seed=None
)


@pytest.mark.parametrize("param_dict", parameters, ids=names)
def test_instantiate(param_dict):
    @distributed_test(world_size=param_dict.pop("world_size", 2))
    def wrapper():
        run_test_model_instantiation(param_dict=param_dict)

    wrapper()


OPTIMIZER_PARAMS = {
    "optimizer": [
        {"type": "adam", "params": {"lr": 0.0006}},
        {"type": "onebitadam", "params": {"lr": 0.0006}},
        {"type": "cpu_adam", "params": {"lr": 0.0006}},
        {"type": "cpu_torch_adam", "params": {"lr": 0.0006}},
        {"type": "sm3", "params": {"lr": 0.0006}},
        {"type": "madgrad_wd", "params": {"lr": 0.0006}},
    ]
}
opt_params, opt_name = parametrize(
    OPTIMIZER_PARAMS, max_tests=int(os.getenv("MAX_TESTCASES", 50)), seed=None
)


@pytest.mark.parametrize("param_dict", parameters, ids=names)
def test_instantiate_optimizers(param_dict):
    @distributed_test(world_size=2)
    def wrapper():
        run_test_model_instantiation(param_dict=param_dict)

    wrapper()


def run_test_model_instantiation(yaml_list=None, param_dict=None):
    from deepspeed.runtime.pipe.engine import PipelineEngine, DeepSpeedEngine

    model, optimizer, lr_scheduler, args_loaded = model_setup(yaml_list, param_dict)
    if args_loaded.pipe_parallel_size < 2:
        assert isinstance(model, DeepSpeedEngine), "test model instantiation " + str(
            yaml_list
        )
    else:
        assert isinstance(model, PipelineEngine), "test model instantiation " + str(
            yaml_list
        )
    if torch.distributed.get_world_size() == 1 or torch.distributed.get_rank() == 0:
        clear_test_dirs()
