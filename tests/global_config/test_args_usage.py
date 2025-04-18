"""
plausibility check for the usage of global_config in the megatron codebase
"""
import pytest
import re
from ..common import get_root_directory


@pytest.mark.cpu
def test_GlobalConfig_usage():
    """ "
    checks for code pieces of the pattern "args.*" and verifies that such used arg is defined in GlobalConfig
    """
    from savanna.arguments import GlobalConfig

    declared_all = True
    global_config_attributes = set(GlobalConfig.__dataclass_fields__.keys())

    # we exclude a number of properties (implemented with the @property decorator) or functions that we know exists
    exclude = set(
        [
            "params_dtype",
            "deepspeed_config",
            "get",
            "pop",
            "get_deepspeed_main_args",
            'optimizer["params"]',
            "operator_config[layer_number]",
            "adlr_autoresume_object",
            "update_value",
            "all_config",
            "tensorboard_writer",
            "tokenizer",
            "train_batch_size]",
            "items",
            "configure_distributed_args",
            "build_tokenizer",
            "operator_config[i]",
            "print",
        ]
    )

    # test file by file
    for filename in (get_root_directory() / "megatron").glob("**/*.py"):
        if filename.name in ["text_generation_utils.py", "train_tokenizer.py"]:
            continue

        # load file
        with open(filename, "r") as f:
            file_contents = f.read()

        # find args matches
        matches = list(
            re.findall(r"(?<=args\.).{2,}?(?=[\s\n(){}+-/*;:,=,[,\]])", file_contents)
        )
        if len(matches) == 0:
            continue

        # compare
        for match in matches:
            if match not in global_config_attributes and match not in exclude:
                print(
                    f"(arguments used not found in args): {filename.name}: {match}",
                    flush=True,
                )
                declared_all = False

    assert declared_all, "all arguments used in code defined in GlobalConfig"
