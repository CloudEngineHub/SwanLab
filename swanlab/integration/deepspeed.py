"""
Docs: <WAITIGNG UPDATE>

For adaptation to the huggingface accelerate. You can used SwanLab as your tracker, experiment logs can be uploaded to
SwanLab or viewed using the local version of SwanLab. Detailed of used swanlab in accelerate train scripts are as follows:
------train.py in accelerate------
...
from swanlab.integration.accelerate import SwanLabTracker
...
tracker = SwanLabTracker("some_run_name")
accelerator = Accelerator(log_with=tracker)
...
---------------------------------
These also can be mixed with existing trackers, including with "all":
------train.py in accelerate------
...
from swanlab.integration.accelerate import SwanLabTracker
...
tracker = SwanLabTracker("some_run_name")
accelerator = Accelerator(log_with=[tracker, "all"])
...
---------------------------------
"""

import os
from abc import ABC, abstractmethod

import swanlab


try:
    from deepspeed.monitor.monitor import Monitor
    import deepspeed.comm as dist
    from deepspeed.runtime.config import DeepSpeedConfig
except ImportError:
    raise RuntimeError(
        "This contrib module requires deepspeed to be installed. \n"
        "Please install it with command: \n\t pip install accelerate\n"
        "Find more detail at: https://www.deepspeed.ai/getting-started/"
    )


class SwanlabMonitor(Monitor):

    def __init__(self, monitor_config):
        swanlab_config = monitor_config.swanlab_config
        super().__init__(swanlab_config)

        self.enabled = swanlab_config.enabled
        self.workspace = swanlab_config.workspace
        self.project = swanlab_config.project
        self.experiment_name = swanlab_config.experiment_name

        if self.enabled and dist.get_rank() == 0:
            swanlab.init(project=self.project, workspace=self.workspace, experiment_name=self.experiment_name)

    def update_config(self, args=None, config=None, config_params=None):
        # Set config using config_params for deepspeed backwards compat
        if config is None and config_params is not None:
            config = config_params

        if isinstance(config, DeepSpeedConfig):
            swanlab.config.update(dp_config._param_dict)

        if hasattr(args, "deepscale_config") and args.deepscale_config is not None:
            args.deepspeed_config = args.deepscale_config
            args.deepscale_config = None
        if hasattr(args, "deepspeed_config") and args.deepspeed_config is not None:
            assert (
                config is None
            ), "Not sure how to proceed, we were given deepspeed configs in the deepspeed arguments and deepspeed.initialize() function call"
            config = args.deepspeed_config

        if config is not None:
            dp_config = DeepSpeedConfig(config)
            swanlab.config.update(dp_config._param_dict)

    def write_events(self, event_list):
        if self.enabled and dist.get_rank() == 0:
            for event in event_list:
                label = event[0]
                value = event[1]
                step = event[2]
                swanlab.log({label: value}, step=step)
