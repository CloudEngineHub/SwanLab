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
from typing import Any, Dict, List, Optional, Union

import swanlab


try:
    import deepspeed
except ImportError:
    raise RuntimeError(
        "This contrib module requires deepspeed to be installed. \n"
        "Please install it with command: \n\t pip install accelerate\n"
        "see more detail from: https://www.deepspeed.ai/getting-started/"
    )
