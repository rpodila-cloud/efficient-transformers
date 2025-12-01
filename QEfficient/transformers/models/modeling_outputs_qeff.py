# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


@dataclass
class QEffBaseModelOutputWithPast(BaseModelOutputWithPast):
    prefill_queries: Optional[torch.FloatTensor] = None


@dataclass
class QEffCausalLMOutputWithPast(CausalLMOutputWithPast):
    prefill_queries: Optional[torch.FloatTensor] = None
