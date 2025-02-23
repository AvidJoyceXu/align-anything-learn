# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
#
# This code is inspired by the HuggingFace's Transformers library.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import torch
import torch.utils.checkpoint
from transformers.models.llava_next.modeling_llava_next import LlavaNextForConditionalGeneration


class AccustomedLlavaNextModel(LlavaNextForConditionalGeneration):

    @property
    def processor_available(self):
        return True

    def infer_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return the dict used for model inference"""
        return {
            'input_ids': batch['input_ids'],
            'attention_mask': batch.get('attention_mask'),
            'pixel_values': batch.get('pixel_values'),
            'labels': batch.get('labels'),
            'image_sizes': batch.get('image_sizes'),
        }
