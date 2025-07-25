# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Molformer model configuration"""

from collections import OrderedDict
from typing import Mapping

from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

MOLFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "ibm/MoLFormer-XL-both-10pct": "https://huggingface.co/ibm/MoLFormer-XL-both-10pct/resolve/main/config.json",
}


class MolformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MolformerModel`]. It is used to instantiate an
    Molformer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Molformer
    [ibm/MoLFormer-XL-both-10pct](https://huggingface.co/ibm/MoLFormer-XL-both-10pct) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 2362):
            Vocabulary size of the Molformer model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`MolformerModel`] or [`TFMolformerModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 768):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embedding_dropout_prob (`float`, *optional*, defaults to 0.2):
            The dropout probability for the word embeddings.
        max_position_embeddings (`int`, *optional*, defaults to 202):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 1536).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        linear_attention_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the linear attention layers normalization step.
        num_random_features (`int`, *optional*, defaults to 32):
            Random feature map dimension used in linear attention.
        feature_map_kernel (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the generalized random features. If string,
            `"gelu"`, `"relu"`, `"selu"`, and `"gelu_new"` ar supported.
        deterministic_eval (`bool`, *optional*, defaults to `False`):
            Whether the random features should only be redrawn when training or not. If `True` and `model.training` is
            `False`, linear attention random feature weights will be constant, i.e., deterministic.
        classifier_dropout_prob (`float`, *optional*):
            The dropout probability for the classification head. If `None`, use `hidden_dropout_prob`.
        classifier_skip_connection (`bool`, *optional*, defaults to `True`):
            Whether a skip connection should be made between the layers of the classification head or not.
        pad_token_id (`int`, *optional*, defaults to 2):
            The id of the _padding_ token.

    Example:

    ```python
    >>> from transformers import MolformerModel, MolformerConfig

    >>> # Initializing a Molformer ibm/MoLFormer-XL-both-10pct style configuration
    >>> configuration = MolformerConfig()

    >>> # Initializing a model from the ibm/MoLFormer-XL-both-10pct style configuration
    >>> model = MolformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "molformer"

    def __init__(
        self,
        vocab_size=2362,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=768,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        embedding_dropout_prob=0.2,
        max_position_embeddings=202,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        linear_attention_eps=1e-6,
        num_random_features=32,
        feature_map_kernel="relu",
        deterministic_eval=False,
        classifier_dropout_prob=None,
        classifier_skip_connection=True,
        pad_token_id=2,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.embedding_dropout_prob = embedding_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.linear_attention_eps = linear_attention_eps
        self.num_random_features = num_random_features
        self.feature_map_kernel = feature_map_kernel
        self.deterministic_eval = deterministic_eval
        self.classifier_dropout_prob = classifier_dropout_prob
        self.classifier_skip_connection = classifier_skip_connection


# Copied from transformers.models.roberta.configuration_roberta.RobertaOnnxConfig with Roberta->Molformer
class MolformerOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )
