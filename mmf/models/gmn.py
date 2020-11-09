# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from mmf.common.registry import registry
from mmf.models.pythia import Pythia

from mmf.modules.layers import MemNNLayer
from torch import nn


@registry.register_model("gmn")
class GraphMemoNet(Pythia):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def config_path(cls):
        return "configs/models/gmn/defaults.yaml"

    def build(self):
        super().build()
        
        # self._build_word_embedding()
        # self._init_text_embeddings("text")
        # self._init_feature_encoders("image")
        # self._init_feature_embeddings("image")
        # self._init_MN("image")
        # self._init_combine_layer("image", "text")
        # self._init_classifier(self._get_classifier_input_dim())
        # self._init_extras()