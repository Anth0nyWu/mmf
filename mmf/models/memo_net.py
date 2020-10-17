# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from mmf.common.registry import registry
from mmf.models.pythia import Pythia
# from mmf.modules.layers import ClassifierLayer
from mmf.modules.layers import MemNNLayer
# from mmf.utils.build import (
#     build_classifier_layer,
#     build_image_encoder,
#     build_text_encoder,
# )

@registry.register_model("memo_net")
class MemoNet(Pythia):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def config_path(cls):
        return "configs/models/memo_net/defaults.yaml"

    def build(self):
        # super().build()
        
        self._build_word_embedding()
        self._init_text_embeddings("text")
        self._init_feature_encoders("image")
        self._init_feature_embeddings("image")
        self._init_MN()

        # self._build_classifier(self._get_classifier_input_dim())
        self._init_extras()


    def _init_MN(self):
        self.MN = MemNNLayer(
            # vocab_size, embd_size, ans_size, max_story_len,
            vocab_size = self.config.mem_nn.vocab_size,
            embd_size = self.config.mem_nn.embd_size,
            ans_size = self.config.mem_nn.vocab_size,
            max_story_len = self.config.mem_nn.max_story_len,
        )

    def process_MN(self, *args):
        image = args[0]  # x
        text = args[1]  # q
        # layer = "MemNN"
        # return getattr(self, layer)(image,text)
        return self.MN(image,text)

    def get_optimizer_parameters(self, config):
        combine_layer = self.MN
        params = [
            {"params": self.word_embedding.parameters()},
            {"params": self.image_feature_embeddings_list.parameters()},
            {"params": self.text_embeddings.parameters()},
            {"params": self.MN.parameters()},
            # {"params": combine_layer.parameters()},
            # {"params": self.classifier.parameters()},
            {
                "params": self.image_feature_encoders.parameters(),
                "lr": (config.optimizer.params.lr * 0.1),
            },
        ]

        return params


    def forward(self, sample_list):

        # question 
        sample_list.text = self.word_embedding(sample_list.text)
        text_embedding_total = self.process_text_embedding(sample_list)

        # image
        image_embedding_total, _ = self.process_feature_embedding(
            "image", sample_list, text_embedding_total
        )

        # print(image_embedding_total.size())
        # print(text_embedding_total.size())

       
        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)
        
        # pythia
        # joint_embedding = self.combine_embeddings(
        #     ["image", "text"], [image_embedding_total, text_embedding_total]
        # )

        # model_output = {"scores": self.calculate_logits(joint_embedding)}

        # return model_output

        # memory

        image_embedding_total = image_embedding_total.unsqueeze(2)
        model_output = {"scores": self.process_MN(image_embedding_total, text_embedding_total)}
        # print(model_output['scores'].size())

        return model_output

