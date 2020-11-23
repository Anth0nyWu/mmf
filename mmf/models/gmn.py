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
        
        self._build_word_embedding()
        self._init_text_embeddings("text")
        self._init_feature_encoders("image")
        self._init_feature_embeddings("image")

        self._init_combine_layer("image", "text")
        self._init_classifier(self._get_classifier_input_dim())
        self._init_extras()


    def process_feature_embedding(
        self, attr, sample_list, text_embedding_total, extra=None, batch_size_t=None
    ):
        if extra is None:
            extra = []
        feature_embeddings = []
        feature_attentions = []   #useless var
        features = []
        batch_size_t = (
            sample_list.get_batch_size() if batch_size_t is None else batch_size_t
        )

        # Convert list of keys to the actual values
        extra = sample_list.get_fields(extra)

        feature_idx = 0

        # Get all of the features, which are in the form, "image_feature_0"
        # "image_feature_1" ...
        while True:
            feature = getattr(sample_list, f"{attr}_feature_{feature_idx:d}", None)
            if feature is None:
                break
            # print("feature_idx", feature_idx)
            # print("feature1", feature.size()) # torch.Size([4, 100/196, 2048])
            feature_idx += 1
            feature = feature[:batch_size_t]
            # print("feature2", feature.size())
            features.append(feature)

        feature_encoders = getattr(self, attr + "_feature_encoders")

        # print("=====feat encoding=====")
        # print("feat_encoders", feature_encoders)
        '''
        feat_encoders ModuleList(
            (0): ImageFeatureEncoder(
                (module): FinetuneFasterRcnnFpnFc7(
                (lc): Linear(in_features=2048, out_features=2048, bias=True)
                )
            )
            (1): ImageFeatureEncoder(
                (module): Identity()
            )
        )

        '''

        # Each feature should have a separate image feature encoders
        assert len(features) == len(feature_encoders), (
            "Number of feature encoders, {} are not equal "
            "to number of features, {}.".format(len(feature_encoders), len(features))
        )

        # Now, iterate to get final attended image features
        for i, feature in enumerate(features):
            # Get info related to the current feature. info is generally
            # in key of format "image_info_0" for 0th feature
            feature_info = getattr(sample_list, f"{attr}_info_{i:d}", {})
            # print("feature_i: ", i, feature.size())   # torch.Size([4, 100/196, 2048])
            # print("feature_info", feature_info)  # {}
            
            # For Pythia, we need max_features to mask attention
            feature_dim = getattr(feature_info, "max_features", None)
            if feature_dim is not None:
                feature_dim = feature_dim[:batch_size_t]
            # print("feat_dim", feature_dim)  # none

            # Attribute in which encoders are saved, for "image" it
            # will be "image_feature_encoders", other example is
            # "context_feature_encoders"
            encoders_attr = attr + "_feature_encoders"
            feature_encoder = getattr(self, encoders_attr)[i]
            # Encode the features
            encoded_feature = feature_encoder(feature)

            print("encoded_feat:", i, encoded_feature.size()) # torch.Size([64, 100, 2048])
            #feature1--finetune; feat2--identity
            print("=====feat_embedding===== ")

            # Get all of the feature embeddings
            list_attr = attr + "_feature_embeddings_list"
            feature_embedding_models = getattr(self, list_attr)[i]  # image_feature_embeddings_list

            # Forward through these embeddings one by one
            for feature_embedding_model in feature_embedding_models:
                inp = (encoded_feature, text_embedding_total, feature_dim, extra)
                # torch.Size([64, 100, 2048]), [64,2048], none, samplelist()
                # print(feature_embedding_model)
                # print(encoded_feature.size())
                # print(text_embedding_total.size())

                embedding, attention = feature_embedding_model(*inp)
                # memo = self.MNs[i](encoded_feature, text_embedding_total)  # torch.Size([bs, 2048])
                # memo = self.MNs[i](encoded_feature_512, text_embedding_total)  # torch.Size([bs, 2048])
                # print("memo:", memo.size())  
                # print("embedding", embedding.size())

                # embedding_memo_superpos = embedding + memo
                feature_embeddings.append(embedding_memo_superpos)
                # feature_embeddings.append(memo)

        # Concatenate all features embeddings and return along with attention
        feature_embedding_total = torch.cat(feature_embeddings, dim=1)

        # print("feature_embeddings_tot", feature_embedding_total.size()) #[bs*4096]

        return feature_embedding_total, feature_attentions    

    # def get_optimizer_parameters(self, config):
    #     combine_layer = self.image_text_multi_modal_combine_layer
    #     params = [
    #         {"params": self.word_embedding.parameters()},
    #         {"params": self.image_feature_embeddings_list.parameters()},
    #         {"params": self.text_embeddings.parameters()},
    #         {"params": self.MNs.parameters()},
    #         {"params": self.FCs.parameters()},
    #         {"params": combine_layer.parameters()},
    #         {"params": self.classifier.parameters()},
    #         {
    #             "params": self.image_feature_encoders.parameters(),
    #             "lr": (config.optimizer.params.lr * 0.1),
    #         },
    #     ]

    #     return params

    # def forward(self, sample_list):
    #     # torch.cuda.set_device(0)
    #     # print ("=====sample_list=====")
    #     # print(sample_list.fields())
    #     # for key in sample_list.keys():
    #     #     print(key+":")
    #     #     if isinstance(sample_list[key],str) :
    #     #         print("str:"+sample_list[key])
    #     #     else:
    #     #         print(sample_list[key].size())
        
    #     # question 
    #     sample_list.text = self.word_embedding(sample_list.text)
    #     text_embedding_total = self.process_text_embedding(sample_list)
    #     # print("text_embedding", text_embedding_total.size())

    #     # image
    #     image_embedding_total, _ = self.process_feature_embedding(
    #         "image", sample_list, text_embedding_total
    #     )
    #     # print("img_embedding", image_embedding_total.size())
       
    #     if self.inter_model is not None:
    #         image_embedding_total = self.inter_model(image_embedding_total)
        
    #     # print("image_embedding", image_embedding_total.size()) # [batch*4096]
    #     # print("text_embedding:" , image_embedding_total.size()) # [batch*4096]
    #     # print("=====combine layer=====")
        
    #     joint_embedding = self.combine_embeddings(
    #         ["image", "text"], [image_embedding_total, text_embedding_total]
    #     )

    #     # print("joint_embedding:", joint_embedding.size()) # [batch*5000]

    #     model_output = {"scores": self.calculate_logits(joint_embedding)}

    #     # print("model_output:", model_output['scores'].size())
    #     # for name, param in self.MNs.named_parameters():
    #     #     print (name, param)

    #     return model_output

