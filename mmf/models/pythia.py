# Copyright (c) Facebook, Inc. and its affiliates.
import copy

import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.embeddings import (
    ImageFeatureEmbedding,
    MultiHeadImageFeatureEmbedding,
    PreExtractedEmbedding,
    TextEmbedding,
)
from mmf.modules.encoders import ImageFeatureEncoder
from mmf.modules.layers import ClassifierLayer, ModalCombineLayer
from torch import nn


@registry.register_model("pythia")
class Pythia(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._global_config = registry.get("config")
        self._datasets = self._global_config.datasets.split(",")

    @classmethod
    def config_path(cls):
        return "configs/models/pythia/defaults.yaml"

    @classmethod
    def format_state_key(cls, key):
        return key.replace("fa_history", "fa_context")

    def build(self):
        self._build_word_embedding()
        self._init_text_embeddings("text")
        self._init_feature_encoders("image")
        self._init_feature_embeddings("image")
        self._init_combine_layer("image", "text")
        self._init_classifier(self._get_classifier_input_dim())
        self._init_extras()

    def _build_word_embedding(self):
        assert len(self._datasets) > 0
        text_processor = registry.get(self._datasets[0] + "_text_processor")
        vocab = text_processor.vocab
        self.word_embedding = vocab.get_embedding(torch.nn.Embedding, embedding_dim=300)

    def _init_text_embeddings(self, attr="text"):
        if "embeddings" not in attr:
            attr += "_embeddings"

        text_embeddings = []
        text_embeddings_list_config = self.config[attr]

        embeddings_out_dim = 0

        for text_embedding in text_embeddings_list_config:
            embedding_type = text_embedding.type
            embedding_kwargs = copy.deepcopy(text_embedding.params)

            self._update_text_embedding_args(embedding_kwargs)

            embedding = TextEmbedding(embedding_type, **embedding_kwargs)

            text_embeddings.append(embedding)
            embeddings_out_dim += embedding.text_out_dim

        setattr(self, attr + "_out_dim", embeddings_out_dim)
        setattr(self, attr, nn.ModuleList(text_embeddings))

    def _update_text_embedding_args(self, args):
        # Add model_data_dir to kwargs
        args.model_data_dir = self.config.model_data_dir

    def _init_feature_encoders(self, attr):
        feat_encoders = []
        feat_encoders_list_config = self.config[attr + "_feature_encodings"]
        feature_dim = self.config[attr + "_feature_dim"]
        setattr(self, attr + "_feature_dim", feature_dim)

        print("feat_encoders_list_config", feat_encoders_list_config)

        for feat_encoder in feat_encoders_list_config:
            encoder_type = feat_encoder.type
            encoder_kwargs = copy.deepcopy(feat_encoder.params)
            encoder_kwargs.model_data_dir = self.config.model_data_dir

            feat_model = ImageFeatureEncoder(
                encoder_type, feature_dim, **encoder_kwargs
            )

            feat_encoders.append(feat_model)
            setattr(self, attr + "_feature_dim", feat_model.out_dim)

        setattr(self, attr + "_feature_encoders", nn.ModuleList(feat_encoders))

    def _init_feature_embeddings(self, attr):
        feature_embeddings_list = []
        num_feature_feat = len(getattr(self.config, f"{attr}_feature_encodings"))

        # print("num_feature_feat", num_feature_feat)
        # print(getattr(self.config, f"{attr}_feature_encodings"))
        # [{'type': 'finetune_faster_rcnn_fpn_fc7', 'params': {'bias_file': 'models/detectron.defaults/fc7_b.pkl', 'weights_file': 'models/detectron.defaults/fc7_w.pkl', 'model_data_dir': '/media/ubuntu/MyDisk/data_mmf/vg'}}, {'type': 'default', 'params': {'model_data_dir': '/media/ubuntu/MyDisk/data_mmf/vg'}}]

        self.feature_embeddings_out_dim = 0

        for _ in range(num_feature_feat): #2
            feature_embeddings = []
            feature_attn_model_list = self.config[attr + "_feature_embeddings"]

            # print ("feature_attn_model_list", feature_attn_model_list)
            # # [{'modal_combine': {'type': 'non_linear_element_multiply', 'params': {'dropout': 0, 'hidden_dim': 5000}}, 'normalization': 'softmax', 'transform': {'type': 'linear', 'params': {'out_dim': 1}}}]
            # print("attr_feat_dim", getattr(self, attr + "_feature_dim")) #2048
            # print("text_embeddings_out_dim", self.text_embeddings_out_dim) # 2048

            for feature_attn_model_params in feature_attn_model_list:
                feature_embedding = ImageFeatureEmbedding(
                    getattr(self, attr + "_feature_dim"),  #2048
                    self.text_embeddings_out_dim,
                    **feature_attn_model_params,
                )
                # print ("feature_embedding", feature_embedding) #a embedding model
                feature_embeddings.append(feature_embedding)
                self.feature_embeddings_out_dim += feature_embedding.out_dim

            feature_embeddings = nn.ModuleList(feature_embeddings)
            feature_embeddings_list.append(feature_embeddings)

        self.feature_embeddings_out_dim *= getattr(self, attr + "_feature_dim")

        setattr(
            self, attr + "_feature_embeddings_out_dim", self.feature_embeddings_out_dim
        )
        del self.feature_embeddings_out_dim
        setattr(
            self,
            attr + "_feature_embeddings_list",
            nn.ModuleList(feature_embeddings_list),
        )

    def _get_embeddings_attr(self, attr):
        embedding_attr1 = attr
        if hasattr(self, attr + "_embeddings_out_dim"):
            embedding_attr1 = attr + "_embeddings_out_dim"
        else:
            embedding_attr1 = attr + "_feature_embeddings_out_dim"

        return embedding_attr1

    def _init_combine_layer(self, attr1, attr2):
        config_attr = attr1 + "_" + attr2 + "_modal_combine"

        multi_modal_combine_layer = ModalCombineLayer(
            self.config[config_attr].type,
            getattr(self, self._get_embeddings_attr(attr1)),
            getattr(self, self._get_embeddings_attr(attr2)),
            **self.config[config_attr].params,
        )

        setattr(
            self,
            attr1 + "_" + attr2 + "_multi_modal_combine_layer",
            multi_modal_combine_layer,
        )

    def _init_classifier(self, combined_embedding_dim):
        # TODO: Later support multihead
        num_choices = registry.get(self._datasets[0] + "_num_final_outputs")
        # print (num_choices)
        self.classifier = ClassifierLayer(
            self.config.classifier.type,
            in_dim=combined_embedding_dim,
            out_dim=num_choices,
            **self.config.classifier.params,
        )

    def _init_extras(self):
        self.inter_model = None

    def get_optimizer_parameters(self, config):
        combine_layer = self.image_text_multi_modal_combine_layer
        params = [
            {"params": self.word_embedding.parameters()},
            {"params": self.image_feature_embeddings_list.parameters()},
            {"params": self.text_embeddings.parameters()},
            {"params": combine_layer.parameters()},
            {"params": self.classifier.parameters()},
            {
                "params": self.image_feature_encoders.parameters(),
                "lr": (config.optimizer.params.lr * 0.1),
            },
        ]

        return params

    def _get_classifier_input_dim(self):
        return self.image_text_multi_modal_combine_layer.out_dim

    def process_text_embedding(
        self, sample_list, embedding_attr="text_embeddings", info=None
    ):

        # print("=====text embedding=====")

        text_embeddings = []

        # Get "text" attribute in case of "text_embeddings" case
        # and "context" attribute in case of "context_embeddings"
        texts = getattr(sample_list, embedding_attr.split("_")[0])

        # print("text", texts.size())  # bs*20*300

        # Get embedding models
        text_embedding_models = getattr(self, embedding_attr)

        for text_embedding_model in text_embedding_models:
            # print("text_model", text_embedding_model)
            '''
            text_model TextEmbedding(
                (module): AttentionTextEmbedding(
                    (recurrent_unit): LSTM(300, 1024, batch_first=True)
                    (dropout): Dropout(p=0, inplace=False)
                    (conv1): Conv1d(1024, 512, kernel_size=(1,), stride=(1,))
                    (conv2): Conv1d(512, 2, kernel_size=(1,), stride=(1,))
                    (relu): ReLU()
                )
            )
            '''
            # TODO: Move this logic inside
            if isinstance(text_embedding_model, PreExtractedEmbedding):
                embedding = text_embedding_model(sample_list.question_id)
            else:
                embedding = text_embedding_model(texts)

            # print("text_embedding: ", embedding.size()) # torch.Size([4(bs), 2048])
            text_embeddings.append(embedding)

        text_embeddding_total = torch.cat(text_embeddings, dim=1)
        # print("text_embedding_tot: ", text_embeddding_total.size()) # torch.Size([4(bs), 2048])

        return text_embeddding_total

    def process_feature_embedding(
        self, attr, sample_list, text_embedding_total, extra=None, batch_size_t=None
    ):
        if extra is None:
            extra = []
        feature_embeddings = []
        feature_attentions = []
        features = []
        batch_size_t = (
            sample_list.get_batch_size() if batch_size_t is None else batch_size_t
        )

        # Convert list of keys to the actual values
        extra = sample_list.get_fields(extra)

        feature_idx = 0
        # print("=====feature encoder=====")

        # Get all of the features, which are in the form, "image_feature_0"
        # "image_feature_1" ...
        while True:
            feature = getattr(sample_list, f"{attr}_feature_{feature_idx:d}", None)
            if feature is None:
                break
            feature_idx += 1
            feature = feature[:batch_size_t]
            features.append(feature)

        feature_encoders = getattr(self, attr + "_feature_encoders")
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
            # print("feature_i: ", i, feature.size())
            # print("feature_info", feature_info)
            
            # For Pythia, we need max_features to mask attention
            feature_dim = getattr(feature_info, "max_features", None)
            if feature_dim is not None:
                feature_dim = feature_dim[:batch_size_t]

            # print("feat_dim", feature_dim)  # none

            # Attribute in which encoders are saved, for "image" it
            # will be "image_feature_encoders", other example is
            # "context_feature_encoders"
            encoders_attr = attr + "_feature_encoders"
            feature_encoder = getattr(self, encoders_attr)[i]   #repeat of line 271
            # print("feature_encoder", feature_encoder)
            '''
            feature_i:  0 torch.Size([64, 100, 2048])
            feature_info {}
            feature_encoder ImageFeatureEncoder(
            (module): FinetuneFasterRcnnFpnFc7(
                (lc): Linear(in_features=2048, out_features=2048, bias=True)
            )
            )
            feature_i:  1 torch.Size([64, 196, 2048])
            feature_info {}
            feature_encoder ImageFeatureEncoder(
            (module): Identity()
            )
            '''

            # Encode the features
            encoded_feature = feature_encoder(feature)

            # print("encoded_feat:", encoded_feature.size()) # torch.Size([64, 100, 2048])
            # print("=====feat_embedding", i, "===== ")

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
                feature_embeddings.append(embedding)
                feature_attentions.append(attention.squeeze(-1))

                # print("feature_embeddings_&_attns")
                # print(embedding.size())  # torch.Size([64, 2048])
                # print(attention.size()) # torch.Size([64, 196, 1])

        # Concatenate all features embeddings and return along with attention
        feature_embedding_total = torch.cat(feature_embeddings, dim=1)

        # print("feature_embeddings_tot")
        # print(feature_embedding_total.size())

        return feature_embedding_total, feature_attentions

    def combine_embeddings(self, *args):
        feature_names = args[0]
        feature_embeddings = args[1]

        layer = "_".join(feature_names) + "_multi_modal_combine_layer"
        return getattr(self, layer)(*feature_embeddings)

    def calculate_logits(self, joint_embedding, **kwargs):
        return self.classifier(joint_embedding)

    def forward(self, sample_list):
        print ("=====sample_list=====")
        print(sample_list.fields())
        for key in sample_list.keys():
            print(key+":")
            if isinstance(sample_list[key],str) :
                print("str:", sample_list[key])
            elif isinstance(sample_list[key],dict) :
                for key2 in sample_list[key].keys():
                    print(key2+":")
                    print(type(sample_list[key][key2]))

        sample_list.text = self.word_embedding(sample_list.text)
        text_embedding_total = self.process_text_embedding(sample_list)

        image_embedding_total, _ = self.process_feature_embedding(
            "image", sample_list, text_embedding_total
        )

        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)

        # print("image_embedding", image_embedding_total.size())  # [batch*4096]
        # print("text_embedding:" , image_embedding_total.size())  # [batch*4096]
        # print("=====combine layer=====")

        joint_embedding = self.combine_embeddings(
            ["image", "text"], [image_embedding_total, text_embedding_total]
        )

        # print("joint_embedding:", joint_embedding.size()) # [batch*5000]

        model_output = {"scores": self.calculate_logits(joint_embedding)}

        # print("model_output:", model_output['scores'].size()) # [64, 3129]
        return model_output


# TODO: Update
@registry.register_model("pythia_question_only")
class PythiaQuestionOnly(Pythia):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, sample_list):
        text_embedding_total = self.process_text_embedding(sample_list)
        text_embedding_total = text_embedding_total.new_zeros(
            text_embedding_total.size()
        )

        fa_txt = self.image_text_multi_modal_combine_layer.module.fa_txt
        dropout = self.image_text_multi_modal_combine_layer.module.dropout

        joint_embedding = dropout(fa_txt(text_embedding_total))

        linear_text = self.classifier.module.linear_text
        f_o_text = self.classifier.module.f_o_text
        scores = linear_text(f_o_text(joint_embedding))

        model_output = {"scores": scores}

        return model_output


# TODO: Update
@registry.register_model("pythia_image_only")
class PythiaImageOnly(Pythia):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, sample_list):
        text_embedding_total = self.process_text_embedding(sample_list)
        text_embedding_total = text_embedding_total.new_zeros(
            text_embedding_total.size()
        )

        image_embedding_total, _ = self.process_feature_embedding(
            "image", sample_list, text_embedding_total
        )

        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)

        fa_image = self.image_text_multi_modal_combine_layer.module.fa_image
        dropout = self.image_text_multi_modal_combine_layer.module.dropout

        joint_embedding = dropout(fa_image(image_embedding_total))

        model_output = {"scores": self.calculate_logits(joint_embedding)}

        return model_output


@registry.register_model("multihead")
class PythiaMultiHead(Pythia):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def config_path(cls):
        return None

    def build(self):
        self._build_word_embedding()
        self._init_text_embeddings("text")
        self._init_feature_encoders("image")
        self._init_feature_projectors("image")
        self._init_feature_embeddings("image")
        self._init_combine_layer("image", "text")
        self._init_classifier(self._get_classifier_input_dim())
        self._init_extras()

    def _init_feature_projectors(self, attr):
        feature_projectors = []
        feat_encoders_list_config = self.config[attr + "_feature_projections"]
        feat_dim = getattr(self, attr + "_feature_dim")

        for feat_encoder in feat_encoders_list_config:
            encoder_type = feat_encoder.type
            encoder_kwargs = feat_encoder.params

            feat_model = ImageFeatureEncoder(encoder_type, feat_dim, **encoder_kwargs)

            feature_projectors.append(feat_model)
            setattr(self, attr + "_feature_dim", feat_model.out_dim)

        setattr(self, attr + "_feature_projectors", nn.ModuleList(feature_projectors))

    def _init_feature_embeddings(self, attr):
        feature_embeddings_list = []
        num_feature_feat = len(getattr(self.config, f"{attr}_feature_encodings"))

        self.feature_embeddings_out_dim = 0

        for _ in range(num_feature_feat):
            feature_embeddings = []
            feature_attn_model_list = self.config[attr + "_feature_embeddings"]

            for feature_attn_model_params in feature_attn_model_list:
                feature_embedding = MultiHeadImageFeatureEmbedding(
                    getattr(self, attr + "_feature_dim"),
                    self.text_embeddings_out_dim,
                    **feature_attn_model_params,
                )
                feature_embeddings.append(feature_embedding)
                self.feature_embeddings_out_dim += feature_embedding.out_dim

            feature_embeddings = nn.ModuleList(feature_embeddings)
            feature_embeddings_list.append(feature_embeddings)

        setattr(
            self, attr + "_feature_embeddings_out_dim", self.feature_embeddings_out_dim
        )
        del self.feature_embeddings_out_dim
        setattr(
            self,
            attr + "_feature_embeddings_list",
            nn.ModuleList(feature_embeddings_list),
        )

    def process_feature_embedding(
        self, attr, sample_list, text_embedding_total, extra=None, batch_size_t=None
    ):
        if extra is None:
            extra = []
        feature_embeddings = []
        feature_attentions = []
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
            feature_idx += 1
            feature = feature[:batch_size_t]
            features.append(feature)

        feature_encoders = getattr(self, attr + "_feature_encoders")
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
            # For Pythia, we need max_features to mask attention
            feature_dim = getattr(feature_info, "max_features", None)
            if feature_dim is not None:
                feature_dim = feature_dim[:batch_size_t]

            # Attribute in which encoders are saved, for "image" it
            # will be "image_feature_encoders", other example is
            # "context_feature_encoders"
            encoders_attr = attr + "_feature_encoders"
            feature_encoder = getattr(self, encoders_attr)[i]

            # Encode the features
            encoded_feature = feature_encoder(feature)

            projector_attr = attr + "_feature_projectors"
            feature_projector = getattr(self, projector_attr)[i]

            encoded_feature = feature_projector(encoded_feature)
            # Get all of the feature embeddings
            list_attr = attr + "_feature_embeddings_list"
            feature_embedding_models = getattr(self, list_attr)[i]

            # Forward through these embeddings one by one
            for feature_embedding_model in feature_embedding_models:
                inp = (encoded_feature, text_embedding_total, feature_dim, extra)

                embedding, attention = feature_embedding_model(*inp)
                feature_embeddings.append(embedding)
                feature_attentions.append(attention.squeeze(-1))

        # Concatenate all features embeddings and return along with attention
        feature_embedding_total = torch.cat(feature_embeddings, dim=1)
        return feature_embedding_total, feature_attentions
