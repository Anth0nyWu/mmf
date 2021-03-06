# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Optional

import torch
from mmf.common.registry import registry
from mmf.modules.decoders import LanguageDecoder
from torch import nn
from torch.nn.utils.weight_norm import weight_norm

# from torch.nn import Sequential as Seq, Linear as Lin, ReLU
# from torch_scatter import scatter_mean
# from torch_geometric.nn import MetaLayer

import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from torch.autograd import Variable

class ConvNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding_size="same",
        pool_stride=2,
        batch_norm=True,
    ):
        super().__init__()

        if padding_size == "same":
            padding_size = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding_size
        )
        self.max_pool2d = nn.MaxPool2d(pool_stride, stride=pool_stride)
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.batch_norm_2d = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.max_pool2d(nn.functional.leaky_relu(self.conv(x)))

        if self.batch_norm:
            x = self.batch_norm_2d(x)

        return x


class Flatten(nn.Module):
    def forward(self, input):
        if input.dim() > 1:
            input = input.view(input.size(0), -1)

        return input


class UnFlatten(nn.Module):
    def forward(self, input, sizes=None):
        if sizes is None:
            sizes = []
        return input.view(input.size(0), *sizes)


class GatedTanh(nn.Module):
    """
    From: https://arxiv.org/pdf/1707.07998.pdf
    nonlinear_layer (f_a) : x\\in R^m => y \\in R^n
    \tilda{y} = tanh(Wx + b)
    g = sigmoid(W'x + b')
    y = \tilda(y) \\circ g
    input: (N, *, in_dim)
    output: (N, *, out_dim)
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.gate_fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        y_tilda = torch.tanh(self.fc(x))
        gated = torch.sigmoid(self.gate_fc(x))

        # Element wise multiplication
        y = y_tilda * gated

        return y


# TODO: Do clean implementation without Sequential
class ReLUWithWeightNormFC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        layers = []
        layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ClassifierLayer(nn.Module):
    def __init__(self, classifier_type, in_dim, out_dim, **kwargs):
        super().__init__()

        if classifier_type == "weight_norm":
            self.module = WeightNormClassifier(in_dim, out_dim, **kwargs)
        elif classifier_type == "logit":
            self.module = LogitClassifier(in_dim, out_dim, **kwargs)
        elif classifier_type == "language_decoder":
            self.module = LanguageDecoder(in_dim, out_dim, **kwargs)
        elif classifier_type == "bert":
            self.module = BertClassifierHead(
                in_dim, out_dim, kwargs.get("config", None)
            ).module
        elif classifier_type == "mlp":
            self.module = MLPClassifer(in_dim, out_dim, **kwargs)
        elif classifier_type == "triple_linear":
            self.module = TripleLinear(in_dim, out_dim)
        elif classifier_type == "linear":
            self.module = nn.Linear(in_dim, out_dim)
        else:
            raise NotImplementedError("Unknown classifier type: %s" % classifier_type)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class BertClassifierHead(nn.Module):
    def __init__(self, in_dim=768, out_dim=2, config=None, *args, **kwargs):
        super().__init__()
        from transformers.modeling_bert import BertPredictionHeadTransform

        if config is None:
            from transformers.configuration_bert import BertConfig

            config = BertConfig.from_pretrained("bert-base-uncased")

        assert config.hidden_size == in_dim

        self.module = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            BertPredictionHeadTransform(config),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class MLPClassifer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=None,
        num_layers=0,
        dropout=0.5,
        hidden_act="relu",
        batch_norm=True,
        **kwargs,
    ):
        super().__init__()
        from mmf.utils.modeling import ACT2FN

        activation = ACT2FN[hidden_act]
        self.layers = nn.ModuleList()

        if hidden_dim is None:
            hidden_dim = in_dim

        for _ in range(num_layers):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(activation())
            self.layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.layers.append(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LogitClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        input_dim = in_dim
        num_ans_candidates = out_dim
        text_non_linear_dim = kwargs["text_hidden_dim"]
        image_non_linear_dim = kwargs["img_hidden_dim"]

        self.f_o_text = ReLUWithWeightNormFC(input_dim, text_non_linear_dim)
        self.f_o_image = ReLUWithWeightNormFC(input_dim, image_non_linear_dim)
        self.linear_text = nn.Linear(text_non_linear_dim, num_ans_candidates)
        self.linear_image = nn.Linear(image_non_linear_dim, num_ans_candidates)

        if "pretrained_image" in kwargs and kwargs["pretrained_text"] is not None:
            self.linear_text.weight.data.copy_(
                torch.from_numpy(kwargs["pretrained_text"])
            )

        if "pretrained_image" in kwargs and kwargs["pretrained_image"] is not None:
            self.linear_image.weight.data.copy_(
                torch.from_numpy(kwargs["pretrained_image"])
            )

    def forward(self, joint_embedding):
        text_val = self.linear_text(self.f_o_text(joint_embedding))
        image_val = self.linear_image(self.f_o_image(joint_embedding))
        logit_value = text_val + image_val

        return logit_value


class WeightNormClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, dropout):
        super().__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hidden_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hidden_dim, out_dim), dim=None),
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class Identity(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class ModalCombineLayer(nn.Module):
    def __init__(self, combine_type, img_feat_dim, txt_emb_dim, **kwargs):
        super().__init__()
        if combine_type == "MFH":
            self.module = MFH(img_feat_dim, txt_emb_dim, **kwargs)
        elif combine_type == "non_linear_element_multiply":
            self.module = NonLinearElementMultiply(img_feat_dim, txt_emb_dim, **kwargs)
        elif combine_type == "two_layer_element_multiply":
            self.module = TwoLayerElementMultiply(img_feat_dim, txt_emb_dim, **kwargs)
        elif combine_type == "top_down_attention_lstm":
            self.module = TopDownAttentionLSTM(img_feat_dim, txt_emb_dim, **kwargs)
        elif combine_type == "mem_nn":
            # print (img_feat_dim, txt_emb_dim, kwargs)
            # (2048, 2048, params{'dropout': 0, 'hidden_dim': 5000})
            # self, vocab_size, embd_size, ans_size, max_story_len, hops=3, dropout=0.2, te=True, pe=True
            self.module = MemNNLayer(**kwargs)
        else:
            raise NotImplementedError("Not implemented combine type: %s" % combine_type)

        self.out_dim = self.module.out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class MfbExpand(nn.Module):
    def __init__(self, img_feat_dim, txt_emb_dim, hidden_dim, dropout):
        super().__init__()
        self.lc_image = nn.Linear(in_features=img_feat_dim, out_features=hidden_dim)
        self.lc_ques = nn.Linear(in_features=txt_emb_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_feat, question_embed):
        image1 = self.lc_image(image_feat)
        ques1 = self.lc_ques(question_embed)
        if len(image_feat.data.shape) == 3:
            num_location = image_feat.data.size(1)
            ques1_expand = torch.unsqueeze(ques1, 1).expand(-1, num_location, -1)
        else:
            ques1_expand = ques1
        joint_feature = image1 * ques1_expand
        joint_feature = self.dropout(joint_feature)
        return joint_feature


class MFH(nn.Module):
    def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
        super().__init__()
        self.mfb_expand_list = nn.ModuleList()
        self.mfb_sqz_list = nn.ModuleList()
        self.relu = nn.ReLU()

        hidden_sizes = kwargs["hidden_sizes"]
        self.out_dim = int(sum(hidden_sizes) / kwargs["pool_size"])

        self.order = kwargs["order"]
        self.pool_size = kwargs["pool_size"]

        for i in range(self.order):
            mfb_exp_i = MfbExpand(
                img_feat_dim=image_feat_dim,
                txt_emb_dim=ques_emb_dim,
                hidden_dim=hidden_sizes[i],
                dropout=kwargs["dropout"],
            )
            self.mfb_expand_list.append(mfb_exp_i)
            self.mfb_sqz_list.append(self.mfb_squeeze)

    def forward(self, image_feat, question_embedding):
        feature_list = []
        prev_mfb_exp = 1

        for i in range(self.order):
            mfb_exp = self.mfb_expand_list[i]
            mfb_sqz = self.mfb_sqz_list[i]
            z_exp_i = mfb_exp(image_feat, question_embedding)
            if i > 0:
                z_exp_i = prev_mfb_exp * z_exp_i
            prev_mfb_exp = z_exp_i
            z = mfb_sqz(z_exp_i)
            feature_list.append(z)

        # append at last feature
        cat_dim = len(feature_list[0].size()) - 1
        feature = torch.cat(feature_list, dim=cat_dim)
        return feature

    def mfb_squeeze(self, joint_feature):
        # joint_feature dim: N x k x dim or N x dim

        orig_feature_size = len(joint_feature.size())

        if orig_feature_size == 2:
            joint_feature = torch.unsqueeze(joint_feature, dim=1)

        batch_size, num_loc, dim = joint_feature.size()

        if dim % self.pool_size != 0:
            exit(
                "the dim %d is not multiply of \
             pool_size %d"
                % (dim, self.pool_size)
            )

        joint_feature_reshape = joint_feature.view(
            batch_size, num_loc, int(dim / self.pool_size), self.pool_size
        )

        # N x 100 x 1000 x 1
        iatt_iq_sumpool = torch.sum(joint_feature_reshape, 3)

        iatt_iq_sqrt = torch.sqrt(self.relu(iatt_iq_sumpool)) - torch.sqrt(
            self.relu(-iatt_iq_sumpool)
        )

        iatt_iq_sqrt = iatt_iq_sqrt.view(batch_size, -1)  # N x 100000
        iatt_iq_l2 = nn.functional.normalize(iatt_iq_sqrt)
        iatt_iq_l2 = iatt_iq_l2.view(batch_size, num_loc, int(dim / self.pool_size))

        if orig_feature_size == 2:
            iatt_iq_l2 = torch.squeeze(iatt_iq_l2, dim=1)

        return iatt_iq_l2


# need to handle two situations,
# first: image (N, K, i_dim), question (N, q_dim);
# second: image (N, i_dim), question (N, q_dim);
class NonLinearElementMultiply(nn.Module):
    def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
        super().__init__()

        # print("img_feat_dim", image_feat_dim)
        # print("que_emb_dim", image_feat_dim)
        # print ("kwargs", kwargs)

        self.fa_image = ReLUWithWeightNormFC(image_feat_dim, kwargs["hidden_dim"])
        self.fa_txt = ReLUWithWeightNormFC(ques_emb_dim, kwargs["hidden_dim"])

        context_dim = kwargs.get("context_dim", None)
        if context_dim is not None:
            self.fa_context = ReLUWithWeightNormFC(context_dim, kwargs["hidden_dim"])

        self.dropout = nn.Dropout(kwargs["dropout"])
        self.out_dim = kwargs["hidden_dim"]

    def forward(self, image_feat, question_embedding, context_embedding=None):
        image_fa = self.fa_image(image_feat)
        question_fa = self.fa_txt(question_embedding)

        # print("image_fa:")
        # print(image_fa.size())
        # print("question_fa:")
        # print(question_fa.size())

        if len(image_feat.size()) == 3 and len(question_fa.size()) != 3:
            question_fa_expand = question_fa.unsqueeze(1)
        else:
            question_fa_expand = question_fa

        joint_feature = image_fa * question_fa_expand

        if context_embedding is not None:
            context_fa = self.fa_context(context_embedding)

            context_text_joint_feaure = context_fa * question_fa_expand
            joint_feature = torch.cat([joint_feature, context_text_joint_feaure], dim=1)

        joint_feature = self.dropout(joint_feature)

        # print("joint_feature in layer:")
        # print(joint_feature.size())

        return joint_feature


class TopDownAttentionLSTM(nn.Module):
    def __init__(self, image_feat_dim, embed_dim, **kwargs):
        super().__init__()
        self.fa_image = weight_norm(nn.Linear(image_feat_dim, kwargs["attention_dim"]))
        self.fa_hidden = weight_norm(
            nn.Linear(kwargs["hidden_dim"], kwargs["attention_dim"])
        )
        self.top_down_lstm = nn.LSTMCell(
            embed_dim + image_feat_dim + kwargs["hidden_dim"],
            kwargs["hidden_dim"],
            bias=True,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(kwargs["dropout"])
        self.out_dim = kwargs["attention_dim"]

    def forward(self, image_feat, embedding):
        image_feat_mean = image_feat.mean(1)

        # Get LSTM state
        state = registry.get(f"{image_feat.device}_lstm_state")
        h1, c1 = state["td_hidden"]
        h2, c2 = state["lm_hidden"]

        h1, c1 = self.top_down_lstm(
            torch.cat([h2, image_feat_mean, embedding], dim=1), (h1, c1)
        )

        state["td_hidden"] = (h1, c1)

        image_fa = self.fa_image(image_feat)
        hidden_fa = self.fa_hidden(h1)

        joint_feature = self.relu(image_fa + hidden_fa.unsqueeze(1))
        joint_feature = self.dropout(joint_feature)

        return joint_feature


class TwoLayerElementMultiply(nn.Module):
    def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
        super().__init__()

        self.fa_image1 = ReLUWithWeightNormFC(image_feat_dim, kwargs["hidden_dim"])
        self.fa_image2 = ReLUWithWeightNormFC(
            kwargs["hidden_dim"], kwargs["hidden_dim"]
        )
        self.fa_txt1 = ReLUWithWeightNormFC(ques_emb_dim, kwargs["hidden_dim"])
        self.fa_txt2 = ReLUWithWeightNormFC(kwargs["hidden_dim"], kwargs["hidden_dim"])

        self.dropout = nn.Dropout(kwargs["dropout"])

        self.out_dim = kwargs["hidden_dim"]

    def forward(self, image_feat, question_embedding):
        image_fa = self.fa_image2(self.fa_image1(image_feat))
        question_fa = self.fa_txt2(self.fa_txt1(question_embedding))

        if len(image_feat.size()) == 3:
            num_location = image_feat.size(1)
            question_fa_expand = torch.unsqueeze(question_fa, 1).expand(
                -1, num_location, -1
            )
        else:
            question_fa_expand = question_fa

        joint_feature = image_fa * question_fa_expand
        joint_feature = self.dropout(joint_feature)

        return joint_feature


class TransformLayer(nn.Module):
    def __init__(self, transform_type, in_dim, out_dim, hidden_dim=None):
        super().__init__()

        if transform_type == "linear":
            self.module = LinearTransform(in_dim, out_dim)
        elif transform_type == "conv":
            self.module = ConvTransform(in_dim, out_dim, hidden_dim)
        else:
            raise NotImplementedError(
                "Unknown post combine transform type: %s" % transform_type
            )
        self.out_dim = self.module.out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class LinearTransform(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lc = weight_norm(
            nn.Linear(in_features=in_dim, out_features=out_dim), dim=None
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.lc(x)


class ConvTransform(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=hidden_dim, kernel_size=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=out_dim, kernel_size=1
        )
        self.out_dim = out_dim

    def forward(self, x):
        if len(x.size()) == 3:  # N x k xdim
            # N x dim x k x 1
            x_reshape = torch.unsqueeze(x.permute(0, 2, 1), 3)
        elif len(x.size()) == 2:  # N x dim
            # N x dim x 1 x 1
            x_reshape = torch.unsqueeze(torch.unsqueeze(x, 2), 3)

        iatt_conv1 = self.conv1(x_reshape)  # N x hidden_dim x * x 1
        iatt_relu = nn.functional.relu(iatt_conv1)
        iatt_conv2 = self.conv2(iatt_relu)  # N x out_dim x * x 1

        if len(x.size()) == 3:
            iatt_conv3 = torch.squeeze(iatt_conv2, 3).permute(0, 2, 1)
        elif len(x.size()) == 2:
            iatt_conv3 = torch.squeeze(torch.squeeze(iatt_conv2, 3), 2)

        return iatt_conv3


class BCNet(nn.Module):
    """
    Simple class for non-linear bilinear connect network
    """

    def __init__(self, v_dim, q_dim, h_dim, h_out, act="ReLU", dropout=None, k=3):
        super().__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out
        if dropout is None:
            dropout = [0.2, 0.5]

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])

        if k > 1:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out is None:
            pass

        elif h_out <= self.c:
            self.h_mat = nn.Parameter(
                torch.Tensor(1, h_out, 1, h_dim * self.k).normal_()
            )
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v, q):
        if self.h_out is None:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)
            logits = d_.transpose(1, 2).transpose(2, 3)
            return logits

        # broadcast Hadamard product, matrix-matrix production
        # fast computation but memory inefficient
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v)).unsqueeze(1)
            q_ = self.q_net(q)
            h_ = v_ * self.h_mat
            logits = torch.matmul(h_, q_.unsqueeze(1).transpose(2, 3))
            logits = logits + self.h_bias
            return logits

        # batch outer product, linear projection
        # memory efficient but slow computation
        else:
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))
            return logits.transpose(2, 3).transpose(1, 2)

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v).transpose(1, 2).unsqueeze(2)
        q_ = self.q_net(q).transpose(1, 2).unsqueeze(3)
        logits = torch.matmul(torch.matmul(v_, w.unsqueeze(1)), q_)
        logits = logits.squeeze(3).squeeze(2)

        if self.k > 1:
            logits = logits.unsqueeze(1)
            logits = self.p_net(logits).squeeze(1) * self.k

        return logits


class FCNet(nn.Module):
    """
    Simple class for non-linear fully connect network
    """

    def __init__(self, dims, act="ReLU", dropout=0):
        super().__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))

            if act is not None:
                layers.append(getattr(nn, act)())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))

        if act is not None:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=None):
        super().__init__()
        if dropout is None:
            dropout = [0.2, 0.5]
        self.glimpse = glimpse
        self.logits = weight_norm(
            BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3),
            name="h_mat",
            dim=None,
        )

    def forward(self, v, q, v_mask=True):
        p, logits = self.forward_all(v, q, v_mask)
        return p, logits

    def forward_all(self, v, q, v_mask=True):
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v, q)

        if v_mask:
            v_abs_sum = v.abs().sum(2)
            mask = (v_abs_sum == 0).unsqueeze(1).unsqueeze(3)
            mask = mask.expand(logits.size())
            logits.masked_fill_(mask, -float("inf"))

        expanded_logits = logits.view(-1, self.glimpse, v_num * q_num)
        p = nn.functional.softmax(expanded_logits, 2)

        return p.view(-1, self.glimpse, v_num, q_num), logits


class TripleLinear(nn.Module):
    """
    The three-branch classifier in https://arxiv.org/abs/2004.11883:
    During training, all three branches will produce the prediction on its own.
    During inference, only the fused branch is used to predict the answers.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(3)])

    def forward(self, joint_embedding: torch.Tensor) -> torch.Tensor:
        if self.training:
            feat = [self.linears[i](joint_embedding[:, i]) for i in range(3)]
            return torch.stack(feat, dim=1)

        return self.linears[0](joint_embedding)


class BranchCombineLayer(nn.Module):
    """Three-branch fusion module used for fusing MoVie and MCAN in
    https://arxiv.org/abs/2004.11883
    """

    def __init__(self, img_dim: int, ques_dim: int):
        super().__init__()
        self.out_dim = img_dim * 2
        self.linear_cga = nn.ModuleList(
            [nn.Linear(img_dim, self.out_dim) for _ in range(2)]
        )
        self.linear_cbn = nn.ModuleList(
            [nn.Linear(img_dim, self.out_dim) for _ in range(2)]
        )
        self.linear_ques = nn.ModuleList(
            [nn.Linear(ques_dim, self.out_dim) for _ in range(2)]
        )
        self.layer_norm = nn.ModuleList([nn.LayerNorm(self.out_dim) for _ in range(3)])

    def forward(
        self, v_cga: torch.Tensor, v_cbn: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        feat = [
            self.layer_norm[0](
                self.linear_ques[0](q)
                + self.linear_cbn[0](v_cbn)
                + self.linear_cga[0](v_cga)
            ),
            self.layer_norm[1](self.linear_cbn[1](v_cbn)),
            self.layer_norm[2](self.linear_ques[1](q) + self.linear_cga[1](v_cga)),
        ]

        if self.training:
            return torch.stack(feat, dim=1)

        return feat[0]


class AttnPool1d(nn.Module):
    """An attention pooling layer that learns weights using an mlp
    """

    def __init__(self, num_features: int, num_attn: int = 1, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(num_features // 2, num_attn),
        )
        self.p_attn = None
        self.num_attn = num_attn

    def forward(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = query.size(0)
        score = self.linear(query).transpose(-2, -1)
        if mask is not None:
            score.data.masked_fill_(mask.unsqueeze(1), -1e9)
        self.p_attn = nn.functional.softmax(score, dim=-1)

        return torch.matmul(self.p_attn, value).view(b, self.num_attn, -1)


# TODO 

class EdgeModel(nn.Module):
    def __init__(self, num_nodes):
        super(EdgeModel, self).__init__()
        self.edges = zeros()

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)

class NodeModel(nn.Module):
    def __init__(self, nodes):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = Seq(Lin(..., ...), ReLU(), Lin(..., ...))
        self.node_mlp_2 = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u[batch]], dim=1)
        return self.node_mlp_2(out)

class GlobalModel(nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.global_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        return self.global_mlp(out)


class GparhMemoryLayer(nn.Module):
    # def __init__(self, vocab_size, embd_size, ans_size, max_story_len, hops=3, dropout=0.1, te=False, pe=False):

    """
    metalayer
    https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.meta.MetaLayer
    """

    def __init__(self, edge_model=None, node_model=None, global_model=None):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(3)])

    def forward(self, joint_embedding: torch.Tensor) -> torch.Tensor:
        if self.training:
            feat = [self.linears[i](joint_embedding[:, i]) for i in range(3)]
            return torch.stack(feat, dim=1)

        return self.linears[0](joint_embedding)


class MemNNLayer(nn.Module):
    # self, image_feat_dim, ques_emb_dim, **kwargs
    def __init__(self, vocab_size, embd_size, ans_size, max_story_len, hops=3, dropout=0.1, te=False, pe=False):
        super(MemNNLayer, self).__init__()
        self.hops = hops
        self.embd_size = embd_size
        self.temporal_encoding = te
        self.position_encoding = pe

        init_rng = 0.1
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.A = nn.ModuleList([nn.Embedding(vocab_size, embd_size) for _ in range(hops+1)])  #

        # print("emb_A", self.A[0], "*", len(self.A)) # (2048, 30) * 4

        for i in range(len(self.A)):
            self.A[i].weight.data.normal_(0, init_rng)
            self.A[i].weight.data[0] = 0 # for padding index
        self.B = self.A[0] # query encoder

        # Temporal Encoding: see 4.1
        if self.temporal_encoding:
            self.TA = nn.Parameter(torch.Tensor(1, max_story_len, embd_size).normal_(0, 0.1))
            self.TC = nn.Parameter(torch.Tensor(1, max_story_len, embd_size).normal_(0, 0.1))

        self.out_dim = ans_size

    def forward(self, x, q):
        # print("===MN===")
        # x (bs, story_len, s_sent_len) (batch_size, story_len, sentence length)
        # q (bs, q_sent_len)

        # print("x:",x.size()) # x: torch.Size([4, 100, 2048])
        # print("q:", q.size()) # torch.Size([4, 2048])

        bs = x.size(0)
        story_len = x.size(1)
        s_sent_len = x.size(2)

        # Position Encoding
        if self.position_encoding:
            J = s_sent_len
            d = self.embd_size
            # pe = to_var(torch.zeros(J, d)) # (s_sent_len, embd_size)
            pe = torch.zeros(J, d)
            if torch.cuda.is_available():
                pe = Variable(pe.cuda())


            for j in range(1, J+1):
                for k in range(1, d+1):
                    l_kj = (1 - j / J) - (k / d) * (1 - 2 * j / J)
                    pe[j-1][k-1] = l_kj
            pe = pe.unsqueeze(0).unsqueeze(0) # (1, 1, s_sent_len, embd_size)
            pe = pe.repeat(bs, story_len, 1, 1) # (bs, story_len, s_sent_len, embd_size)

        x = x.view(bs*story_len, -1) # (bs*story_len, s_sent_len)

        q=q.long()
        u = self.dropout(self.B(q)) # (bs, q_sent_len, embd_size) [4, 2048, 30]
        # print("emb_B", self.B)
        # print("B(q)", self.B(q).size())
        # print("u1", u.size())
        u = torch.sum(u, 1) # (bs, embd_size) [4, 30]
        # print("u2", u.size())

        # Adjacent weight tying
        for k in range(self.hops):
            x=x.long()
            m = self.dropout(self.A[k](x))            # (bs*story_len, s_sent_len, embd_size)
            # print("m1", m.size())
            m = m.view(bs, story_len, s_sent_len, -1) # (bs, story_len, s_sent_len, embd_size)
            # print("m2", m.size())
            if self.position_encoding:
                m *= pe # (bs, story_len, s_sent_len, embd_size)
            m = torch.sum(m, 2) # (bs, story_len, embd_size)
            if self.temporal_encoding:
                m += self.TA.repeat(bs, 1, 1)[:, :story_len, :]

            c = self.dropout(self.A[k+1](x))           # (bs*story_len, s_sent_len, embd_size)
            # print("c1:", c.size()) # [784/400, 2048, 30]
            c = c.view(bs, story_len, s_sent_len, -1)  # (bs, story_len, s_sent_len, embd_size)
            c = torch.sum(c, 2)                        # (bs, story_len, embd_size)
            # print("c2:", c.size()) # [4, 196/100, 30]

            if self.temporal_encoding:
                c += self.TC.repeat(bs, 1, 1)[:, :story_len, :] # (bs, story_len, embd_size)

            m = torch.bmm(m, u.unsqueeze(2)).squeeze() # (bs, story_len)
            m = F.softmax(m, -1).unsqueeze(1)          # (bs, 1, story_len)
            m = torch.bmm(m, c).squeeze(1)             # use m as c, (bs, embd_size)
            u = m + u # (bs, embd_size)

            # p = torch.bmm(m, u.unsqueeze(2)).squeeze() # (bs, story_len)
            # p = F.softmax(p, -1).unsqueeze(1)          # (bs, 1, story_len)
            # o = torch.bmm(p, c).squeeze(1)             # use m as c, (bs, embd_size)
            # u = o + u # (bs, embd_size)[4,30]

        W = torch.t(self.A[-1].weight) # (embd_size, vocab_size)
        # A is a list of embedders, of which weight is in shape of [2048,30]
        # print("W_t", self.A[-1].weight)
        # print("W:", W.size())  # torch.Size([30, 2048])
        # print("W:", W)
       
        u = torch.bmm(u.unsqueeze(1), W.unsqueeze(0).repeat(bs, 1, 1)).squeeze() # (bs, ans_size)
        # [bs, 1, 30]*[bs,30,2048]=>[bs, 2048]
        # out = torch.bmm(u.unsqueeze(1), W.unsqueeze(0).repeat(bs, 1, 1)).squeeze() # (bs, ans_size)
        # self.out_dim = out.size()
        # print("out_dim", self.out_dim)  # 300 seems useless
        # print("out", out.size())  # torch.Size([4, 2048])
        # print("softmax")
        # print("sotfmax", F.log_softmax(out, -1).size()) # torch.Size([4, 2048])

        return u
        # return out
        # return value is weighted features or softmax of it

        # return F.log_softmax(out, -1)

class GraphLayer(nn.Module):
    def __init__(self, attr, hidden_dim, node_dim, edge_dim):
            from mmf.modules.embeddings import BiLSTMTextEmbedding
            super( GraphLayer, self ).__init__() 
            # self.phi_e = BiLSTMTextEmbedding( 
            #     hidden_dim = 2048,
            #     embedding_dim = 300,
            #     num_layers = 1,
            #     dropout = 0.0,
            #     out_dim = 2048
            #     )
            # self.phi_v = BiLSTMTextEmbedding( 
            #     hidden_dim = 2048,
            #     embedding_dim = 300,
            #     num_layers = 1,
            #     dropout = 0.0,
            #     out_dim = 2048
            #     )
            # self.phi_u = BiLSTMTextEmbedding( 
            #     hidden_dim = 2048,
            #     embedding_dim = 300,
            #     num_layers = 1,
            #     dropout = 0.0,
            #     out_dim = 2048
            #     )
            self.phi_e = nn.GRU(input_size=2048, hidden_size=2048, batch_first=True)
            self.phi_v = nn.GRU(input_size=2048, hidden_size=2048, batch_first=True)
            self.phi_u = nn.GRU(input_size=2048, hidden_size=2048, batch_first=True)
            # self.phi_e = nn.GRUCell(input_size=2048, hidden_size=2048)
            # self.phi_e = myGRU(2048,2048)
            # self.phi_v = myGRU(2048,2048)
            # self.phi_u = myGRU(2048,2048)

            self.W1 = nn.Parameter(torch.Tensor(hidden_dim, node_dim))
            self.b1 = nn.Parameter(torch.Tensor(hidden_dim))
            self.W2 = nn.Parameter(torch.Tensor(hidden_dim, node_dim))
            self.b2 = nn.Parameter(torch.Tensor(hidden_dim))
            self.W3 = nn.Parameter(torch.Tensor(hidden_dim, edge_dim))
            self.b3 = nn.Parameter(torch.Tensor(hidden_dim))
            self.W4 = nn.Parameter(torch.Tensor(hidden_dim, edge_dim))
            self.b4 = nn.Parameter(torch.Tensor(hidden_dim))

    def rho_vu(self, node_features, W1, W2, b1, b2):
        out = torch.zeros(len(b1)).cuda(0)
        for i in node_features :
            out = torch.add(out, (torch.sigmoid(torch.add(torch.mm(W1, i.unsqueeze(1)).squeeze(1), b1))*torch.tanh(torch.add(torch.mm(W2, i.unsqueeze(1)).squeeze(1), b2))))
            # more simple torch.sum()
        return torch.tanh(out)

    def rho_eu(self, edge_features, W1, W2, b1, b2):
        out = torch.zeros(len(b1)).cuda(0)
        for i in edge_features :
            # a = torch.sigmoid(torch.add(torch.mm(W1, i.unsqueeze(1)).squeeze(1), b1))
            # b = torch.tanh(torch.add(torch.mm(W2, i.unsqueeze(1)).squeeze(1), b2))
            # print(a.size(), b.size())
            out = torch.add(out, (torch.sigmoid(torch.add(torch.mm(W1, i.unsqueeze(1)).squeeze(1), b1))*torch.tanh(torch.add(torch.mm(W2, i.unsqueeze(1)).squeeze(1), b2))))
            # more simple torch.sum()
        return torch.tanh(out)

    def rho_ev(self, edge_features_adj):
        return torch.sum(edge_features_adj, dim = 0)


    def forward(self, node_features, edge_ends, edge_features, global_features):
        num_edges = len(edge_ends) #5554, kilos of edges
        num_nodes = len(node_features)
        # print(num_edges, num_nodes) #5554, kilos of edges -> 1k
        # print(len(edge_ends[0]))  # 2, 2 nodes
        # print(type(node_features), type(edge_features), type(global_features))
        # print(len(node_features), len(edge_features), len(global_features)) # 100 5554 196
        # print(node_features[0].size(), edge_features[0].size(), global_features[0].size()) # torch size 2048
        # node_features = torch.cat(node_features, dim = 0)
        # edge_features = torch.cat(edge_features, dim = 0)
        # print(node_features.size(), edge_features.size(), global_features.size())

        for i in range(num_edges): # i-th edge
            # compute new edge
            # print(edge_ends[i])
            # print(edge_features[i].unsqueeze(0).size(), node_features[edge_ends[i][0]].unsqueeze(0).size(), global_features.size())       
            gru_input_e = torch.cat((node_features[edge_ends[i][0]].unsqueeze(0), node_features[edge_ends[i][1]].unsqueeze(0) ),  dim = 0) # cat global
            gru_hidden_e = edge_features[i].unsqueeze(0).unsqueeze(0) #.cuda(0)
            # print("gru1 in", gru_input_e.size(), gru_hidden_e.size())
            gru_out_e, gru_hidnew_e = self.phi_e(gru_input_e.unsqueeze(0), gru_hidden_e)
            # print("gru1 out", gru_out_e.size(), gru_hidnew_e.size())
            edge_features[i] = gru_hidnew_e.squeeze(0).squeeze(0)
        
        for j in range (num_nodes): # j-th node
            edge_features_adj = []
            for i in range(num_edges):
                if (edge_ends[i][0] == j):
                    edge_features_adj.append(edge_features[i].unsqueeze(0))
            edge_features_adj = torch.cat(edge_features_adj, dim = 0)
            # print("edge_feat_adj", edge_features_adj.size())
            aggr_edge_adj = self.rho_ev(edge_features_adj)
            gru_input_v = aggr_edge_adj.unsqueeze(0)#.unsqueeze(0)# .cuda(0) # cat global
            gru_hidden_v = node_features[j].unsqueeze(0).unsqueeze(0)
            # print("gru2 in", gru_input_v.size(), gru_hidden_v.size())
            gru_out_v, gru_hidnew_v = self.phi_v(gru_input_v.unsqueeze(0), gru_hidden_v)
            # gru_out_v, gru_hidnew_v = gru_input_v, gru_hidden_v
            # print("gru2 out", gru_out_v.size(), gru_hidnew_v.size())
            gru_hidnew_v.detach_()
            node_features[j] = gru_hidnew_v.squeeze(0).squeeze(0)

        aggr_edge_total = self.rho_eu(edge_features, self.W3, self.W4, self.b3, self.b4)
        aggr_node_total = self.rho_vu(node_features, self.W1, self.W2, self.b1, self.b2)
        gru_hidden_u = global_features.unsqueeze(0).unsqueeze(0)
        gru_input_u = torch.cat((aggr_edge_total.unsqueeze(0), aggr_node_total.unsqueeze(0)), dim = 0)
        # print(gru_input_u.size(), gru_hidden_u.size())
        gru_out_u, gru_hidnew_u = self.phi_u(gru_input_u.unsqueeze(0), gru_hidden_u)
        # gru_out_u, gru_hidnew_u = gru_input_u.unsqueeze(0), gru_hidden_u
        gru_hidnew_u.detach_()
        global_features = gru_hidnew_u.squeeze(0).squeeze(0)

        return node_features, edge_features, global_features

class myGRU(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super( myGRU, self ).__init__() 
        self.rnn = nn.GRU(input_size = embedding_dim,  hidden_size = hidden_dim, batch_first=True)
        
    def init_hidden(self, init):
        # 开始时刻, 没有隐状态
        # 关于维度设置的详情,请参考 Pytorch 文档
        # 各个维度的含义是 (Seguence, minibatch_size, hidden_dim)
        return init
    def forward(self, x ,y):
        # 
        self.hidden = self.init_hidden(y)  # !!!
        out1, self.hidden_out = self.rnn(x,self.hidden)     
        # 
        return out1, self.hidden_out

# class MemoLayer(nn.Module):
#     def __init__(self, H, W):
#             # H*W 
#     def is_located_on(self, bbox, grid): # to judge if a bbox locates on a grid
#         pass