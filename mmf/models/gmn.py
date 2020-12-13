# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from mmf.common.registry import registry
from mmf.models.pythia import Pythia
from mmf.models.memo_net import MemoNet
from mmf.modules.embeddings import (
    ImageFeatureEmbedding,
    MultiHeadImageFeatureEmbedding,
    PreExtractedEmbedding,
    TextEmbedding,
)
from mmf.modules.layers import MemNNLayer
# from mmf.modules.layers import EdgeLayer
# from mmf.modules.layers import NodeLayer
# from mmf.modules.layers import GlobalLayer
# from mmf.modules.layers import GraphMemoryLayer
from mmf.modules.layers import GraphLayer
from mmf.modules.fusions import MCB
# from torch_scatter import scatter_mean
# from torch_geometric.nn import MetaLayer
from torch import nn
import numpy as np
import cv2



@registry.register_model("gmn")
class GraphMemoNet(MemoNet):
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
        self._init_GMN()
        self._init_MCB()
        # self._init_MN("image")
        self._init_combine_layer("image", "text")
        self._init_classifier(self._get_classifier_input_dim())
        self._init_extras()

    def _init_GMN(self):
        self.visual_graph = GraphLayer("image", hidden_dim=2048, node_dim=2048, edge_dim=2048)
        self.textual_graph = GraphLayer("text", hidden_dim=2048, node_dim=2048, edge_dim=2048)
        # 1204 2048
        # self.memory = 

    def _init_MCB(self):
        input_dims = [self.config.mcb.visual_input_dim, self.config.mcb.textual_input_dim]
        output_dim = self.config.mcb.output_dim
        self.mcb = MCB(input_dims, output_dim)

    def is_adjcent(self, bbox1, bbox2):
        d_ecu_convert = (bbox1[0]+bbox1[2]-bbox2[0]-bbox2[2])*(bbox1[0]+bbox1[2]-bbox2[0]-bbox2[2])+(bbox1[1]+bbox1[3]-bbox2[1]-bbox2[3])*(bbox1[1]+bbox1[3]-bbox2[1]-bbox2[3])
        adj = (d_ecu_convert<1.0)
        return adj


    def process_text_embedding(
        self, sample_list, embedding_attr="text_embeddings", info=None
    ):

        # print("=====text embedding=====")

        #metadata
        bs = len(sample_list.question_id) 
        num_regions_vg = []
        for i in range(bs):
            num_regions_vg.append(len(sample_list.region_description["region_id"][i]))
        
        text_embeddings = []
        embedding_phrase = [[] for i in range (bs)]
        embedding_phrases = []
        # Get "text" attribute in case of "text_embeddings" case
        # and "context" attribute in case of "context_embeddings"
        texts = getattr(sample_list, embedding_attr.split("_")[0])
        phrases = sample_list.region_description["phrase"]
        # print("text", texts.size())  # bs*20*300
        # print("phrases", phrases)

        # Get embedding models
        text_embedding_models = getattr(self, embedding_attr)
        for text_embedding_model in text_embedding_models:
            # print("text_model", text_embedding_model)
            '''
            text_model TextEmbedding(
                (module): BiLSTM(
                    ...
                )
            )
            '''
            # TODO: Move this logic inside
            if isinstance(text_embedding_model, PreExtractedEmbedding):
                embedding = text_embedding_model(sample_list.question_id)
            else:
                embedding = text_embedding_model(texts)
            # print("text_embedding: ", embedding.size()) 
            # torch.Size([4(bs), 2048attn/1280gnu])
                for i in range(bs):
                    for j in range (num_regions_vg[i]):
                        embedding_phrase[i].append(text_embedding_model(phrases[i][j].unsqueeze(0))) 
                        # print(embedding_phrase[i][j].size()) # torch.Size([1,2048/1280])
                        #[50,50,49,49*tensors]

            text_embeddings.append(embedding)
            embedding_phrases.append(embedding_phrase)           
        
        # cat different embedding models(only 1 model here)
        text_embeddding_total = torch.cat(text_embeddings, dim=1)
        # print("text_embedding_tot: ", text_embeddding_total.size()) # torch.Size([4(bs), 2048])
        # embedding_phrase_total =  torch.cat(embedding_phrases, dim=0)
        embedding_phrase_total = embedding_phrases # cannot cat a list, [1,4,50/49*[1,2048]]
        # print(len(embedding_phrase_total))
        # print(len(embedding_phrase_total[0]))
        # print(len(embedding_phrase_total[0][0]))

        return text_embeddding_total, embedding_phrase_total

        
    def process_feature_embedding(
        self, attr, sample_list, text_embedding_total, embedding_phrase_total, extra=None, batch_size_t=None
    ):
        if extra is None:
            extra = []
        feature_embeddings = []
        feature_attentions = []   #useless var
        features = []
        batch_size_t = (
            sample_list.get_batch_size() if batch_size_t is None else batch_size_t
        )

        #metadata
        bs = len(sample_list.question_id) 
        num_regions_vg = []
        for i in range(bs):
            num_regions_vg.append(len(sample_list.region_description["region_id"][i]))

        # Convert list of keys to the actual values
        extra = sample_list.get_fields(extra)
        #bboxes
        bbox_phrase_width = sample_list.region_description["width"]
        bbox_phrase_height = sample_list.region_description["height"]
        bbox_phrase_x = sample_list.region_description["x"]
        bbox_phrase_y = sample_list.region_description["y"]
        bbox_phrase = [[] for i in range (bs)]
        for i in range (bs):
            for j in range (num_regions_vg[i]):
                bbox_phrase[i].append([bbox_phrase_x[i][j], bbox_phrase_y[i][j], bbox_phrase_x[i][j]+bbox_phrase_width[i][j], bbox_phrase_y[i][j]+bbox_phrase_height[i][j]])
        # bbox_phrase_width = torch.tensor(sample_list.region_description["width"]).cuda()
        # bbox_phrase_height = torch.tensor(sample_list.region_description["height"]).cuda()
        # bbox_phrase_x = torch.tensor(sample_list.region_description["x"]).cuda()
        # bbox_phrase_y = torch.tensor(sample_list.region_description["y"]).cuda()
        # bbox_phrase_x1 = torch.add(bbox_phrase_width, bbox_phrase_x)
        # bbox_phrase_y1 = torch.add(bbox_phrase_height, bbox_phrase_y)
        # bbox_phrase = torch.stack(bbox_phrase_x, bbox_phrase_y, bbox_phrase_x1, bbox_phrase_y1)
        # print("bbox_phrase",bbox_phrase)
        bbox_feature = torch.tensor(sample_list.image_info_0["bbox"]).cuda()  # [bs, 100, 4]
        
        # ====print img with bbox==========
        # for i in range (bs):
        #     img_src = sample_list.region_description["image_url"][i]
        #     cap = cv2.VideoCapture(img_src)
        #     if( cap.isOpened() ) :
        #         ret,img = cap.read()
        #         # cv2.imshow("image",img)
        #         # cv2.waitKey()
        #     for j in bbox_feature[i]:
        #         cv2.rectangle(img, (j[0],j[1]), (j[2],j[3]), (0,255,0), 1)
        #     for k in bbox_phrase[i]:
        #         p1 = (int(k[0]*sample_list.image_info_0["image_width"][i]),int(k[1]*sample_list.image_info_0["image_height"][i]))
        #         p2 = (int(k[2]*sample_list.image_info_0["image_width"][i]),int(k[3]*sample_list.image_info_0["image_height"][i]))
        #         print(p1,p2)
        #         cv2.rectangle(img, p1, p2, (255,0,0), 1)
        #     print(sample_list.region_description["image_id"][i])
        #     cv2.imwrite(str(sample_list.region_description["image_id"][i].item())+".jpg", img)
        
        for i in range (bs):
            bbox_feature[i, :, 0] /= sample_list.image_info_0["image_width"][i]
            bbox_feature[i, :, 1] /= sample_list.image_info_0["image_height"][i]
            bbox_feature[i, :, 2] /= sample_list.image_info_0["image_width"][i]
            bbox_feature[i, :, 3] /= sample_list.image_info_0["image_height"][i]
        # print("bbox_feature", bbox_feature)

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
        # result: 
        # feat_encoders ModuleList(
        #     (0): ImageFeatureEncoder(
        #         (module): FinetuneFasterRcnnFpnFc7(
        #         (lc): Linear(in_features=2048, out_features=2048, bias=True)
        #         )
        #     )
        #     (1): ImageFeatureEncoder(
        #         (module): Identity()
        #     )
        # )

 

        # Each feature should have a separate image feature encoders
        assert len(features) == len(feature_encoders), (
            "Number of feature encoders, {} are not equal "
            "to number of features, {}.".format(len(feature_encoders), len(features))
        )

        # Now, iterate to get final attended image features
        encoded_feature = []
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
            encoded_feature.append(feature_encoder(feature)) 

            # print("encoded_feat:", i, encoded_feature.size()) # torch.Size([64, 100/196, 2048])
            #feature1--finetune; feat2--identity
            list_attr = attr + "_feature_embeddings_list"
            feature_embedding_models = getattr(self, list_attr)[i]  # image_feature_embeddings_list

        
        # print("=====feat_embedding===== ")
        # Forward through these embeddings one by one
        # current data: encoded_feature[0,1], text_embedding_total, embedding_phrase total, feature_dim, extra, bbox_phrase, bbox_feature
        
        # init graph data
        visual_node_features = [[] for i in range (bs)]
        textual_node_features = [[] for i in range (bs)]
        visual_edge_ends = [[] for i in range (bs)]
        textual_edge_ends = [[] for i in range (bs)]
        visual_edge_features = [[] for i in range (bs)]
        textual_edge_features = [[] for i in range (bs)]

        visual_edge_feat_dim = 2048 # 2048
        textual_edge_feat_dim = 2048 # 2048

        visual_global_features = encoded_feature[1][:,0,:]
        textual_global_features = text_embedding_total

        ## visual graph init
        for i in range(bs):
            for j in range (len(encoded_feature[0][i])):
                # j-th feature in i-th batch
                # print("mcb", encoded_feature[0][i][j].size(), text_embedding_total[i].size()) # torchsize([2048])
                visual_node_features[i].append((self.mcb( [encoded_feature[0][i][j], text_embedding_total[i]] )).unsqueeze(0)) # [bs, 100, 2048]
                for k in range(len(bbox_feature[i])):
                    if self.is_adjcent(bbox_feature[i][j], bbox_feature[i][k]):
                        visual_edge_ends[i].append([j,k]) # [bs, num_edges, 2]
                        visual_edge_features[i].append(torch.zeros(1,visual_edge_feat_dim))# [bs, num_edges, 2048]            
            visual_edge_features[i] = torch.cat(visual_edge_features[i]).cuda(0)
            visual_node_features[i] = torch.cat(visual_node_features[i])
                # visual_nodes[i] = VisualNodeModel(visual_node_features[i])
                # visual_edges[i] = VisualEdgeModel()

            ## textual graph init
            # print(embedding_phrase_total[i]) # [1,4,50/49,2048]
            embedding_phrase_total[0][i]=torch.cat(embedding_phrase_total[0][i], dim = 0)
            # print("embedding_phrase_total", embedding_phrase_total[0][0].size())
            # print("text_embedding_total", text_embedding_total[i].size())        
            textual_node_features[i]= (embedding_phrase_total[0][i]*text_embedding_total[i]).unsqueeze(0) # [49, 2048] * [2048] =>[49,2048]
            #  textual_node_features = torch.mmf(embedding_phrase_total,text_embedding_total)
            # print("num_regions_vg", num_regions_vg[i], len(bbox_phrase[i]))
            ## edge init
            for j in range (num_regions_vg[i]):
                # textual_node_features[i].append(multiply(embedding_phrase_total[i][j], text_embedding_total[i]))
                for k in range (len(bbox_phrase[i])):
                    if self.is_adjcent(bbox_phrase[i][j], bbox_phrase[i][k]):
                        textual_edge_ends[i].append([j,k]) # [bs, num_edges, 2]
                        textual_edge_features[i].append(torch.zeros(1,textual_edge_feat_dim))# [bs, num_edges, 2048]
            textual_edge_features[i] = torch.cat(textual_edge_features[i]).cuda(0)
            # textual_nodes[i] = TextualNodeModel(extual_node_features[i])
            # textual_edges[i] = TextualEdgeModel()
            #  
            visual_node_features[i], visual_edge_features[i], visual_global_features[i] = self.visual_graph(visual_node_features[i], visual_edge_ends[i], visual_edge_features[i], visual_global_features[i])
            # textual_node_features[i], textual_edge_features[i], textual_global_features[i] = self.textual_graph(textual_node_features[i], textual_edge_ends[i], textual_edge_features[i], textual_global_features[i])
            # visual_node_features, visual_edge_features, visual_global_features = self.visual_graph(visual_node_features, visual_edge_ends, visual_edge_features, visual_global_features)
            # textual_node_features, textual_edge_features, textual_global_features = self.textual_graph(textual_node_features, textual_edge_ends, textual_edge_features, textual_global_features)

        # print("len vis node feat", len(visual_node_features)) #[4*[100, 2048]]
        # print("vis node feat[0]", visual_node_features[1].size())
        a = torch.cat(visual_node_features, dim = 0).reshape(len(visual_node_features),100,2048)
        encoded_feature[0] = a
        # print(encoded_feature[0].size())

        for i, feature in enumerate(features):
            for feature_embedding_model in feature_embedding_models:
                inp = (encoded_feature[i], text_embedding_total, feature_dim, extra)
                # torch.Size([64, 100, 2048]), [64,2048], none, samplelist()
                # print(feature_embedding_model) #attn & identity as listed in yml
                # print(encoded_feature[i].size())
                # print(text_embedding_total.size())
                # print("erxtra", extra) # infos in samplelist with key extra 

                embedding, attention = feature_embedding_model(*inp)
                
                # out = self.GMNs[i](encoded_feature, text_embedding_total, embedding_phrase_total, )
                # memo = self.MNs[i](encoded_feature, text_embedding_total, embedding_phrase_total)  # torch.Size([bs, 2048])
                # memo = self.MNs[i](encoded_feature_512, text_embedding_total)  # torch.Size([bs, 2048])
                # print("memo:", memo.size())  
                # print("embedding", embedding.size())

                # embedding_memo_superpos = embedding + memo
                # feature_embeddings.append(embedding_memo_superpos)
                # feature_embeddings.append(memo)
                feature_embeddings.append(embedding)

        # Concatenate all features embeddings and return along with attention
        feature_embedding_total = torch.cat(feature_embeddings, dim=1)

        # print("feature_embeddings_tot", feature_embedding_total.size()) #[bs*4096]

        return feature_embedding_total, feature_attentions    

    def get_optimizer_parameters(self, config):
        combine_layer = self.image_text_multi_modal_combine_layer
        params = [
            {"params": self.word_embedding.parameters()},
            {"params": self.image_feature_embeddings_list.parameters()},
            {"params": self.text_embeddings.parameters()},
            {"params": self.mcb.parameters()},
            {"params": self.visual_graph.parameters()},
            {"params": self.textual_graph.parameters()},
            {"params": combine_layer.parameters()},
            {"params": self.classifier.parameters()},
            {
                "params": self.image_feature_encoders.parameters(),
                "lr": (config.optimizer.params.lr * 0.1),
            },
        ]

        return params

    def forward(self, sample_list):
        torch.autograd.set_detect_anomaly(True)
        # torch.cuda.set_device(0)
        print ("=====sample_list=====")
        # print(sample_list.fields())
        # for key in sample_list.keys():
        #     print(key+":")
        #     # print(type(sample_list[key]))
        #     if isinstance(sample_list[key],str) :
        #         print("str:", sample_list[key])
        #     elif isinstance(sample_list[key],dict) : # region description: dict
        #         for key2 in sample_list[key].keys():
        #             print("    "+key2+":")
        #             # if type(sample_list[key][key2]) is np.ndarray:
        #             #     print(sample_list[key][key2].shape)
        #             # else:
        #             #     print(sample_list[key][key2])
        #             print(sample_list[key][key2])
        #     elif isinstance(sample_list[key],list) :
        #         for i in sample_list[key]:  # image info 1: [none, ]
        #             if i != None:
        #                 print(i.keys(), i.values)
        #             else: 
        #                 print (i)
        #     else: 
        #         print(sample_list[key].size())
        #metadata
        bs = len(sample_list.question_id) 
        num_regions_vg = []
        for i in range(bs):
            num_regions_vg.append(len(sample_list.region_description["region_id"][i]))
        # print("num_regions_vg", num_regions_vg)

        # question & caption word embedding(word->300D)
        # print("text", sample_list.text)  #tensor[4,20]
        sample_list.text = self.word_embedding(sample_list.text)
        # print("text", sample_list.text.size()) # torch.Size([4, 20, 300])
        for i in range(bs):
            for j in range (num_regions_vg[i]):
                # [[50,20],[50,20],[49,20][49,20]]
                sample_list.region_description["phrase"][i][j] = torch.tensor(sample_list.region_description["phrase"][i][j]).cuda() #torch.Size([20])
                sample_list.region_description["phrase"][i][j] = self.word_embedding(sample_list.region_description["phrase"][i][j]) #torch.Size([20, 300])
        
        # question & caption GRU embedding
        text_embedding_total, embedding_phrase_total = self.process_text_embedding(sample_list)
        # print("text_embedding", text_embedding_total.size())

        # image feat
        image_embedding_total, _ = self.process_feature_embedding(
            "image", sample_list, text_embedding_total, embedding_phrase_total
        )
        # print("img_embedding", image_embedding_total.size())
       
        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)
        
        # print("image_embedding", image_embedding_total.size()) # [batch*4096]
        # print("text_embedding:" , image_embedding_total.size()) # [batch*4096]
        # print("=====combine layer=====")
        
        joint_embedding = self.combine_embeddings(
            ["image", "text"], [image_embedding_total, text_embedding_total]
        )

        # print("joint_embedding:", joint_embedding.size()) # [batch*5000]

        model_output = {"scores": self.calculate_logits(joint_embedding)}

        # print("model_output:", model_output['scores'].size())
        # for name, param in self.MNs.named_parameters():
        #     print (name, param)

        return model_output

