# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.datasets.databases.annotation_database import AnnotationDatabase


class SceneGraphDatabase(AnnotationDatabase):
    def __init__(self, config, scene_graph_path, *args, **kwargs):
        super().__init__(config, scene_graph_path, *args, **kwargs)
        #上面一行内存溢出
        self.data_dict = {}
        for item in self.data:
            # print("item", type(item))
            # print("image_id" , [item["image_id"]])
            self.data_dict[item["image_id"]] = item

    def __getitem__(self, idx):
        return self.data_dict[idx]
