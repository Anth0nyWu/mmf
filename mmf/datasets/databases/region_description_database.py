
from mmf.datasets.databases.annotation_database import AnnotationDatabase


class RegionDescriptionDatabase(AnnotationDatabase):
    def __init__(self, config, scene_graph_path, *args, **kwargs):
        super().__init__(config, scene_graph_path, *args, **kwargs)
        self.data_dict = {}
        for item in self.data:
            # print("item", type(item))
            # print("image_id" , [item["id"]])
            self.data_dict[item["id"]] = item
            

    def __getitem__(self, idx):
        return self.data_dict[idx]