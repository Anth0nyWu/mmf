# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import json

import torch
from mmf.common.sample import Sample, SampleList
from mmf.datasets.builders.vqa2 import VQA2Dataset
from mmf.datasets.databases.scene_graph_database import SceneGraphDatabase
from mmf.datasets.databases.region_description_database import RegionDescriptionDatabase
from mmf.datasets.databases.metadata_database import MetadataDatabase
from mmf.utils.general import get_absolute_path
from mmf.utils.configuration import get_mmf_env


_CONSTANTS = {"image_id_key": "image_id"}


class VisualGenomeDataset(VQA2Dataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(
            config,
            dataset_type,
            imdb_file_index,
            dataset_name="visual_genome",
            *args,
            **kwargs
        )

        self._return_scene_graph = config.return_scene_graph
        self._return_objects = config.return_objects
        self._return_relationships = config.return_relationships
        self._return_region_descriptions = config.return_region_descriptions
        self._no_unk = config.get("no_unk", False)
        self.scene_graph_db = None
        self.region_descriptions_db = None
        self.image_metadata_db = None

        build_scene_graph_db = (
            self._return_scene_graph
            or self._return_objects
            or self._return_relationships
        )
        # print("config", config)
        if self._return_region_descriptions:
            print("use_region_descriptions_true")
            self.region_descriptions_db = self.build_region_descriptions_db()
            self.image_metadata_db = self.build_image_metadata_db()

        if build_scene_graph_db:
            scene_graph_file = config.scene_graph_files[dataset_type][imdb_file_index]
            print("scene_graph_file", scene_graph_file)
            # scene_graph_file = self._get_absolute_path(scene_graph_file)
            scene_graph_file = get_absolute_path(get_mmf_env("data_dir")+"/"+scene_graph_file)
            print("scene_graph_file", scene_graph_file)
            self.scene_graph_db = SceneGraphDatabase(config, scene_graph_file)
            print("use_scene_graph_true")
            self.scene_graph_db = self.build_scene_graph_db()

    def build_region_descriptions_db(self):
        region_descriptions_path = self._get_path_based_on_index(
            self.config, "region_descriptions", self._index
        )
        print("region_descriptions:", region_descriptions_path)
        return RegionDescriptionDatabase(
            self.config, region_descriptions_path, annotation_db=self.annotation_db
        )

    def build_image_metadata_db(self):
        metadatas_path = self._get_path_based_on_index(
            self.config, "metadatas", self._index
        )
        print("metadatas:", metadatas_path)
        return MetadataDatabase(
            self.config, metadatas_path, annotation_db=self.annotation_db
        )


    def build_scene_graph_db(self):
        scene_graph_files_path = self._get_path_based_on_index(
            self.config, "scene_graph_files", self._index
        )
        print("scene_graph_files_path:", scene_graph_files_path)
        return SceneGraphDatabase(
            self.config, scene_graph_files_path, annotation_db=self.annotation_db
        )

    def load_item(self, idx):
        # print("===load item===")
        # load idx-th line in q_a file
        # print("idx0", idx)
        sample_info = self.annotation_db[idx]
        # print("sample_info", sample_info)
        sample_info = self._preprocess_answer(sample_info)
        sample_info["question_id"] = sample_info["id"]
        if self._check_unk(sample_info):
            return self.load_item((idx + 1) % len(self.annotation_db))

        current_sample = super().load_item(idx)
        # print("current sample", current_sample)
        current_sample = self._load_scene_graph(idx, current_sample)
        current_sample = self._load_region_description(idx, current_sample)

        # print("region current sample", current_sample)
        return current_sample

    def _get_image_id(self, idx):
        return self.annotation_db[idx][_CONSTANTS["image_id_key"]]

    def _get_image_info(self, idx):
        # Deep copy so that we can directly update the nested dicts
        # return copy.deepcopy(self.scene_graph_db[self._get_image_id(idx)])
        img_id =  self._get_image_id(idx)
        # print("img id", self._get_image_id(idx))

        image_info = copy.deepcopy(self.region_descriptions_db[self._get_image_id(idx)])
        # image width/length
        image_info["height"] = self.image_metadata_db[img_id]["height"]
        image_info["width"] = self.image_metadata_db[img_id]["width"]

        return image_info

    def _preprocess_answer(self, sample_info):
        sample_info["answers"] = [
            self.vg_answer_preprocessor(
                {"text": sample_info["answers"][0]},
                remove=["?", ",", ".", "a", "an", "the"],
            )["text"]
        ]

        return sample_info

    def _check_unk(self, sample_info):
        if not self._no_unk:
            return False
        else:
            index = self.answer_processor.word2idx(sample_info["answers"][0])
            # print("ans_processor", self.answer_processor.answer_vocab)
            # print("index1", index)
            # print("index2", self.answer_processor.answer_vocab.UNK_INDEX)
            return index == self.answer_processor.answer_vocab.UNK_INDEX

    def _load_scene_graph(self, idx, sample):
        if self.scene_graph_db is None:
            return sample

        image_info = self._get_image_info(idx)
        regions = image_info["regions"]

        objects, object_map = self._load_objects(idx)

        if self._return_objects:
            sample.objects = objects

        relationships, relationship_map = self._load_relationships(idx, object_map)

        if self._return_relationships:
            sample.relationships = relationships

        regions, _ = self._load_regions(idx, object_map, relationship_map)

        if self._return_scene_graph:
            sample.scene_graph = regions

        return sample

    def _load_region_description(self, idx, sample):
        if self.region_descriptions_db is None:
            return sample
        image_info = self._get_image_info(idx)

        # print("image_info", image_info)
        regions, _ = self._load_regions(idx)
        # print("regions", regions)

        sample.region_description = regions
        return sample

    def _load_objects(self, idx):
        image_info = self._get_image_info(idx)
        image_height = image_info["height"]
        image_width = image_info["width"]
        object_map = {}
        objects = []

        for obj in image_info["objects"]:
            obj["synsets"] = self.synset_processor({"tokens": obj["synsets"]})["text"]
            obj["names"] = self.name_processor({"tokens": obj["names"]})["text"]
            obj["height"] = obj["h"] / image_height
            obj.pop("h")
            obj["width"] = obj["w"] / image_width
            obj.pop("w")
            obj["y"] /= image_height
            obj["x"] /= image_width
            obj["attributes"] = self.attribute_processor({"tokens": obj["attributes"]})[
                "text"
            ]
            obj = Sample(obj)
            object_map[obj["object_id"]] = obj
            objects.append(obj)
        objects = SampleList(objects)

        return objects, object_map

    def _load_relationships(self, idx, object_map):
        if self._return_relationships is None and self._return_scene_graph is None:
            return None, None

        image_info = self._get_image_info(idx)
        relationship_map = {}
        relationships = []

        for relationship in image_info["relationships"]:
            relationship["synsets"] = self.synset_processor(
                {"tokens": relationship["synsets"]}
            )["text"]
            relationship["predicate"] = self.predicate_processor(
                {"tokens": relationship["predicate"]}
            )["text"]
            relationship["object"] = object_map[relationship["object_id"]]
            relationship["subject"] = object_map[relationship["subject_id"]]

            relationship = Sample(relationship)
            relationship_map[relationship["relationship_id"]] = relationship
            relationships.append(relationship)

        relationships = SampleList(relationships)
        return relationships, relationship_map

    def _load_regions(self, idx, object_map, relationship_map):
        if self._return_scene_graph is None:
            return None, None

        image_info = self._get_image_info(idx)
        image_height = image_info["height"]
        image_width = image_info["width"]
        region_map = {}
        regions = []

        for region in image_info["regions"]:
            for synset in region["synsets"]:
                synset["entity_name"] = self.name_processor(
                    {"tokens": [synset["entity_name"]]}
                )["text"]
                synset["synset_name"] = self.synset_processor(
                    {"tokens": [synset["synset_name"]]}
                )["text"]

            region["height"] /= image_height
            region["width"] /= image_width
            region["y"] /= image_height
            region["x"] /= image_width

            relationships = []
            objects = []

            for relationship_idx in region["relationships"]:
                relationships.append(relationship_map[relationship_idx])

            for object_idx in region["objects"]:
                objects.append(object_map[object_idx])

            region["relationships"] = relationships
            region["objects"] = objects
            region["phrase"] = self.text_processor({"text": region["phrase"]})["text"]

            region = Sample(region)
            region_map[region["region_id"]] = region
            regions.append(region)

        regions = SampleList(regions)
        return regions, region_map

    def _load_regions(self, idx):
        if self._return_region_descriptions is None:
            return None, None

        image_info = self._get_image_info(idx)
        # print("img_info", image_info)        # {"regions":[], "id": }
        image_height = image_info["height"]
        image_width = image_info["width"]
        region_map = {}
        region_cat = {}
        region_cat["cat_description"]=[]
        regions = []
        
        for region in image_info["regions"]:
            region["height"] /= image_height
            region["width"] /= image_width
            region["y"] /= image_height
            region["x"] /= image_width
            region["phrase"] = self.text_processor({"text": region["phrase"]})["text"]
            # region["region_id"]=torch.tensor(region["region_id"])
            # region["height"]=torch.tensor(region["height"])
            # region["width"]=torch.tensor(region["width"])
            # region["y"]=torch.tensor(region["y"])
            # region["x"]=torch.tensor(region["x"])
            region["phrase"]=region["phrase"].numpy()
            # print("region", region)
            # {'region_id': 3989716, 'width': 0.050666666666666665, 'height': 0.016, 'image_id': 2332304, 'phrase': tensor([66632, 44395, 46900, 66632, 49920,     0,     0,     0,     0,     0,
            # 0,     0,     0,     0,     0,     0,     0,     0,     0,     0]), 'y': 0.182, 'x': 0.56}
            region = Sample(region)
            # sampled_region: Sample([('region_id', 3989715), ('width', 0.05333333333333334), ('height', 0.018), ('image_id', 2332304), ('phrase', tensor([48867, 46900, 66632, 60502,     0,     0,     0,     0,     0,     0,
            # 0,     0,     0,     0,     0,     0,     0,     0,     0,     0])), ('y', 0.268), ('x', 0.6426666666666667)])


            # cat region.values
            # region_cat["cat_description"]=[region["region_id"], region["height"], region["width"], region["y"], region["x"]] # .append(region["phrase"])
            # # transform to tensor
            # region_cat["cat_description"]=torch.tensor(region_cat["cat_description"]) # ??? dtype
            # # cat phrase
            # region_cat["cat_description"]= torch.cat((region_cat["cat_description"], region["phrase"].float()))           
            # region_cat = Sample(region_cat)
            # print("sampled_region_cat", region_cat)
            
            region_map[region["region_id"]] = region
            regions.append(region)

        # print("regions", regions)
        regions = SampleList(regions)
        regions["image_id"]=torch.tensor(regions["image_id"][0], dtype = torch.int32)
        # print("regions sample list", regions)
        return regions, region_map

