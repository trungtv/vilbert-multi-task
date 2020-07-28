# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Requires vqa-maskrcnn-benchmark to be built and installed. See Readme
# for more details.
import argparse
import os
import insightface

import cv2
import numpy as np
import torch
from PIL import Image
import glob

class FeatureExtractor:
    IMAGE_SIZE = (122, 122)

    def __init__(self):
        self.args = self.get_parser().parse_args()
        self.embedding_model = self._build_embeddign_model()

        os.makedirs(self.args.output_folder, exist_ok=True)

    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
        parser.add_argument(
            "--num_features",
            type=int,
            default=100,
            help="Number of features to extract.",
        )
        parser.add_argument(
            "--output_folder", type=str, default="./output", help="Output folder"
        )
        parser.add_argument(
            "--model_name", type=str, default="arcface_r100_v1", help="Model name"
        )
        parser.add_argument(
            "--model_ctx_id", type=int, default=-1, help="Model ctx_id, without GPU = -1"
        )
        parser.add_argument("--image_dir", type=str, help="Image directory or file")
        return parser

    def _build_embedding_model(self):

        model = insightface.model_zoo.get_model(self.args.model_name)
        model.prepare(self.args.ctx_id, self.args.batch_size)
        return model

    def _image_transform(self, path):
        img = Image.open(path)
        im = cv2.resize(
            img, self.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR
        )
        return im

    def get_embedding_features(self, image_paths):
        img_list = []

        for image_path in image_paths:
            im = self._image_transform(image_path["file_path"])
            img_list.append(im)

        return self.embedding_model.get_embedding_batch(img_list)

    def _chunks(self, array, chunk_size):
        for i in range(0, len(array), chunk_size):
            yield array[i: i + chunk_size]

    def _save_feature(self, file_name, feature):
        info = []
        file_base_name = str(file_name).split(".")[0]
        info["image_id"] = file_base_name
        info["features"] = feature.cpu().numpy()
        file_base_name = str(file_base_name) + ".npy"

        np.save(os.path.join(self.args.output_folder, file_base_name), info)

    def extract_features(self):
        image_dir = self.args.image_dir
        if os.path.isfile(image_dir):
            features = self.get_embedding_features([image_dir])
            self._save_feature(image_dir, features[0])
        else:
            files = glob.glob(os.path.join(image_dir, "*"))
            for chunk in self._chunks(files, self.args.batch_size):
                try:
                    features = self.get_embedding_features(chunk)
                    for idx, file_name in enumerate(chunk):
                        self._save_feature(file_name, features[idx])
                except BaseException:
                    continue

if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    feature_extractor.extract_features()
