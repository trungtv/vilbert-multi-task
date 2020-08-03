# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Requires vqa-maskrcnn-benchmark to be built and installed. See Readme
# for more details.
import argparse
import os
import insightface
from script.face_analysis import FaceAnalysis

import face_alignment
import cv2
import numpy as np
import torch
from PIL import Image
import glob


class VideoRecord(object):
    def __init__(self, video_path, embedding_and_landmark_fn):
        self.video_path = video_path
        capture = cv2.VideoCapture(video_path)

        success = True
        count = 0
        self.embeddings = list()
        self.landmarks = list()
        self.lip_landmarks = list()
        while success:
            success, frame = capture.read()
            if success:
                face_embedding, landmark, lip_landmark = embedding_and_landmark_fn(frame)
                if face_embedding is not None and landmark is not None and lip_landmark is not None:
                    self.embeddings.append(face_embedding)
                    self.landmarks.append(landmark)
                    self.lip_landmarks.append(lip_landmark)
                    count += 1
        capture.release()
        self.length = count
    @property
    def path(self):
        return self.video_path

    @property
    def features(self):
        return self.embeddings, self.landmarks, self.lip_landmarks

    @property
    def num_frames(self):
        return self.length


class FeatureExtractor:
    IMAGE_SIZE = (122, 122)

    def __init__(self):
        self.args = self.get_parser().parse_args()
        self.embedding_model = self._build_embeddign_model()
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
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
        parser.add_argument("--video_dir", type=str, help="video directory")
        return parser

    def _build_embedding_model(self):
        model = FaceAnalysis(det_name='retinaface_r50_v1', rec_name='arcface_r100_v1', ga_name=None)
        # model = insightface.model_zoo.get_model(self.args.model_name)
        model.prepare(self.args.ctx_id, self.args.batch_size)
        return model

    def get_embedding_and_lib_landmark(self, img):
        faces = self.embedding_model.get(img)
        if len(faces) != 1:
            return None, None
        preds = self.fa.get_landmarks(faces[0].img)
        return faces[0].embedding, preds[0].flatten(), preds[0][48:].flatten()

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

    def _save_feature(self, file_name, embeddings, landmarks, lip_landmarks):
        info = []
        file_base_name = str(file_name).split(".")[0]
        info["video_id"] = file_base_name
        info["embeddings"] = embeddings.cpu().numpy()
        info["landmarks"] = landmarks.cpu().numpy()
        info["lip_landmarks"] = lip_landmarks.cpu().numpy()
        file_base_name = str(file_base_name) + ".npy"
        file_path = os.path.join(self.args.output_folder, file_base_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, info)

    def extract_features(self):
        video_dir = self.args.video_dir
        files = glob.glob(os.path.join(video_dir, "*"))
        for file in files:
            try:
                video_record = VideoRecord(file, self.get_embedding_and_lib_landmark)
                embeddings, landmarks, lip_landmarks = video_record.features
                file_name = file.replace(video_dir, '')
                self._save_feature(file_name, embeddings, landmarks, lip_landmarks)
            except BaseException:
                continue

if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    feature_extractor.extract_features()
