# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import logging
import math
import os
import random

import lmdb
import numpy as np
import tensorpack.dataflow as td

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import sys
import pdb

import msgpack
import msgpack_numpy

msgpack_numpy.patch()

MAX_MSGPACK_LEN = 1000000000
MAX_VIDEO_LEN = 75

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def deserialize_lmdb(ds):
    return msgpack.loads(
        ds[1],
        raw=False,
        max_bin_len=MAX_MSGPACK_LEN,
        max_array_len=MAX_MSGPACK_LEN,
        max_map_len=MAX_MSGPACK_LEN,
        max_str_len=MAX_MSGPACK_LEN,
    )


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(
        self,
        image_feat=None,
        image_target=None,  # Could be facial landmark
        caption=None,
        is_next=None,
        num_frames=None,
        lm_labels=None,
    ):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.image_feat = image_feat
        self.caption = caption
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model
        self.image_target = image_target
        self.num_frames = num_frames

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids=None,
        input_mask=None,
        segment_ids=None,
        is_next=None,
        lm_label_ids=None,
        image_feat=None,
        image_target=None,
        image_loc=None,
        image_label=None,
        image_mask=None,
        masked_label=None,
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids
        self.image_feat = image_feat
        self.image_loc = image_loc
        self.image_label = image_label
        self.image_target = image_target
        self.image_mask = image_mask
        self.masked_label = masked_label


class FaceReadingLoaderTrain(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the 
            GPU, which is faster).
    """

    def __init__(
        self,
        corpus_path,
        tokenizer,
        bert_model,
        seq_len,
        encoding="utf-8",
        visual_target=0,
        hard_negative=False,
        batch_size=512,
        shuffle=False,
        num_workers=25,
        cache=10000,
        drop_last=False,
        cuda=False,
        local_rank=-1,
        objective=0,
        visualization=False,
    ):
        TRAIN_DATASET_SIZE = 3119449

        if dist.is_available() and local_rank != -1:

            num_replicas = dist.get_world_size()
            rank = dist.get_rank()

            lmdb_file = os.path.join(
                corpus_path, "training_feat_part_" + str(rank) + ".lmdb"
            )
        else:
            lmdb_file = os.path.join(corpus_path, "training_feat_all.lmdb")
            # lmdb_file = os.path.join(corpus_path, "validation_feat_all.lmdb")

            print("Loading from %s" % lmdb_file)

        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = len(ds)
        ds = td.LocallyShuffleData(ds, cache)
        caption_path = os.path.join(corpus_path, "caption_train.json")
        # caption_path = os.path.join(corpus_path, "caption_val.json")

        preprocess_function = BertPreprocessBatch(
            caption_path,
            tokenizer,
            bert_model,
            seq_len,
            MAX_VIDEO_LEN,
            self.num_dataset,
            encoding="utf-8",
            visual_target=visual_target,
            objective=objective,
        )

        ds = td.PrefetchData(ds, 5000, 1)
        ds = td.MapData(ds, preprocess_function)
        # self.ds = td.PrefetchData(ds, 1)
        ds = td.PrefetchDataZMQ(ds, num_workers)
        self.ds = td.BatchData(ds, batch_size)
        # self.ds = ds
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers

    def __iter__(self):

        for batch in self.ds.get_data():
            input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_target, image_label, image_mask, video_id = (
                batch
            )

            batch_size = input_ids.shape[0]
            image_feat = np.array(image_feat, dtype=np.float32)
            g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            batch = (
                input_ids,
                input_mask,
                segment_ids,
                lm_label_ids,
                is_next,
                image_feat,
                image_target,
                image_label,
                image_mask,
            )

            yield tuple([torch.tensor(data) for data in batch] + [video_id])

    def __len__(self):
        return self.ds.size()


class FaceReadingLoaderVal(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the 
            GPU, which is faster).
    """

    def __init__(
        self,
        corpus_path,
        tokenizer,
        bert_model,
        seq_len,
        encoding="utf-8",
        visual_target=0,
        batch_size=512,
        shuffle=False,
        num_workers=25,
        cache=5000,
        drop_last=False,
        cuda=False,
        objective=0,
        visualization=False,
    ):

        lmdb_file = os.path.join(corpus_path, "validation_feat_all.lmdb")
        caption_path = os.path.join(corpus_path, "caption_val.json")
        print("Loading from %s" % lmdb_file)

        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = len(ds)
        preprocess_function = BertPreprocessBatch(
            caption_path,
            tokenizer,
            bert_model,
            seq_len,
            MAX_VIDEO_LEN,
            self.num_dataset,
            encoding="utf-8",
            visual_target=visual_target,
            visualization=visualization,
            objective=objective,
        )

        ds = td.MapData(ds, preprocess_function)
        self.ds = td.BatchData(ds, batch_size)
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers

    def __iter__(self):
        for batch in self.ds.get_data():
            input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc, image_target, image_label, image_mask, masked_label, image_id = (
                batch
            )

            batch_size = input_ids.shape[0]
            sum_count = np.sum(masked_label == 0, axis=1, keepdims=True)
            sum_count[sum_count == 0] = 1
            g_image_feat = np.sum(image_feat, axis=1) / sum_count
            image_feat = np.concatenate(
                [np.expand_dims(g_image_feat, axis=1), image_feat], axis=1
            )
            image_feat = np.array(image_feat, dtype=np.float32)

            g_image_loc = np.repeat(
                np.array([[0, 0, 1, 1, 1]], dtype=np.float32), batch_size, axis=0
            )
            image_loc = np.concatenate(
                [np.expand_dims(g_image_loc, axis=1), image_loc], axis=1
            )

            image_loc = np.array(image_loc, dtype=np.float32)
            g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            batch = (
                input_ids,
                input_mask,
                segment_ids,
                lm_label_ids,
                is_next,
                image_feat,
                image_loc,
                image_target,
                image_label,
                image_mask,
            )

            yield tuple([torch.tensor(data) for data in batch] + [image_id])

    def __len__(self):
        return self.ds.size()


class BertPreprocessBatch(object):
    def __init__(
        self,
        caption_path,
        tokenizer,
        bert_model,
        seq_len,
        video_len,
        data_size,  # number of training examples
        split="Train",
        encoding="utf-8",
        visual_target=0,
        visualization=False,
        objective=0,
    ):

        self.split = split
        self.seq_len = seq_len
        self.video_len = video_len
        self.tokenizer = tokenizer
        self.visual_target = visual_target
        self.num_caps = data_size
        self.captions = list(json.load(open(caption_path, "r")).values())
        self.visualization = visualization
        self.objective = objective
        self.bert_model = bert_model

    def __call__(self, data):
        # image_target_wp can be used to capture facial landmark
        image_feature_wp, image_target_wp, n_frames, video_id, caption = (
            data
        )

        image_feature = np.zeros((self.video_len, 2048), dtype=np.float32)
        image_target = np.zeros((self.video_len, 1601), dtype=np.float32)

        image_feature[:n_frames] = image_feature_wp
        image_target[:n_frames] = image_target_wp

        # visual target can be used for facial landmark
        if self.visual_target == 0:
            image_feature = copy.deepcopy(image_feature)
            image_target = copy.deepcopy(image_target)
        else:
            image_feature = copy.deepcopy(image_feature)
            image_target = copy.deepcopy(image_feature)

        caption, label = self.random_cap(caption)

        tokens_caption = self.tokenizer.encode(caption)

        cur_example = InputExample(
            image_feat=image_feature,
            image_target=image_target,
            caption=tokens_caption,
            is_next=label,
            num_frames=n_frames
        )

        # transform sample to features
        cur_features = self.convert_example_to_features(
            cur_example, self.seq_len, self.tokenizer, self.video_len
        )

        cur_tensors = (
            cur_features.input_ids,
            cur_features.input_mask,
            cur_features.segment_ids,
            cur_features.lm_label_ids,
            cur_features.is_next,
            cur_features.image_feat,
            cur_features.image_target,
            cur_features.image_label,
            cur_features.image_mask,
            video_id,
        )

        return cur_tensors

    def random_cap(self, caption):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """

        if self.visualization:
            return caption, 0

        if self.objective != 2 and random.random() > 0.5:
            caption = self.get_random_caption()
            label = 1
        else:
            label = 0

        return caption, label

    def get_random_caption(self):
        """
        Get random caption from another document for nextSentence task.
        :return: str, content of one line
        """
        # add the hard negative mining objective here.
        rand_doc_idx = random.randint(0, self.num_caps - 1)
        caption = self.captions[rand_doc_idx]

        return caption

    def convert_example_to_features(
        self, example, max_seq_length, tokenizer, max_video_length
    ):
        """
        """
        image_feat = example.image_feat
        tokens = example.caption
        image_loc = example.image_loc
        image_target = example.image_target
        num_frames = int(example.num_frames)
        is_next = example.is_next
        overlaps = example.overlaps

        self._truncate_seq_pair(tokens, max_seq_length - 2)

        tokens, tokens_label = self.random_word(tokens, tokenizer, is_next)
        image_feat, image_label = self.random_frame(
            image_feat, num_frames, is_next
        )

        # concatenate lm labels and account for CLS, SEP, SEP
        lm_label_ids = [-1] + tokens_label + [-1]
        tokens = tokenizer.add_special_tokens_single_sentence(tokens)
        segment_ids = [0] * len(tokens)

        input_ids = tokens  # tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # input_ids = input_ids[:1] input_ids[1:]
        input_mask = [1] * (len(input_ids))
        image_mask = [1] * (num_frames)
        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_video_length:
            image_mask.append(0)
            image_label.append(-1)  # masking 1 to be predicted, no masking not to be predicted -1

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_label_ids) == max_seq_length
        assert len(image_mask) == max_video_length
        assert len(image_label) == max_video_length

        features = InputFeatures(
            input_ids=np.array(input_ids),
            input_mask=np.array(input_mask),
            segment_ids=np.array(segment_ids),
            lm_label_ids=np.array(lm_label_ids),
            is_next=np.array(example.is_next),
            image_feat=image_feat,
            image_target=image_target,
            image_label=np.array(image_label),
            image_mask=np.array(image_mask),
        )
        return features

    def _truncate_seq_pair(self, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_b)
            if total_length <= max_length:
                break

            tokens_b.pop()

    def random_word(self, tokens, tokenizer, is_next):
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability

            # if is_next == 1 and self.objective != 0:
            #     prob = 1 # not sample mask
            if prob < 0.15 and (not self.visualization):
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = np.random.randint(len(tokenizer))
                    # torch.randint(len(tokenizer), labels.shape, dtype=torch.long)

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(token)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return tokens, output_label

    def random_frame(self, image_feat, num_frames, is_next):
        """
        """
        output_label = []

        for i in range(num_frames):
            prob = random.random()
            # mask token with 15% probability

            # if is_next == 1 and self.objective != 0:
            #     prob = 1 # if the target is inaligned mask, then not sample mask
            if prob < 0.15 and not self.visualization:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.9:
                    image_feat[i] = 0

                # 10% randomly change token to random token
                # elif prob < 0.9:
                # tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(1)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return image_feat, output_label


class FaceReadingLoaderRetrieval(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the 
            GPU, which is faster).
    """

    def __init__(
        self,
        corpus_path,
        tokenizer,
        seq_len,
        encoding="utf-8",
        visual_target=0,
        batch_size=512,
        shuffle=False,
        num_workers=10,
        cache=50000,
        drop_last=False,
        cuda=False,
    ):

        lmdb_file = "/coc/dataset/conceptual_caption/validation_feat_all.lmdb"
        if not os.path.exists(lmdb_file):
            lmdb_file = "/coc/pskynet2/jlu347/multi-modal-bert/data/conceptual_caption/validation_feat_all.lmdb"
        caption_path = "/coc/pskynet2/jlu347/multi-modal-bert/data/conceptual_caption/caption_val.json"

        print("Loading from %s" % lmdb_file)

        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = len(ds)
        preprocess_function = BertPreprocessRetrieval(
            caption_path,
            tokenizer,
            seq_len,
            MAX_VIDEO_LEN,
            1000,
            encoding="utf-8",
            visual_target=visual_target,
        )

        ds = td.MapData(ds, preprocess_function)
        self.ds = td.BatchData(ds, 1)
        self.ds.reset_state()

        self.batch_size = 1
        self.num_workers = num_workers
        self._entry = []

        self.features_all = np.zeros((1000, 37, 2048), dtype=np.float32)
        self.spatials_all = np.zeros((1000, 37, 5), dtype=np.float32)
        self.image_mask_all = np.zeros((1000, 37), dtype=np.float32)
        self.image_ids = []
        # load first 1000 file here.
        for i, batch in enumerate(self.ds.get_data()):
            if i >= 1000:
                break
            input_ids, input_mask, segment_ids, is_next, image_feat, image_loc, image_mask, image_id, caption = (
                batch
            )

            batch_size = input_ids.shape[0]
            g_image_feat = np.sum(image_feat, axis=1) / np.sum(
                image_mask, axis=1, keepdims=True
            )
            image_feat = np.concatenate(
                [np.expand_dims(g_image_feat, axis=1), image_feat], axis=1
            )
            image_feat = np.array(image_feat, dtype=np.float32)

            g_image_loc = np.repeat(
                np.array([[0, 0, 1, 1, 1]], dtype=np.float32), batch_size, axis=0
            )
            image_loc = np.concatenate(
                [np.expand_dims(g_image_loc, axis=1), image_loc], axis=1
            )

            image_loc = np.array(image_loc, dtype=np.float32)
            g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            batch = (input_ids, input_mask, segment_ids, image_id, caption)
            self._entry.append(batch)

            self.features_all[i] = image_feat
            self.image_mask_all[i] = np.array(image_mask)
            self.spatials_all[i] = image_loc
            self.image_ids.append(image_id)
            sys.stdout.write("%d/%d\r" % (i, 1000))
            sys.stdout.flush()

    def __iter__(self):

        for index in range(self.__len__()):
            caption_idx = int(index / 2)
            image_idx = index % 2

            if image_idx == 0:
                image_entries = self.image_ids[:500]
                features_all = self.features_all[:500]
                spatials_all = self.spatials_all[:500]
                image_mask_all = self.image_mask_all[:500]

            else:
                image_entries = self.image_ids[500:]
                features_all = self.features_all[500:]
                spatials_all = self.spatials_all[500:]
                image_mask_all = self.image_mask_all[500:]

            caption, input_mask, segment_ids, txt_image_id, caption = self._entry[
                caption_idx
            ]
            target_all = np.zeros((500))
            for i, image_id in enumerate(image_entries):
                if image_id == txt_image_id:
                    target_all[i] = 1

            batch = (
                features_all,
                spatials_all,
                image_mask_all,
                caption,
                input_mask,
                segment_ids,
                target_all,
                caption_idx,
                image_idx,
            )
            batch = [torch.tensor(data) for data in batch]
            batch.append(txt_image_id)
            batch.append(caption)

            yield batch

    def __len__(self):
        return len(self._entry) * 2


class BertPreprocessRetrieval(object):
    def __init__(
        self,
        caption_path,
        tokenizer,
        seq_len,
        video_len,
        data_size,
        split="Train",
        encoding="utf-8",
        visual_target=0,
    ):

        self.split = split
        self.seq_len = seq_len
        self.video_len = video_len
        self.tokenizer = tokenizer
        self.visual_target = visual_target
        self.num_caps = data_size
        self.captions = list(json.load(open(caption_path, "r")).values())[:data_size]

    def __call__(self, data):

        image_feature_wp, image_target_wp, num_frames, video_id, caption = (
            data
        )

        image_feature = np.zeros((self.video_len, 2048), dtype=np.float32)
        image_target = np.zeros((self.video_len, 1601), dtype=np.float32)

        num_boxes = int(num_frames)
        image_feature[:num_frames] = image_feature_wp
        image_target[:num_frames] = image_target_wp

        label = 0

        tokens_caption = self.tokenizer.tokenize(caption)
        cur_example = InputExample(
            image_feat=image_feature,
            image_target=image_target,
            caption=tokens_caption,
            is_next=label,
            num_frames=num_frames,
        )

        # transform sample to features
        cur_features = self.convert_example_to_features(
            cur_example, self.seq_len, self.tokenizer, self.video_len
        )

        cur_tensors = (
            cur_features.input_ids,
            cur_features.input_mask,
            cur_features.segment_ids,
            cur_features.is_next,
            cur_features.image_feat,
            cur_features.image_mask,
            float(video_id),
            caption,
        )
        return cur_tensors

    def convert_example_to_features(
        self, example, max_seq_length, tokenizer, max_video_length
    ):
        """
        Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
        IDs, LM labels, input_mask, CLS and SEP tokens etc.
        :param example: InputExample, containing sentence input as strings and is_next label
        :param max_seq_length: int, maximum length of sequence.
        :param tokenizer: Tokenizer
        :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
        """
        image_feat = example.image_feat
        caption = example.caption
        image_loc = example.image_loc
        # image_target = example.image_target
        num_frames = int(example.num_frames)
        self._truncate_seq_pair(caption, max_seq_length - 2)
        caption, caption_label = self.random_word(caption, tokenizer)
        caption_label = None
        image_feat, image_loc, image_label, masked_label = self.random_frame(
            image_feat, image_loc, num_frames
        )
        image_label = None

        tokens = []
        segment_ids = []

        tokens.append("[CLS]")
        segment_ids.append(0)

        for token in caption:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # input_ids = input_ids[:1] input_ids[1:]
        input_mask = [1] * (len(input_ids))
        image_mask = [1] * (num_frames)
        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_video_length:
            image_mask.append(0)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(image_mask) == max_video_length

        features = InputFeatures(
            input_ids=np.array(input_ids),
            input_mask=np.array(input_mask),
            segment_ids=np.array(segment_ids),
            is_next=np.array(example.is_next),
            image_feat=image_feat,
            image_mask=np.array(image_mask),
        )
        return features

    def _truncate_seq_pair(self, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_b)
            if total_length <= max_length:
                break

            tokens_b.pop()
