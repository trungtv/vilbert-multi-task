import os
import numpy as np
# from tensorpack.dataflow import RNGDataFlow, PrefetchDataZMQ
from tensorpack.dataflow import *
import lmdb
import json
import pdb
import csv
FIELDNAMES = ['video_id', 'num_frames', 'features']
import sys
import pandas as pd
import zlib
import base64

csv.field_size_limit(sys.maxsize)


def open_tsv(fname, folder):
    print("Opening %s Data File..." % fname)
    df = pd.read_csv(fname, sep='\t', names=["caption","url"], usecols=range(0,2))
    df['folder'] = folder
    print("Processing", len(df), " Images:")
    return df
def _file_name(row):
    return "%s/%s" % (row['folder'], (zlib.crc32(row['url'].encode('utf-8')) & 0xffffffff))

class FaceReadingCaption(RNGDataFlow):
    """
    """
    def __init__(self, corpus_path, shuffle=False):
        """
        Same as in :class:`ILSVRC12`.
        """
        self.shuffle = shuffle
        self.num_file = 30
        self.name = os.path.join(corpus_path, 'conceptual_caption_trainsubset_resnet101_faster_rcnn_genome.tsv.%d')
        self.infiles = [self.name % i for i in range(self.num_file)]
        self.counts = []
        self.num_caps = 3136161

        self.captions = {}
        df = open_tsv('/srv/share/datasets/conceptual_caption/DownloadConceptualCaptions/Train_GCC-training.tsv', 'training')
        for i, img in enumerate(df.iterrows()):
            caption = img[1]['caption']#.decode("utf8")
            url = img[1]['url']
            im_name = _file_name(img[1])
            image_id = im_name.split('/')[1]
            self.captions[image_id] = caption

    def __len__(self):
        return self.num_caps

    def __iter__(self):
        for infile in self.infiles:
            count = 0
            with open(infile) as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
                for item in reader:
                    video_id = item['video_id']
                    num_frames = item['num_frames']
                    features = np.frombuffer(base64.b64decode(item['features']), dtype=np.float32).reshape(int(num_frames), 2048)
                    face_landmarks = np.frombuffer(base64.b64decode(item['face_landmarks']), dtype=np.float32).reshape(
                        int(num_frames), 2048)
                    caption = self.captions[video_id]

                    yield [features, face_landmarks, num_frames, video_id, caption]

if __name__ == '__main__':
    corpus_path = '/srv/share2/jlu347/bottom-up-attention/feature/conceptual_caption'
    ds = FaceReadingCaption(corpus_path)
    ds1 = PrefetchDataZMQ(ds, nr_proc=1)
    LMDBSerializer.save(ds1, 'training_feat_all.lmdb')
