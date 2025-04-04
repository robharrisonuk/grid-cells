# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Minimal queue based TFRecord reader for the Grid Cell paper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import tensorflow as tf
import trajectory
import numpy as np

nest = tf.contrib.framework.nest

DatasetInfo = collections.namedtuple(
    'DatasetInfo', ['basepath', 'size', 'sequence_length', 'coord_range'])

_DATASETS = dict(
    square_room=DatasetInfo(
        basepath='square_room_100steps_2.2m_1000000',
        size=100,
        sequence_length=100,
        coord_range=((-1.1, 1.1), (-1.1, 1.1))),)


def _get_dataset_files(dateset_info, root):
    """Generates lists of files for a given dataset version."""
    basepath = dateset_info.basepath
    base = os.path.join(root, basepath)
    num_files = dateset_info.size
    template = '{:0%d}-of-{:0%d}.tfrecord' % (4, 4)
    return [
        os.path.join(base, template.format(i, num_files - 1))
            for i in range(num_files)]


class DataReader(object):
    """Minimal queue based TFRecord reader.

    You can use this reader to load the datasets used to train the grid cell
    network in the 'Vector-based Navigation using Grid-like Representations
    in Artificial Agents' paper.
    See README.md for a description of the datasets and an example of how to use
    the reader.
    """

    def __init__(
        self,
        dataset,
        root,
        # Queue params
        num_threads=4,
        capacity=256,
        min_after_dequeue=128,
        seed=None):
        """Instantiates a DataReader object and sets up queues for data reading.

        Args:
          dataset: string, one of ['jaco', 'mazes', 'rooms_ring_camera',
            'rooms_free_camera_no_object_rotations',
            'rooms_free_camera_with_object_rotations', 'shepard_metzler_5_parts',
            'shepard_metzler_7_parts'].
          root: string, path to the root folder of the data.
          num_threads: (optional) integer, number of threads used to feed the reader
            queues, defaults to 4.
          capacity: (optional) integer, capacity of the underlying
            RandomShuffleQueue, defaults to 256.
          min_after_dequeue: (optional) integer, min_after_dequeue of the underlying
            RandomShuffleQueue, defaults to 128.
          seed: (optional) integer, seed for the random number generators used in
            the reader.

        Raises:
          ValueError: if the required version does not exist;
        """

        if dataset not in _DATASETS:
            raise ValueError('Unrecognized dataset {} requested. Available datasets '
                       'are {}'.format(dataset, _DATASETS.keys()))

        self._dataset_info = _DATASETS[dataset]
        self._steps = _DATASETS[dataset].sequence_length

        with tf.device('/cpu'):
            file_names = _get_dataset_files(self._dataset_info, root)
            filename_queue = tf.train.string_input_producer(file_names, seed=seed)
            reader = tf.TFRecordReader()

            read_ops = [
                self._make_read_op(reader, filename_queue) for _ in range(num_threads)
            ]
            dtypes = nest.map_structure(lambda x: x.dtype, read_ops[0])
            shapes = nest.map_structure(lambda x: x.shape[1:], read_ops[0])

            self._queue = tf.RandomShuffleQueue(
              capacity=capacity,
              min_after_dequeue=min_after_dequeue,
              dtypes=dtypes,
              shapes=shapes,
              seed=seed)

            enqueue_ops = [self._queue.enqueue_many(op) for op in read_ops]
            tf.train.add_queue_runner(tf.train.QueueRunner(self._queue, enqueue_ops))

    def read(self, batch_size):
        """Reads batch_size."""
        in_pos, in_hd, ego_vel, target_pos, target_hd = self._queue.dequeue_many(batch_size)
        return in_pos, in_hd, ego_vel, target_pos, target_hd

    def get_coord_range(self):
        return self._dataset_info.coord_range

    def _make_read_op(self, reader, filename_queue):
        """Instantiates the ops used to read and parse the data into tensors."""
        _, raw_data = reader.read_up_to(filename_queue, num_records=64)
        feature_map = {
            'init_pos':
                tf.FixedLenFeature(shape=[2], dtype=tf.float32),
            'init_hd':
                tf.FixedLenFeature(shape=[1], dtype=tf.float32),
            'ego_vel':
                tf.FixedLenFeature(
                    shape=[self._dataset_info.sequence_length, 3],
                    dtype=tf.float32),
            'target_pos':
                tf.FixedLenFeature(
                    shape=[self._dataset_info.sequence_length, 2],
                    dtype=tf.float32),
            'target_hd':
                tf.FixedLenFeature(
                    shape=[self._dataset_info.sequence_length, 1],
                    dtype=tf.float32),
        }
        example = tf.parse_example(raw_data, feature_map)
        batch = [
            example['init_pos'], example['init_hd'],
            example['ego_vel'][:, :self._steps, :],
            example['target_pos'][:, :self._steps, :],
            example['target_hd'][:, :self._steps, :]
        ]
        return batch


# ---------------------------------------------------------------------------------

def CreateArtificialTFRecords(dataset, root):
    with tf.device('/cpu'):
        dataset_info = _DATASETS[dataset]
        file_names = _get_dataset_files(dataset_info, root)

        sequence_length = dataset_info.sequence_length

        for f in file_names:
            with tf.io.TFRecordWriter(f) as writer:
                for i in range(100):
                    traj, _ = trajectory.trajectory_builder(sequence_length, 1, -1.0, 1.0, -1.0, 1.0)

                    init_pos = traj['init_pos'].reshape(2)
                    init_hd = traj['init_hd'].reshape(1)
                    ego_vel = traj['ego_vel'].reshape(-1)
                    target_pos = traj['target_pos'].reshape(-1)
                    target_hd = traj['target_hd'].reshape(-1)

                    feature = {
                        'init_pos': tf.train.Feature(float_list=tf.train.FloatList(value=init_pos)),
                        'init_hd': tf.train.Feature(float_list=tf.train.FloatList(value=init_hd)),
                        'ego_vel': tf.train.Feature(float_list=tf.train.FloatList(value=ego_vel)),
                        'target_pos': tf.train.Feature(float_list=tf.train.FloatList(value=target_pos)),
                        'target_hd': tf.train.Feature(float_list=tf.train.FloatList(value=target_hd))
                    }

                    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                    example_proto.SerializeToString()

                    writer.write(example_proto.SerializeToString())



# ---------------------------------------------------------------------------------

# Read the data back out.
def decode_fn(record_bytes):

    return tf.io.parse_single_example(
        # Data
        record_bytes,
        # Schema
        {'init_pos':
                tf.FixedLenFeature(shape=[2], dtype=tf.float32),
            'init_hd':
                tf.FixedLenFeature(shape=[1], dtype=tf.float32),
            'ego_vel':
                tf.FixedLenFeature(
                    shape=[100, 3],
                    dtype=tf.float32),
            'target_pos':
                tf.FixedLenFeature(
                    shape=[100, 2],
                    dtype=tf.float32),
            'target_hd':
                tf.FixedLenFeature(
                    shape=[100, 1],
                    dtype=tf.float32),
         }
    )

# ---------------------------------------------------------------------------------

def ConvertDataSetToNumpy(dataset, root, export_folder):

    dataset_info = _DATASETS[dataset]
    file_names = _get_dataset_files(dataset_info, root)
    seq_len = dataset_info.sequence_length

    tf.compat.v1.enable_eager_execution()

    for file_name in file_names:

        base_name = os.path.splitext(os.path.basename(file_name))[0]
        export_file_name = os.path.join(export_folder, base_name+".npz")
        print(f'Exporting: {file_name} -> {export_file_name}')

        num_examples = 0
        for _ in tf.data.TFRecordDataset([file_name]).map(decode_fn):
            num_examples += 1

        init_pos = np.empty([num_examples, 2], dtype=np.float32)
        init_hd = np.empty([num_examples, 1], dtype=np.float32)
        ego_vel = np.empty([num_examples, seq_len, 3], dtype=np.float32)
        target_pos = np.empty([num_examples, seq_len, 2], dtype=np.float32)
        target_hd = np.empty([num_examples, seq_len, 1], dtype=np.float32)

        idx = 0
        for batch in tf.data.TFRecordDataset([file_name]).map(decode_fn):
            init_pos[idx] = batch['init_pos'].numpy()
            init_hd[idx] = batch['init_hd'].numpy()
            ego_vel[idx] = batch['ego_vel'].numpy()
            target_pos[idx] = batch['target_pos'].numpy()
            target_hd[idx] = batch['target_hd'].numpy()
            idx += 1

        np.savez_compressed(export_file_name,
                            init_pos=init_pos,
                            init_hd=init_hd,
                            ego_vel=ego_vel,
                            target_pos=target_pos,
                            target_hd=target_hd)



