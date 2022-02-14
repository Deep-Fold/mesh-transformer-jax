

import tensorflow as tf

import sys

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        raw_dataset = tf.data.TFRecordDataset(sys.argv[1])

        for raw_record in raw_dataset.take(1):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            print(example)