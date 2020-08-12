import pandas as pd
import tensorflow as tf

# I solved it by adding "tfrecord:" in path, folllow the pattern of their TFDS examples

save_path = "pegasus/data/testdata/test_pattern_1.tfrecord"

file = open("cnn.txt", "r")
article = file.read()
file.close()

input_dict = dict(
                  inputs=[article],
                  targets=[""]
                 )

data = pd.DataFrame(input_dict)

with tf.io.TFRecordWriter(save_path) as writer:
    for row in data.values:
        inputs, targets = row[:-1], row[-1]
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "inputs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[inputs[0].encode('utf-8')])),
                    "targets": tf.train.Feature(bytes_list=tf.train.BytesList(value=[targets.encode('utf-8')])),
                }
            )
        )
        writer.write(example.SerializeToString())