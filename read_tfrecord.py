import tensorflow as tf
import matplotlib.pyplot as plt
import os
tfrecord_file = './train/train.tfrecords'
raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
feature_description = { # 定義Feature結構，告訴解碼器每個Feature的類型是什麼
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}

def _parse_example(example_string): # 將 TFRecord 文件中的每一個序列化的 tf.train.Example 解碼
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image']) # 解碼JPEG圖片
    return feature_dict['image'], feature_dict['label']

dataset = raw_dataset.map(_parse_example)
for image, label in dataset:
    plt.title('dog' if label == 1 else 'cat')
    plt.imshow(image.numpy())
    plt.show()