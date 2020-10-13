import tensorflow as tf
import os

num_epochs = 5
batch_size = 12
learning_rate = 0.001
data_dir = '.'
train_cats_dir = data_dir + '/train/cats/'
train_dogs_dir = data_dir + '/train/dogs/'
test_cats_dir = data_dir + '/valid/cats/'
test_dogs_dir = data_dir + '/valid/dogs/'

feature_description = { # 定義Feature結構，告訴解碼器每個Feature的類型是什麼
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}

def _decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)            # 讀取原始文件
    image_decoded = tf.image.decode_jpeg(image_string)  # 解碼JPEG圖片
    image_resized = tf.image.resize(image_decoded, [256, 256]) / 255.0
    return image_resized, label

def _parse_example(example_string): # 將 TFRecord 文件中的每一個序列化的 tf.train.Example 解碼
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image']) # 解碼JPEG圖片
    feature_dict['image'] = tf.image.resize(feature_dict['image'], [256, 256]) / 255.0
    return feature_dict['image'], feature_dict['label']

if __name__ == '__main__':
    # 建構訓練資料集
    # train_cat_filenames = tf.constant([train_cats_dir + filename for filename in os.listdir(train_cats_dir)])
    # train_dog_filenames = tf.constant([train_dogs_dir + filename for filename in os.listdir(train_dogs_dir)])
    # train_filenames = tf.concat([train_cat_filenames, train_dog_filenames], axis=-1)
    # train_labels = tf.concat([
    #     tf.zeros(train_cat_filenames.shape, dtype=tf.int32), 
    #     tf.ones(train_dog_filenames.shape, dtype=tf.int32)], 
    #     axis=-1)

    # train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    # train_dataset = train_dataset.map(
    #     map_func=_decode_and_resize, 
    #     num_parallel_calls=tf.data.experimental.AUTOTUNE)

    tfrecord_file = './train/train.tfrecords'
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    train_dataset = raw_dataset.map(_parse_example)
    # 取出前buffer_size個資料放入buffer，並從其中隨機取樣，取樣後的資料用後續資料替換
    train_dataset = train_dataset.shuffle(buffer_size=5000)    
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 5, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

    model.fit(train_dataset, epochs=num_epochs)
    # save model
    tf.saved_model.save(model, "models/")