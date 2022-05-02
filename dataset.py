import tensorflow as tf

# for testing hyperparameters

DENSE_FEATURES = 30
SEQ_FEATURES = 4
LABELS = 15

N_BUCKET = 1000
N_ID = 1000
N_BATCH_SIZE = 5 

N_DIM = 160

# for production hyperparameters
"""
DENSE_FEATURES = 30
SEQ_FEATURES = 4
LABELS = 15

N_BUCKET = 10000
N_ID = 1000000000   # 1B for each, which are the most memory consuming features.
N_BATCH_SIZE = 8192

N_DIM = 160
"""


def build_dataset():
  def map_fn(record):
    features = {}
    # bucketized dense feature
    for i in range(DENSE_FEATURES):
      features['dense_%d' % i] = tf.random.uniform(shape=[], minval=0, maxval=N_BUCKET, dtype=tf.int32)
    
    # sequence feature
    features['seq_0'] = tf.random.uniform(shape=[100], minval=0, maxval=N_ID, dtype=tf.int32)
    features['seq_1'] = tf.random.uniform(shape=[50], minval=0, maxval=N_ID, dtype=tf.int32)
    features['seq_2'] = tf.random.uniform(shape=[20], minval=0, maxval=N_ID, dtype=tf.int32)
    features['seq_3'] = tf.random.uniform(shape=[20], minval=0, maxval=N_ID, dtype=tf.int32)

    # label
    for i in range(LABELS):
      features['id_%d' % i] = tf.random.uniform(shape=[], minval=0, maxval=N_ID, dtype=tf.int32)
      features['label_%d' % i] = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)

    return features, 1

  dataset = tf.data.Dataset.from_tensor_slices(tf.zeros([1000000, 1]))
  dataset = dataset.repeat(10000)
  dataset = dataset.map(map_fn, num_parallel_calls=32)
  dataset = dataset.batch(N_BATCH_SIZE)
  return dataset


if __name__ == '__main__':
  dataset = build_dataset()

  iterator = dataset.make_one_shot_iterator()
  next_element = iterator.get_next()
  with tf.Session() as sess:
    for i in range(10):
      batch = sess.run(next_element)
      print(batch)
