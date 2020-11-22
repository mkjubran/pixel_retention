
# from tensorflow.compat import v1 as tf
import tensorflow as tf
import pdb


def pade_model(RETAINED_PIXELS):
    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10
    _BATCH_SIZE = 128

    with tf.variable_scope('pade_var') as pade_scope:
       with tf.name_scope('pade'):
          with tf.name_scope('main_params'):
              x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
              y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')

              input_drop=tf.random.uniform([_BATCH_SIZE, _IMAGE_SIZE, _IMAGE_SIZE, 1], minval=0, maxval=2, dtype=tf.dtypes.int32, seed=None, name=None) 
              input_drop=tf.cast(input_drop, dtype=tf.float32)
              x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')
              x_image = x_image * input_drop
              # pdb.set_trace()
              # tf.sparse_to_dense(sparse_indices=[[0, 0], [1, 2]],
              #                    output_shape=[3, 4],
              #                    default_value=0,
              #                    sparse_values=1,
              #                   )

              global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
              learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

          with tf.variable_scope('conv1') as scope:
              conv = tf.layers.conv2d(
                  inputs=x_image,
                  filters=32,
                  kernel_size=[3, 3],
                  padding='SAME',
                  activation=tf.nn.relu
              )
              conv = tf.layers.conv2d(
                  inputs=conv,
                  filters=64,
                  kernel_size=[3, 3],
                  padding='SAME',
                  activation=tf.nn.relu
              )
              pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
              drop = tf.layers.dropout(pool, rate=0.25, name=scope.name)

          with tf.variable_scope('conv2') as scope:
              conv = tf.layers.conv2d(
                  inputs=drop,
                  filters=128,
                  kernel_size=[3, 3],
                  padding='SAME',
                  activation=tf.nn.relu
              )
              # legacy: kernel_size=[3, 3],
              pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
              conv = tf.layers.conv2d(
                  inputs=pool,
                  filters=128,
                  kernel_size=[2, 2],
                  padding='SAME',
                  activation=tf.nn.relu
              )
              pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
              drop = tf.layers.dropout(pool, rate=0.25, name=scope.name)

          with tf.variable_scope('pade_layer') as scope:
              conv = tf.layers.conv2d(
                  inputs=drop,
                  filters=1,
                  kernel_size=[1, 1],
                  padding='SAME',
                  activation=tf.nn.relu
              )
              # pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
              # tf.image.resize(conv, [_IMAGE_SIZE, IMAGE_SIZE], method=ResizeMethod.BILINEAR, name='pade_map')
              # pade_output = tf.reshape(pool,[_BATCH_SIZE, _IMAGE_SIZE, _IMAGE_SIZE, 1], name=None)
              pade_output = tf.image.resize(conv, [_IMAGE_SIZE, _IMAGE_SIZE], name='pade_map')

              pade_tmp = tf.reshape(pade_output,[_BATCH_SIZE, _IMAGE_SIZE * _IMAGE_SIZE, 1], name=None)
              pade_tmp = tf.transpose(pade_tmp, perm=[0, 2,  1]) # shape: [_BATCH_SIZE, 1, SIZE*SIZE]

              # RETAINED_PIXELS = 512
              top_pade_values, top_pade_indices = tf.math.top_k(pade_tmp, k=RETAINED_PIXELS, sorted=True)
              # top_pade_values, top_pade_indices = tf.math.top_k(pade_tmp, k=64, sorted=True, name=None)


              lowest_top_k = tf.squeeze(tf.slice(top_pade_values, [0,0,int(top_pade_values.shape[2]-1)], [int(top_pade_values.shape[0]),1,1]))
              lowest_top_k_condition = tf.expand_dims(tf.stack([lowest_top_k]*_IMAGE_SIZE*_IMAGE_SIZE,axis=1), axis=1)
              pade_ge_condition = tf.greater_equal(pade_tmp, lowest_top_k_condition)
              pade_output = tf.reshape(pade_ge_condition,[_BATCH_SIZE, 1, _IMAGE_SIZE, _IMAGE_SIZE], name=None)
              # pdb.set_trace()
              pade_output = tf.transpose(pade_output, [0, 2, 3, 1])
              # top_k_condition = tf.expand_dims(tf.stack([tmp_]*_IMAGE_SIZE*_IMAGE_SIZE,axis=1), axis=1)
              # lowest_top_k_slice = tf.slice(top_pade_values, top_pade_values.shape[2])
              # tf.stack()
              # tf.where(tf.greater(pade_tmp, lowest_top_k_condition), a, b)
              # tf.sparse.SparseTensor(indices, values, dense_shape)
              pade_output = tf.cast(pade_output, dtype=tf.float32)
              tf.summary.image('pade_output', pade_output) # , step=None, max_outputs=3, description=None)

              # pdb.set_trace()
              tmp = 0


    return pade_output, x, y



def model(_RETAINED_PIXELS):
    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10
    _BATCH_SIZE = 128
    # _RETAINED_PIXELS = 512
    pade_output, pade_x, pade_y = pade_model(_RETAINED_PIXELS)
    # pade_output_256, pade_x_256, pade_y_256 = pade_model(256)
    # pade_output_512, pade_x_512, pade_y_512 = pade_model(512)

    with tf.name_scope('main_params'):
        x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')

        # shape = tf.placeholder(tf.int32, shape=[None])  # `shape` is a 1-D tensor.
        # random_t = tf.random_uniform(shape)
        # pdb.set_trace()
        # _batch_size = tf.placeholder(tf.int32, shape=[])
        # input_drop=tf.random.uniform(_batch_size, _IMAGE_SIZE, _IMAGE_SIZE, 1], minval=0, maxval=1, dtype=tf.dtypes.int32, seed=None, name=None) 
        # compensate for excluded maxval
        # input_drop=tf.random.uniform([_batch_size, _IMAGE_SIZE, _IMAGE_SIZE, 1], minval=0, maxval=2, dtype=tf.dtypes.int32, seed=None, name=None) 
        # input_drop=tf.random.uniform([_batch_size, _IMAGE_SIZE, _IMAGE_SIZE, 1], minval=0, maxval=2, dtype=tf.dtypes.int32, seed=None, name=None) 
        input_drop=tf.random.uniform([_BATCH_SIZE, _IMAGE_SIZE, _IMAGE_SIZE, 1], minval=0, maxval=2, dtype=tf.dtypes.int32, seed=None, name=None) 
        input_drop=  tf.cast(input_drop, dtype=tf.float32)
        pade_drop =  tf.cast(pade_output,  dtype=tf.float32)
        x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')


        use_pade = True
        if use_pade == False:
           x_image = x_image * input_drop
        else:
           x_image = x_image * pade_drop
        # pdb.set_trace()
        # tf.sparse_to_dense(sparse_indices=[[0, 0], [1, 2]],
        #                    output_shape=[3, 4],
        #                    default_value=0,
        #                    sparse_values=1,
        #                   )

        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    with tf.variable_scope('conv1') as scope:
        conv = tf.layers.conv2d(
            inputs=x_image,
            filters=32,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )
        conv = tf.layers.conv2d(
            inputs=conv,
            filters=64,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )
        pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        drop = tf.layers.dropout(pool, rate=0.25, name=scope.name)

    with tf.variable_scope('conv2') as scope:
        conv = tf.layers.conv2d(
            inputs=drop,
            filters=128,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )
        pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        conv = tf.layers.conv2d(
            inputs=pool,
            filters=128,
            kernel_size=[2, 2],
            padding='SAME',
            activation=tf.nn.relu
        )
        pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        drop = tf.layers.dropout(pool, rate=0.25, name=scope.name)

    with tf.variable_scope('fully_connected') as scope:
        flat = tf.reshape(drop, [-1, 4 * 4 * 128])

        fc = tf.layers.dense(inputs=flat, units=1500, activation=tf.nn.relu)
        drop = tf.layers.dropout(fc, rate=0.5)
        softmax = tf.layers.dense(inputs=drop, units=_NUM_CLASSES, name=scope.name)

    y_pred_cls = tf.argmax(softmax, axis=1)

    return x, y, softmax, y_pred_cls, global_step, learning_rate, pade_output, pade_x, pade_y, x_image


def lr(epoch):
    learning_rate = 1e-3
    if epoch > 80:
        learning_rate *= 0.5e-3
    elif epoch > 60:
        learning_rate *= 1e-3
    elif epoch > 40:
        learning_rate *= 1e-2
    elif epoch > 20:
        learning_rate *= 1e-1
    return learning_rate
