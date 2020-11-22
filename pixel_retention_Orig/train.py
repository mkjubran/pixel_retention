import numpy as np
import tensorflow as tf
from time import time
import math, pdb


from include.data import get_data_set
from include.model import model, lr


train_x, train_y = get_data_set("train")
test_x, test_y = get_data_set("test")
tf.set_random_seed(21)

_RETAINED_PIXELS = 256
x, y, output, y_pred_cls, global_step, learning_rate, pade_output, pade_x, pade_y, x_image = model(_RETAINED_PIXELS)
# x, y, output, y_pred_cls, global_step, learning_rate = model()
global_accuracy = 0
epoch_start = 0


# PARAMS
_BATCH_SIZE = 128
_EPOCH = 60
_SAVE_PATH = "./tensorboard/cifar-10-v1.0.0/"


# LOSS AND OPTIMIZER
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-08).minimize(loss, global_step=global_step)


# PREDICTION AND ACCURACY CALCULATION
correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
summ_op_1 = tf.summary.image('pixel_domain', x_image) # , step=None, max_outputs=3, description=None)
summ_op_2 = tf.summary.image('pade_output_2', pade_output) # , step=None, max_outputs=3, description=None)
# pdb.set_trace()
# SAVER
merged = tf.summary.merge_all()
saver = tf.train.Saver()
sess = tf.Session()


# eval(feed_dict=None, session=None)
# with sess1.as_default() as sess:
train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)
tf.summary.image('pade_output_3', pade_output) # , step=None, max_outputs=3, description=None)
# train_writer.add_summary(summ_op_2.eval(session=sess))


try:
    print("\nTrying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except ValueError:
    print("\nFailed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())


def train(epoch):
    global epoch_start
    epoch_start = time()
    batch_size = int(math.ceil(len(train_x) / _BATCH_SIZE))
    i_global = 0

    # pdb.set_trace()
    # for s in range(batch_size):
    # -1 to maintain batch size fixed 
    for s in range(batch_size - 1 ):
        batch_xs = train_x[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
        batch_ys = train_y[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
        train_writer.add_summary(summ_op_2.eval(feed_dict={x: batch_xs, y: batch_ys, pade_x: batch_xs, pade_y: batch_ys, learning_rate: lr(epoch)}, session=sess))
        train_writer.add_summary(summ_op_1.eval(feed_dict={x: batch_xs, y: batch_ys, pade_x: batch_xs, pade_y: batch_ys, learning_rate: lr(epoch)}, session=sess))
        # train_writer.add_summary(summ_op_2.eval())
        # pdb.set_trace()
        start_time = time()
        i_global, _, batch_loss, batch_acc, img_summ = sess.run(
            [global_step, optimizer, loss, accuracy, summ_op_2],
            feed_dict={x: batch_xs, y: batch_ys, pade_x: batch_xs, pade_y: batch_ys, learning_rate: lr(epoch)})
            # feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(epoch)})
        duration = time() - start_time

        if s % 10 == 0:
            percentage = int(round((s/batch_size)*100))

            bar_len = 29
            filled_len = int((bar_len*int(percentage))/100)
            bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len)

            msg = "Global step: {:>5} - [{}] {:>3}% - acc: {:.4f} - loss: {:.4f} - {:.1f} sample/sec"
            print(msg.format(i_global, bar, percentage, batch_acc, batch_loss, _BATCH_SIZE / duration))

    test_and_save(i_global, epoch)


def test_and_save(_global_step, epoch):
    global global_accuracy
    global epoch_start

    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))
        # Skip last instances to fix batc size
        if j == i + _BATCH_SIZE:
           batch_xs = test_x[i:j, :]
           batch_ys = test_y[i:j, :]
           predicted_class[i:j] = sess.run(
               y_pred_cls,
               feed_dict={x: batch_xs, y: batch_ys, pade_x: batch_xs, pade_y: batch_ys,  learning_rate: lr(epoch)}
           )
           # feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(epoch)}
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean()*100
    correct_numbers = correct.sum()

    hours, rem = divmod(time() - epoch_start, 3600)
    minutes, seconds = divmod(rem, 60)
    mes = "\nEpoch {} - accuracy: {:.2f}% ({}/{}) - time: {:0>2}:{:0>2}:{:05.2f}"
    print(mes.format((epoch+1), acc, correct_numbers, len(test_x), int(hours), int(minutes), seconds))

    if global_accuracy != 0 and global_accuracy < acc:

        summary = tf.Summary(value=[
            tf.Summary.Value(tag="Accuracy/test", simple_value=acc),
        ])
        pade_summary = tf.summary.image('pade_output_4', pade_output) # , step=None, max_outputs=3, description=None)
        train_writer.add_summary(summary, _global_step)
        # train_writer.add_summary(pade_summary, _global_step)

        saver.save(sess, save_path=_SAVE_PATH, global_step=_global_step)

        mes = "This epoch receive better accuracy: {:.2f} > {:.2f}. Saving session..."
        print(mes.format(acc, global_accuracy))
        global_accuracy = acc

    elif global_accuracy == 0:
        global_accuracy = acc

    print("###########################################################################################################")


def main():
    train_start = time()

    for i in range(_EPOCH):
        print("\nEpoch: {}/{}\n".format((i+1), _EPOCH))
        train(i)

    hours, rem = divmod(time() - train_start, 3600)
    minutes, seconds = divmod(rem, 60)
    mes = "Best accuracy pre session: {:.2f}, time: {:0>2}:{:0>2}:{:05.2f}"
    print(mes.format(global_accuracy, int(hours), int(minutes), seconds))


if __name__ == "__main__":
    main()


sess.close()
