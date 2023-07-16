import tensorflow as tf
import gc


# Custom Callback To Include in Callbacks List At Training Time
class GarbageCollectorCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()


def lr_scheduler(epoch, lr, lr_init, max_epochs):
#     slow_iters = 0.1*max_epochs
    power = 0.9
#     if epoch+1 <= slow_iters:
#         return lr_init*((epoch+1)/slow_iters)
#     else:
    return lr_init*(1-epoch/max_epochs)**power