import tensorflow as tf
import numpy as np
import time, os, sys, glob
import argparse, random
import scipy.misc
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer

# taken from https://github.com/openai/iaf/blob/master/tf_utils/adamax.py
class AdamaxOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Adamax algorithm. 
      See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980) 
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)). """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, use_locking=False, name="Adamax"):
        super(AdamaxOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7  # Can't use 1e-8 due to underflow -- not sure if it makes a big difference.
        else:
            eps = 1e-8
        v = self.get_slot(var, "v")
        v_t = v.assign(beta1_t * v + (1. - beta1_t) * grad)
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = v_t / m_t
        var_update = state_ops.assign_sub(var, lr_t * g_t)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")


def image_list(image_dir):
    # Check image directory existency
    if not os.path.exists(image_dir):
        print('Image directory %s not exists' % image_dir)
        return None
    file_type_extended = ('jpg', 'jpeg', 'png')
    file_list = []
    for path, dir, files in os.walk(image_dir):
        #print('path : %s' %path)
        #print('dir : %s ' %dir)
        #print('files : %s' % files)
        #file_dir_path = os.path.join(os.path.basename(image_dir), path)
        for file in files:
            if file.split('.')[-1] in file_type_extended:
                file_list.append(os.path.join(os.path.abspath(image_dir), file))
            else:
                pass
    if len(file_list) == 0:
        print('No image files')
        return None
    else:
        random.shuffle(file_list)
        print('Number of files %d' % len(file_list))
        '''
        shuffle : Boolean. If true, the strings are randomly shuffled within each epoch
        capacity : An integer, Sets the queue capacity maximum
        num_epoch : If not specified string_input_producer can cycle through the strings in input
        FIFOQUEUE + QUEUERUNNER'''
        file_list_queue = tf.train.string_input_producer(file_list, capacity=1000, shuffle=True) # Need to change 'queue'  from ''list'
        return file_list_queue

def read_files_preprocess(file_list_queue, args):
    image_reader = tf.WholeFileReader()
    key, value = image_reader.read(file_list_queue) # Returns both string scalar tensor
    uint8_image = tf.image.decode_jpeg(value, channels=args.num_channels) # Returns of type uint8 with [height, width, channels], # tf.image.decode_image not working
	#image_spec = uint8_image.get_shape().as_list() -> height, width is unknown
	#print(image_expanded_4d.get_shape()) # [1,?,?,args.num_channels]
 	#offset_height = (image_spec[0] - args.input_size) // 2
 	#offset_width = (image_spec[1] - args.input_size) // 2
    cropped_image = tf.cast(tf.image.crop_to_bounding_box(uint8_image, offset_height=50, offset_width=35, target_height=args.input_size ,target_width=args.input_size), tf.float32)
    image_expanded_4d = tf.expand_dims(cropped_image, 0) # Make 4 dimensional considering batch dimension, make it available as input
    resized_image = tf.image.resize_bilinear(image_expanded_4d, size=[args.target_size, args.target_size])
    input_image = tf.squeeze(resized_image, axis=0)
    return input_image

def read_input(file_list_queue, args):
    inp_img = read_files_preprocess(file_list_queue, args)
    num_preprocess_threads = 4
    min_queue_examples = int(0.1*args.num_examples_per_epoch)
    '''    
    	This function adds:
			1. A shuffling queue into which tensors from tensor list are enqueued. 
			2. A dequeue_many operation to create batches from the queue. 
			3. A QueueRunner to enqueue the tensors form tensor list
		shuffle_batch constructs a RandomShuffleQueue and proceeds to fill with a QueueRunner. 
		The queue accumulates examples sequentialy until in contains bach_size + min_after_dequeue examples are present.
		It then selects batch_size random element from the queue to return The value actually returned by shffle_batch is the result of a dequeue_may call on the RandomShuffleQueue
    	enqueue_many argument set as False -> input shape will be [x,y,z], output will be [batch, x,y,z]
    
		Reference from http://stackoverflow.com/questions/36334371/tensorflow-batching-input-queues-then-changing-the-queue-source'''
    #  tensors part should be iterable so [] needed(tensor list)
    input_image = tf.train.shuffle_batch([inp_img], batch_size=args.batch_size, num_threads=num_preprocess_threads, capacity=min_queue_examples+3*args.batch_size,  min_after_dequeue=min_queue_examples)
    input_image = input_image / 127.5  - 1
    return input_image

def save_image(imgs, size, path):
    print('Save images')
    height, width = imgs.shape[1], imgs.shape[2]
    merged_image = np.zeros([size[0]*height, size[1]*width, imgs.shape[3]])
    for image_index, img in enumerate(imgs):
        j = image_index % size[1]
        i = image_index // size[1]
        merged_image[i*height:i*height+height, j*width:j*width+width, :] = img
    merged_image += 1
    merged_image *=127.5
    merged_image = np.clip(merged_image, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, merged_image)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--num_examples_per_epoch', type=int, default=1)
    parser.add_argument('--input_size', type=int, default=108)
    args = parser.parse_args()
    print(args)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    files = image_list('../data_sets/CelebA',100)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print('file queue size %d' %(sess.run(files.size())))
    print(sess.run(files.dequeue_many(20)))
    print(sess.run(files.size()))
    a = read_input(files, args)
    coord.request_stop()
    coord.join(threads)
