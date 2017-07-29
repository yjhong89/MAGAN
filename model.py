import tensorflow as tf
import numpy as np
import time, os, argparse
from operations import *
from utils import *

class MAGAN():
    def __init__(self, args, sess):
        self.args = args
        self.sess = sess

        self.build_model()

    def build_model(self):
        #self.input_image = tf.placeholder(tf.float32, [self.args.batch_size] + [self.args.target_size, self.args.target_size, self.args.num_channels], name='real_image')
        self.real_datas = image_list(self.args.dataset) # Get files FIFO queue
        real_images = read_input(self.real_datas, self.args)
        self.z = tf.placeholder(tf.float32, [None, self.args.z_dim], name='noise_z')
        tf.summary.histogram('z', self.z)

        self.disc_pre_image, self.disc_pre_energy = self.disc_pretrain(real_images)
        tf.summary.scalar('discriminator_pretrain_energy', self.disc_pre_energy)

        self.g = self.generator(self.z)
        tf.summary.image('generated_image', self.g)
        self.real_embedding, self.real_ae, self.discriminator_real_loss = self.discriminator(real_images, reuse=False)
        tf.summary.scalar('discriminator_real_energy', self.discriminator_real_loss)
        self.fake_embedding, self.fake_ae, self.discriminator_fake_loss = self.discriminator(self.g, reuse=True)
        tf.summary.scalar('discriminator_fake_energy', self.discriminator_fake_loss)
        self.generated_sample = self.generator(self.z, reuse=True)
        self.margin = tf.Variable(initial_value=0, name='margin', trainable=False, dtype=tf.float32 )
        tf.summary.scalar('margin', self.margin)
        self.discriminator_loss = self.discriminator_real_loss + tf.maximum(self.margin - self.discriminator_fake_loss, 0)
        tf.summary.scalar('discriminator_loss', self.discriminator_loss)

        self.sess.run(tf.global_variables_initializer())
        self.tr_vrbs = tf.trainable_variables()
        for i in self.tr_vrbs:
            print(i.op.name)

        self.d_pre_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d_pre')
        self.d_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.g_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        self.d_pre_opt = AdamaxOptimizer(self.args.learning_rate).minimize(self.disc_pre_energy, var_list=self.d_pre_param)
        d_opt = AdamaxOptimizer(self.args.learning_rate)
        d_grads = d_opt.compute_gradients(self.discriminator_loss, var_list=self.d_param)
        for grads, vars in d_grads:
            if grads is not None:
                tf.summary.histogram(vars.op.name+'./gradient', grads)
        self.d_optimizer = d_opt.apply_gradients(d_grads)
        g_opt = AdamaxOptimizer(self.args.learning_rate)
        g_grads = g_opt.compute_gradients(self.discriminator_fake_loss, var_list=self.g_param)
        for grads, vars in g_grads:
            if grads is not None:
                tf.summary.histogram(vars.op.name+'./gradient', grads)
        self.g_optimizer = g_opt.apply_gradients(g_grads)

        self.saver = tf.train.Saver()

    '''The architecture is the decoder portion
    (4,4512)4c2-(8,8,256)4c2-(16,16,128)4c2-(32,32,64)4c2-(64,64,3)4c2
    '''
    # self.args.final_dim=64, self.args.target_size=64
    def generator(self, z, reuse=False): # z : [batch_size, 350]
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()

            z_flatten = linear(z, self.args.target_size // 16 * self.args.target_size // 16 * self.args.final_dim * 8, name='linear')
            z_reshaped = tf.reshape(z_flatten, [-1, self.args.target_size // 16, self.args.target_size // 16, self.args.final_dim * 8])
            deconvolution1 = deconv2d(z_reshaped, output_shape=[self.args.batch_size, self.args.target_size // 8, self.args.target_size // 8, self.args.final_dim * 4], filter_width=4, filter_height=4, stride_ver=2, stride_hor=2, name='deconv1')
            deconv2 = tf.nn.relu(deconvolution1, name='gen_relu1')

            deconvolution2 = deconv2d(deconv2, output_shape=[self.args.batch_size, self.args.target_size // 4, self.args.target_size //4, self.args.final_dim * 2], filter_width=4, filter_height=4, stride_hor=2, stride_ver=2, name='deconv2')
            deconv3 = tf.nn.relu(deconvolution2, name='gen_relu2')

            deconvolution3 = deconv2d(deconv3, output_shape=[self.args.batch_size, self.args.target_size // 2, self.args.target_size //2, self.args.final_dim], filter_height=4, filter_width=4, stride_ver=2, stride_hor=2, name='deconv3')
            deconv4 = tf.nn.relu(deconvolution3, name='gen_relu3')

            deconvolution4 = deconv2d(deconv4, output_shape=[self.args.batch_size, self.args.target_size, self.args.target_size, self.args.num_channels], filter_width=4, filter_height=4, stride_hor=2, stride_ver=2, name='deconv4')
            return tf.nn.tanh(deconvolution4, name='generated_image')

    # (64)4c2s-(128)4c2s-(256)4c2s-(512)4c2s-(256)4c2s-(128)4c2s-(64)4c2s-(3)4c2s
    def discriminator(self, image, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            self.embedding = self.encoder(image)
            self.decoded_image = self.decoder(self.embedding)
            self.mse_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(image-self.decoded_image), axis=[1,2,3])))
            return self.embedding, self.decoded_image, self.mse_loss

    # Beriefly pre train the discriminator with an auto-encoder for 2 epochs using only real samples
    def disc_pretrain(self, onlyrealimage):
        with tf.variable_scope('d_pre'):
            self.embedding_pre = self.encoder(onlyrealimage)
            self.decoded_image_pre = self.decoder(self.embedding_pre)
            self.mse_loss_pre = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(onlyrealimage-self.decoded_image_pre), axis=[1,2,3])))
            return self.decoded_image_pre, self.mse_loss_pre


    def encoder(self, realorfake_image): # 64,64,3 image
        convolution1 = conv2d(realorfake_image, self.args.final_dim, filter_height=4, filter_width=4, stride_ver=2, stride_hor=2, name='enc_conv1')
        conv2 = lrelu(convolution1, name='enc_lrelu1')

        convolution2 = conv2d(conv2, self.args.final_dim*2, filter_height=4, filter_width=4, stride_hor=2, stride_ver=2, name='enc_conv2')
        conv3 = lrelu(convolution2, name='enc_lrelu2')

        convolution3 = conv2d(conv3, self.args.final_dim*4, filter_height=4, filter_width=4, stride_ver=2, stride_hor=2, name='enc_conv3')
        conv4 = lrelu(convolution3, name='enc_lrelu3')

        convolution4 = conv2d(conv4, self.args.final_dim*8, filter_width=4, filter_height=4, stride_hor=2, stride_ver=2, name='enc_conv4')
        conv5 = lrelu(convolution4, name='enc_lrelu4')
        embedding = tf.reshape(conv5, [self.args.batch_size, -1])

        return embedding

    def decoder(self, embedding):
        embedding_reshaped = tf.reshape(embedding, [self.args.batch_size, self.args.target_size // 16, self.args.target_size // 16, self.args.final_dim*8 ])
        decoding1 = deconv2d(embedding_reshaped, output_shape=[self.args.batch_size, self.args.target_size // 8, self.args.target_size // 8, self.args.final_dim*4], filter_height=4, filter_width=4, stride_ver=2, stride_hor=2, name='dec_deconv1')
        decode2 = lrelu(decoding1, name='dec_lrelu1')

        decoding2 = deconv2d(decode2, output_shape=[self.args.batch_size, self.args.target_size // 4, self.args.target_size // 4, self.args.final_dim *2], filter_width=4, filter_height=4, stride_hor=2, stride_ver=2, name='dec_deconv2')
        decode3 = lrelu(decoding2, name='dec_lrelu2')

        decoding3 = deconv2d(decode3, output_shape=[self.args.batch_size, self.args.target_size // 2, self.args.target_size // 2, self.args.final_dim], filter_height=4, filter_width=4, stride_ver=2, stride_hor=2, name='dec_deconv3')
        decode4 = lrelu(decoding3, name='dec_lrelu3')

        decoding4 = deconv2d(decode4, output_shape=[self.args.batch_size, self.args.target_size, self.args.target_size, self.args.num_channels], filter_width=4, filter_height=4, stride_hor=2, stride_ver=2, name='dec_deconv4')
        decode5 = lrelu(decoding4, name='dec_lrelu4')

        return tf.nn.tanh(decode5, name='autoencoder_result')


    def train(self):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.summary_merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.args.log_dir, self.sess.graph)

        self.train_count = 0
        self.start_time = time.time()
        # Margin update terms
        prt_sum = 0
        real_sum = 0
        fake_sum = 0
        pre_fake_sum = 0
        training_step_per_epoch = self.args.training_size // self.args.batch_size
        margin_update = tf.cond(tf.logical_and(tf.less(real_sum, fake_sum), tf.less(pre_fake_sum, fake_sum)), lambda : self.margin.assign(self.discriminator_real_loss *self.args.batch_size / self.args.training_size), lambda : self.margin.assign_add(0))
        self.sample_z = np.random.normal(loc=0, scale=1, size=[self.args.showing_height*self.args.showing_width, self.args.z_dim])
        sample_dir = os.path.join(self.args.sample_dir, self.model_dir)
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)

        self.sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        try:
            if self.load():
                print('Checkpoint loaded')
            else:
                print('Checkpoint load failed')
                print('In order to get an initial estimate of the margins, Pretraining')
                for prt_epoch in range(self.args.pre_train_epoch):
                    for pre_training_step in range(training_step_per_epoch):
                        #pre_batch = read_input(input_queue, self.args) # It is tensor
                        pretrained_energy, _ = self.sess.run([self.disc_pre_energy, self.d_pre_opt])
                        print('Steps %d/%d in epoch %d, pretrained energy : %3.4f' %(pre_training_step+1, self.args.training_size//self.args.batch_size, prt_epoch+1, pretrained_energy))
                    print('Pre train epoch %d, pre train margin %3.9f, time : %3.4f' % (prt_epoch+1, pretrained_energy, time.time()-self.start_time))
                # Copy pre train discriminator to discriminator
                for i, v in enumerate(self.d_param):
                    self.sess.run(v.assign(self.d_pre_param[i]))
                print('Copied')
                # Setting initial margin
                for i in range(training_step_per_epoch):
                 prt_sum += self.sess.run(self.disc_pre_energy)
                 print('pretrain energy : %3.4f' % self.sess.run(self.disc_pre_energy))
                self.sess.run(self.margin.assign(prt_sum / training_step_per_epoch))
                print('Initial margin : %3.9f' % self.sess.run(self.margin))

            for epoch in range(self.args.num_epoch):
                print('Epoch %d, margin : %3.9f' % (epoch+1, self.sess.run(self.margin)))
                real_sum = 0
                fake_sum = 0
                pre_fake_sum = 0
                for index in range(training_step_per_epoch):
                    self.train_count += 1
                    batch_z = np.random.normal(loc=0, scale=1, size=[self.args.batch_size, self.args.z_dim])
                    disc_loss, summary, margin_, _ = self.sess.run([self.discriminator_real_loss, self.summary_merged, self.margin, self.d_optimizer], feed_dict={self.z : batch_z})
                    self.summary_writer.add_summary(summary, self.train_count)
                    # Using tf.Tensor as a python 'bool' is not allowed
                    # Update margin when checkpoint loaded
                    if margin_ == 0:
                        print('Checkpoint loaded before')
                        self.sess.run(self.margin.assign(self.discriminator_real_loss / training_step_per_epoch))
                    real_sum += disc_loss
                    # Sample another noise batch for training the generator
                    batch_z = np.random.normal(loc=0, scale=1, size=[self.args.batch_size, self.args.z_dim])
                    gen_loss, _ = self.sess.run([self.discriminator_fake_loss, self.g_optimizer], feed_dict={self.z : batch_z})
                    pre_fake_sum = fake_sum
                    fake_sum += gen_loss
                    print('Epoch %d, step %d/%d, real_sum : %3.4f, fake_sum : %3.4f, pre_fake_sum : %3.4f, duration_time : %3.4f' % (epoch+1, index+1, training_step_per_epoch, real_sum, fake_sum, pre_fake_sum, time.time()-self.start_time)) 
                print('Epoch %d, discriminator real loss : %3.4f, discriminator fake loss : %3.4f' % (epoch+1, real_sum / training_step_per_epoch, fake_sum / training_step_per_epoch))
                if (real_sum / training_step_per_epoch) < margin_:
                    self.sess.run(margin_update)
                    print('Margin updated to %3.4f' % (self.sess.run(self.margin)))
                # Convergence measure
                convergence_measure = tf.add(real_sum / training_step_per_epoch, tf.abs((real_sum/training_step_per_epoch - fake_sum/training_step_per_epoch)))
                c = self.sess.run(convergence_measure)
                print('Convergence measure %3.4f' %c )

                if np.mod(epoch+1, 5) == 0:
                    G_sample = self.sess.run(self.generated_sample, feed_dict={self.z : self.sample_z})
                    save_image(G_sample, [self.args.showing_height, self.args.showing_width], os.path.join(sample_dir, 'train_{:2d}epoch.jpg'.format(epoch+1)))
                    self.save(self.train_count)

        except tf.errors.OutOfRangeError:
            print('Epoch limited')
        except KeyboardInterrupt:
            print('End training')
        finally:
            coord.request_stop()
            coord.join(threads)

    def generator_test(self):
        self.load()
        z_test = np.random.normal(loc=0, scale=1, size=[self.args.showing_height * self.args.showing_width, self.args.z_dim])
        generated = self.sess.run(self.generated_sample)
        save_image(generated, [self.args.showing_height, self.args.showing_width], '{}/test.jpg'.format(self.args.sample_dir))


    @property
    def model_dir(self):
        return '{}batch_size_{}z_dim_{}'.format(self.args.batch_size, self.args.z_dim, 'CelebA')

    def save(self, global_step):
        model_name = 'MAGAN'
        checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)
        print('Save checkpoint at %d' % global_step)

    def load(self):
        checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.train_count = int(ckpt_name.split('-')[-1])
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print('Checkpoint loaded at %d steps' % self.train_count)
            return True
        else:
            self.train_count = 0
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--num_examples_per_epoch', type=int, default=11)
    parser.add_argument('--input_size', type=int, default=108)
    args = parser.parse_args()




