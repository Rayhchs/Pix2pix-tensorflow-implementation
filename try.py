import tensorflow as tf
import numpy as np
from PIL import Image
from data_loader import get_batch_data
import os
import re


class pix2pix(object):
    def __init__(self, sess, batch_size, L1_lambda):
        """

        :param sess: tf.Session
        :param batch_size: batch_size. [int]
        :param L1_lambda: L1_loss lambda. [int]
        """
        self.sess = sess
        self.k_initializer = tf.random_normal_initializer(0, 0.02)
        self.g_initializer = tf.random_normal_initializer(1, 0.02)
        self.L1_lambda = L1_lambda
        self.bulid_model()

    def bulid_model(self):
        """
        初始化模型
        :return:
        """
        # init variable
        self.x_ = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name='x')
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name='y')

        # generator
        self.g = self.generator(self.x_)

        # discriminator
        self.d_real = self.discriminator(self.x_, self.y_)
        self.d_fake = self.discriminator(self.x_, self.g, reuse=True)

        # loss
        self.loss_g, self.loss_d = self.loss(self.d_real, self.d_fake, self.y_, self.g)

        # summary
        tf.summary.scalar("loss_g", self.loss_g)
        tf.summary.scalar("loss_d", self.loss_d)
        self.merged = tf.summary.merge_all()

        # vars
        self.vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
        self.vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

        # saver
        self.saver = tf.train.Saver()

    def discriminator(self, x, y, reuse=None):
        """
        判別器
        :param x: 輸入圖像. [tensor]
        :param y: 目標圖像. [tensor]
        :param reuse: reuse or not. [boolean]
        :return:
        """
        with tf.variable_scope('discriminator', reuse=reuse):
            x = tf.concat([x, y], axis=3)
            h0 = self.lrelu(self.d_conv(x, 64, 2))  # 128 128 64

            h0 = self.d_conv(h0, 128, 2)
            h0 = self.lrelu(self.batch_norm(h0))  # 64 64 128

            h0 = self.d_conv(h0, 256, 2)
            h0 = self.lrelu(self.batch_norm(h0))  # 32 32 256

            h0 = self.d_conv(h0, 512, 1)
            h0 = self.lrelu(self.batch_norm(h0))  # 31 31 512

            h0 = self.d_conv(h0, 1, 1)  # 30 30 1
            h0 = tf.nn.sigmoid(h0)

            return h0

    def generator(self, x):
        """
        生成器
        :param x: 輸入圖像. [tensor]
        :return: h0,生成的圖像. [tensor]
        """
        with tf.variable_scope('generator', reuse=None):
            layers = []
            h0 = self.g_conv(x, 64)
            layers.append(h0)

            for filters in [128, 256, 512, 512, 512, 512, 512]:  # [128, 256, 512, 512, 512, 512, 512]
                h0 = self.lrelu(layers[-1])
                h0 = self.g_conv(h0, filters)
                h0 = self.batch_norm(h0)
                layers.append(h0)

            encode_layers_num = len(layers)  # 8

            for i, filters in enumerate([512, 512, 512, 512, 256, 128, 64]):  # [512, 512, 512, 512, 256, 128, 64]
                skip_layer = encode_layers_num - i - 1
                if i == 0:
                    inputs = layers[-1]
                else:
                    inputs = tf.concat([layers[-1], layers[skip_layer]], axis=3)
                h0 = tf.nn.relu(inputs)
                h0 = self.g_deconv(h0, filters)
                h0 = self.batch_norm(h0)
                if i < 3:
                    h0 = tf.nn.dropout(h0, keep_prob=0.5)
                layers.append(h0)

            inputs = tf.concat([layers[-1], layers[0]], axis=3)
            h0 = tf.nn.relu(inputs)
            h0 = self.g_deconv(h0, 3)
            h0 = tf.nn.tanh(h0, name='g')
            return h0

    def loss(self, d_real, d_fake, y, g):
        """
        定義損失函數
        :param d_real: 真實圖像判別器的輸出. [tensor]
        :param d_fake: 生成圖像判別器的輸出. [tensor]
        :param y: 目標圖像. [tensor]
        :param g: 生成圖像. [tensor]
        :return: loss_g, loss_d, 分別對應生成器的損失函數和判別器的損失函數
        """
        loss_d_real = tf.reduce_mean(self.sigmoid_cross_entropy_with_logits(d_real, tf.ones_like(d_real)))
        loss_d_fake = tf.reduce_mean(self.sigmoid_cross_entropy_with_logits(d_fake, tf.zeros_like(d_fake)))
        loss_d = loss_d_real + loss_d_fake

        loss_g_gan = tf.reduce_mean(self.sigmoid_cross_entropy_with_logits(d_fake, tf.ones_like(d_fake)))
        loss_g_l1 = tf.reduce_mean(tf.abs(y - g))
        loss_g = loss_g_gan + loss_g_l1 * self.L1_lambda
        return loss_g, loss_d

    def lrelu(self, x, leak=0.2):
        """
        lrelu函數
        :param x:
        :param leak:
        :return:
        """
        return tf.maximum(x, leak * x)

    def d_conv(self, inputs, filters, strides):
        """
        判別器卷積層
        :param inputs: 輸入. [tensor]
        :param filters: 輸出通道數. [int]
        :param strides: 卷積核步伐. [int]
        :return:
        """
        padded = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
        return tf.layers.conv2d(padded,
                                kernel_size=4,
                                filters=filters,
                                strides=strides,
                                padding='valid',
                                kernel_initializer=self.k_initializer)

    def g_conv(self, inputs, filters):
        """
        生成器卷積層
        :param inputs: 輸入. [tensor]
        :param filters: 輸出通道數. [int]
        :return:
        """
        return tf.layers.conv2d(inputs,
                                kernel_size=4,
                                filters=filters,
                                strides=2,
                                padding='same',
                                kernel_initializer=self.k_initializer)

    def g_deconv(self, inputs, filters):
        """
        生成器反捲積層
        :param inputs: 輸入. [tensor]
        :param filters: 輸出通道數. [int]
        :return:
        """
        return tf.layers.conv2d_transpose(inputs,
                                          kernel_size=4,
                                          filters=filters,
                                          strides=2,
                                          padding='same',
                                          kernel_initializer=self.k_initializer)

    def batch_norm(self, inputs):
        """
        批標準化函數
        :param inputs: 輸入. [tensor]
        :return:
        """
        return tf.layers.batch_normalization(inputs,
                                             axis=3,
                                             epsilon=1e-5,
                                             momentum=0.1,
                                             training=True,
                                             gamma_initializer=self.g_initializer)

    def sigmoid_cross_entropy_with_logits(self, x, y):
        """
        交叉熵函數
        :param x:
        :param y:
        :return:
        """
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x,
                                                       labels=y)

    def train(self, images, epoch, batch_size):
        """
        訓練函數
        :param images: 圖像路徑列表. [list]
        :param epoch: 迭代次數. [int]
        :param batch_size: batch_size. [int]
        :return:
        """
        # optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optim_d = tf.train.AdamOptimizer(learning_rate=0.0002,
                                             beta1=0.5
                                             ).minimize(self.loss_d, var_list=self.vars_d)
            optim_g = tf.train.AdamOptimizer(learning_rate=0.0002,
                                             beta1=0.5
                                             ).minimize(self.loss_g, var_list=self.vars_g)

        # init variables
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./log", self.sess.graph)

        # training
        for i in range(epoch):
            # 獲取圖像列表
            print("Epoch:%d/%d:" % ((i + 1), epoch))
            batch_num = int(np.ceil(len(images) / batch_size))
            # batch_list = np.array_split(random.sample(images, len(images)), batch_num)
            batch_list = np.array_split(images, batch_num)

            # 訓練生成器和判別器
            for j in range(len(batch_list)):
                batch_x, batch_y = get_batch_data(batch_list[j])
                _, loss_d = self.sess.run([optim_d, self.loss_d],
                                          feed_dict={self.x_: batch_x, self.y_: batch_y})
                _, loss_g = self.sess.run([optim_g, self.loss_g],
                                          feed_dict={self.x_: batch_x, self.y_: batch_y})
                print("%d/%d -loss_d:%.4f -loss_g:%.4f" % ((j + 1), len(batch_list), loss_d, loss_g))

            # 保存損失值
            summary = self.sess.run(self.merged,
                                    feed_dict={self.x_: batch_x, self.y_: batch_y})
            self.writer.add_summary(summary, global_step=i)

            # 保存模型，每10次保存一次
            if (i + 1) % 10 == 0:
                self.saver.save(self.sess, './checkpoint/epoch_%d.ckpt' % (i + 1))

            # 測試，每循環一次測試一次
            if (i + 1) % 1 == 0:
                # 對訓練集最後一張圖像進行測試
                train_save_path = os.path.join('./result/train',
                                               re.sub('.jpg',
                                                      '',
                                                      os.path.basename(images[-1])
                                                      ) + '_' + str(i + 1) + '.jpg'
                                               )
                train_g = self.sess.run(self.g,
                                        feed_dict={self.x_: batch_x}
                                        )
                train_g = 255 * (np.array(train_g[0] + 1) / 2)
                im = Image.fromarray(np.uint8(train_g))
                im.save(train_save_path)

                # 對驗證集進行測試
                img = np.zeros((256, 256 * 3, 3))
                val_img_path = np.array(['./data/val/color/10901.jpg'])
                batch_x, batch_y = get_batch_data(val_img_path)
                val_g = self.sess.run(self.g, feed_dict={self.x_: batch_x})
                img[:, :256, :] = 255 * (np.array(batch_x + 1) / 2)
                img[:, 256:256 * 2, :] = 255 * (np.array(batch_y + 1) / 2)
                img[:, 256 * 2:, :] = 255 * (np.array(val_g[0] + 1) / 2)
                img = Image.fromarray(np.uint8(img))
                img.save('./result/val/10901_%d.jpg' % (i + 1))

    def save_img(self, g, data, save_path):
        """
        保存圖像
        :param g: 生成的圖像. [array]
        :param data: 測試數據. [list]
        :param save_path: 保存路徑. [str]
        :return:
        """
        if len(data) == 1:
            img = np.zeros((256, 256 * 2, 3))
            img[:, :256, :] = 255* (np.array(data[0] + 1) / 2)
            img[:, 256:, :] = 255 * (np.array(g[0] + 1) / 2)
        else:
            img = np.zeros((256, 256 * 3, 3))
            img[:, :256, :] = 255 * (np.array(data[0] + 1) / 2)
            img[:, 256:256 * 2, :] = 255 * (np.array(data[1] + 1) / 2)
            img[:, 256 * 2:, :] = 255 * (np.array(g[0] + 1) / 2)

        im = Image.fromarray(np.uint8(img))
        im.save(os.path.join('./result/test', os.path.basename(save_path)))

    def test(self, images, batch_size=1, save_path=None, mode=None):
        """
        測試函數
        :param images: 測試圖像列表. [list]
        :param batch_size: batch_size. [int]
        :param save_path: 保存路徑
        :return:
        """
        # init variables
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # load model
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./checkpoint'))

        # test
        if mode != 'orig':
            for j in range(len(images)):
                batch_x, batch_y = get_batch_data(np.array([images[j]]))

                g = self.sess.run(self.g, feed_dict={self.x_: batch_x})

                if save_path == None:
                    self.save_img(g,
                                  data=[batch_x[0], batch_y[0]],
                                  save_path=images[j]
                                  )
                else:
                    self.save_img(g,
                                  data=[batch_x[0], batch_y[0]],
                                  save_path=save_path
                                  )
        else:
            for j in range(len(images)):
                batch_x = get_batch_data(np.array([images[j]]), mode=mode)
                g = self.sess.run(self.g, feed_dict={self.x_: batch_x})
                batch_x = 255 * (np.array(batch_x[0] + 1) / 2)
                g = 255 * (np.array(g[0] + 1) / 2)
                img = np.hstack((batch_x, g))
                im = Image.fromarray(np.uint8(img))
                im.save(os.path.join('./result/test', os.path.basename(images[j])))