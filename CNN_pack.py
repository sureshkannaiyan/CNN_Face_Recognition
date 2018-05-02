# Importing required packages
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt

class CNN_pack:
    def CNN(img):
        # filters (used in convolutional layer)
        def sf1():
            # sharpen filter
            sf = np.zeros([3, 3, 1, 1])
            sf[1, 1, :, :] = 7
            sf[0, 1, :, :] = -1
            sf[1, 0, :, :] = -1
            sf[2, 1, :, :] = -1
            sf[1, 2, :, :] = -1
            return sf

        def sf2():
            # sharpen filter
            sf = np.zeros([3, 3, 1, 1])
            sf[1, 1, :, :] = 5
            sf[0, 1, :, :] = -1
            sf[1, 0, :, :] = -1
            sf[2, 1, :, :] = -1
            sf[1, 2, :, :] = -1
            return sf

        def sf3():
            # sharpen filter
            sf = np.zeros([3, 3, 1, 1])
            sf[1, 1, :, :] = 3
            sf[0, 1, :, :] = -1
            sf[1, 0, :, :] = -1
            sf[2, 1, :, :] = -1
            sf[1, 2, :, :] = -1
            return sf

        def bf1():
            # blur filter
            bf = np.zeros([3, 3, 1, 1])
            bf[1, 1, :, :] = 0.75
            bf[0, 1, :, :] = 0.125
            bf[1, 0, :, :] = 0.125
            bf[2, 1, :, :] = 0.125
            bf[1, 2, :, :] = 0.125
            bf[0, 0, :, :] = 0.0625
            bf[0, 2, :, :] = 0.0625
            bf[2, 0, :, :] = 0.0625
            bf[2, 2, :, :] = 0.0625
            return bf

        def bf2():
            # blur filter
            bf = np.zeros([3, 3, 1, 1])
            bf[1, 1, :, :] = 0.50
            bf[0, 1, :, :] = 0.125
            bf[1, 0, :, :] = 0.125
            bf[2, 1, :, :] = 0.125
            bf[1, 2, :, :] = 0.125
            bf[0, 0, :, :] = 0.0625
            bf[0, 2, :, :] = 0.0625
            bf[2, 0, :, :] = 0.0625
            bf[2, 2, :, :] = 0.0625
            return bf

        def bf3():
            # blur filter
            bf = np.zeros([3, 3, 1, 1])
            bf[1, 1, :, :] = 0.25
            bf[0, 1, :, :] = 0.125
            bf[1, 0, :, :] = 0.125
            bf[2, 1, :, :] = 0.125
            bf[1, 2, :, :] = 0.125
            bf[0, 0, :, :] = 0.0625
            bf[0, 2, :, :] = 0.0625
            bf[2, 0, :, :] = 0.0625
            bf[2, 2, :, :] = 0.0625
            return bf

        def laplace1():
            # Laplacian filter
            bf = np.zeros([3, 3, 1, 1])
            bf[1, 1, :, :] = -4
            bf[0, 1, :, :] = 1
            bf[1, 0, :, :] = 0
            bf[2, 1, :, :] = 1
            bf[1, 2, :, :] = 1
            bf[0, 0, :, :] = 0
            bf[0, 2, :, :] = 0
            bf[2, 0, :, :] = 0
            bf[2, 2, :, :] = 0
            return bf

        def sobel_x():
            # sobel_x filter
            bf = np.zeros([3, 3, 1, 1])
            bf[0, 0, :, :] = -1
            bf[0, 1, :, :] = 0
            bf[0, 2, :, :] = 1
            bf[1, 0, :, :] = -2
            bf[1, 1, :, :] = 0
            bf[1, 2, :, :] = 2
            bf[2, 0, :, :] = -1
            bf[2, 1, :, :] = 0
            bf[2, 2, :, :] = 1
            return bf

        def sobel_y():
            # sobel_y filter
            bf = np.zeros([3, 3, 1, 1])
            bf[0, 0, :, :] = -1
            bf[0, 1, :, :] = -2
            bf[0, 2, :, :] = -1
            bf[1, 0, :, :] = 0
            bf[1, 1, :, :] = 0
            bf[1, 2, :, :] = 0
            bf[2, 0, :, :] = 1
            bf[2, 1, :, :] = 2
            bf[2, 2, :, :] = 1
            return bf

        def emboss():
            # embos filter
            bf = np.zeros([3, 3, 1, 1])
            bf[0, 0, :, :] = -2
            bf[0, 1, :, :] = -1
            bf[0, 2, :, :] = 0
            bf[1, 0, :, :] = -1
            bf[1, 1, :, :] = 1
            bf[1, 2, :, :] = 1
            bf[2, 0, :, :] = 0
            bf[2, 1, :, :] = 1
            bf[2, 2, :, :] = 2
            return bf

        def convolve(img, kernel):
            with tf.Graph().as_default():

                #strides = [1, 2, 2, 1]  # [batch, height, width, channels]
                strides=[1, 1, 1, 1]
                #pooling=[1, 3, 3, 1]
                padding = 'SAME'

                num_maps = 1  # set number of maps to 1
                #img = img.convert('L', (0.2989, 0.5870, 0.1140, 0))  # convert to gray scale
                # reshape image to have a leading 1 dimension
                img = np.asarray(img, dtype='float32') / 256.
                img_shape = img.shape
                img_reshaped = img.reshape(1, img_shape[0], img_shape[1], num_maps)

                x = tf.placeholder('float32', [1, None, None, num_maps])
                w = tf.get_variable('w', initializer=tf.to_float(kernel))

                # operations
                conv = tf.nn.conv2d(x, w, strides=strides, padding=padding)
                #sig = tf.sigmoid(conv)
                avg_pool = tf.nn.avg_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding=padding)
                #avg_pool = tf.nn.avg_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding=padding)

                init = tf.global_variables_initializer()

                with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
                    session.run(init)

                    #conv_op, sigmoid_op, avg_pool_op = session.run([conv, sig, avg_pool], feed_dict={x: img_reshaped})
                    conv_op, avg_pool_op = session.run([conv, avg_pool], feed_dict={x: img_reshaped})

                    conv = tf.nn.conv2d(avg_pool_op, w, strides=strides, padding=padding)

                    #sig = tf.sigmoid(conv)
                    #avg_pool = tf.nn.avg_pool(sig, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding=padding)

                    #avg_pool = tf.nn.avg_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding=padding)
                    avg_pool = tf.nn.avg_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding=padding)

                    #conv_op1, sigmoid_op1, avg_pool_op1 = session.run([conv, sig, avg_pool],feed_dict={x: img_reshaped})
                    conv_op1, avg_pool_op1 = session.run([conv, avg_pool],feed_dict={x: img_reshaped})
                    conv = tf.nn.conv2d(avg_pool_op1, w, strides=strides, padding=padding)

                    sig = tf.sigmoid(conv)

                    #avg_pool = tf.nn.avg_pool(sig, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding=padding)
                    avg_pool = tf.nn.avg_pool(sig, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding=padding)

                    #conv_op2, sigmoid_op2, avg_pool_op2 = session.run([conv, sig, avg_pool],feed_dict={x: img_reshaped})
                    conv_op2, avg_pool_op2 = session.run([conv, avg_pool],feed_dict={x: img_reshaped})
                    # plt.imshow(np.reshape(conv_op2.flatten(),[38,38]))
                    # plt.show()
                    #print(avg_pool_op2.shape)
                return avg_pool_op2

        sf = sf1()
        op1 = convolve(img, sf)

        bf = bf1()
        op4 = convolve(img, bf)

        bf = bf1()
        op2 = convolve(img, bf)

        lp = laplace1()
        lp_c = convolve(img, lp)

        sbx = sobel_x()
        sbx_c = convolve(img, sbx)

        sby = sobel_y()
        sby_c = convolve(img, sby)

        # res1 = list(op1.flatten()) + list(op2.flatten()) + list(op3.flatten()) + list(op4.flatten()) + list(op5.flatten()) + list(op6.flatten())
        FV = list(op1.flatten()) + list(op4.flatten()) + list(lp_c.flatten()) + list(sbx_c.flatten()) + list(sby_c.flatten())

        # FV = list(op1.flatten()) + list(op2.flatten())
        #print(len(FV))
        return FV