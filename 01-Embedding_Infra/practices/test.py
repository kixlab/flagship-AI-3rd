# -*- coding: utf-8 -*-

import os
import tensorflow as tf

dirname = os.path.dirname(__file__)

# x = tf.get_variable('x', shape=[2])
x = tf.Variable([1,2], name="x")

saver = tf.train.Saver()

with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)

  
  # save_path = saver.save(sess, os.path.join(dirname, "tmp/test.ckpt"))
  # print("Model saved in path: %s" % save_path)

  saver.restore(sess, os.path.join(dirname, "tmp/test.ckpt"))
  print("Model restored.")
  # Check the values of the variables
  print(x.eval())
