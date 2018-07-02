# -*- coding: utf-8 -*-

import os
import csv
import tensorflow as tf

dirname = os.path.dirname(__file__)

x = {"0": "테스트", "1": "tt"}
# for key in x:
#   print(key , " ", x[key])

# with open('test.csv', 'w', newline='') as csvfile:
#   writer = csv.writer(csvfile)
#   for key in x:
#     writer.writerow([key, x[key]])

# with open('test.csv') as csvfile:
#   result = {}
#   reader = csv.reader(csvfile)
#   for row in reader:
#     result[row[0]] = row[1]

# print(result)

# # x = tf.get_variable('x', shape=[2])
# x = tf.Variable([1,2], name="x")

# saver = tf.train.Saver()

# with tf.Session() as sess:
#   init = tf.global_variables_initializer()
#   sess.run(init)

  
#   # save_path = saver.save(sess, os.path.join(dirname, "tmp/test.ckpt"))
#   # print("Model saved in path: %s" % save_path)

#   saver.restore(sess, os.path.join(dirname, "tmp/test.ckpt"))
#   print("Model restored.")
#   # Check the values of the variables
#   print(x.eval())
