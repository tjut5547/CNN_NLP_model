import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = tf.add(a, b)

sess = tf.Session()
with sess.as_default():
    sess.run(tf.global_variables_initializer())
    test = sess.run(c, feed_dict = {a : [1, 2, 34], b : [1, 2, 3]})
    print (test)
    print (type(test))
    sess.close()

Sess = tf.Session()
with Sess.as_default():
    # Sess.run(tf.global_variables_initializer())
    test = Sess.run(c, feed_dict = {a : [1, 2, 34], b : [1, 2, 3]})
    print (test)
    print (type(test))
    print (tf.get_default_session())
    Sess.close()

print (Sess)
print (sess)