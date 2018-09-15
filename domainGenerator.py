# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 21:34:38 2018

@author: ljh
"""
import numpy as np
import os
import time
import tensorflow as tf

tf.enable_eager_execution()


def make_dictionary():
    words_dic = [chr(i) for i in range(32, 127)]
    words_dic.insert(0, 'None')  # 补0用的
    words_dic.append("unknown")
    words_redic = dict(zip(words_dic, range(len(words_dic))))  # 反向字典
    print('字表大小:', len(words_dic))
    return words_dic, words_redic


# char2idx = {u:i for i, u in enumerate(unique)}
# idx2char = {i:u for i, u in enumerate(unique)}


# 字符到向量
def ch_to_v(datalist, words_redic, normal=1):
    to_num = lambda word: words_redic[word] if word in words_redic else len(words_redic) - 1  # 字典里没有的就是None
    data_vector = []
    for ii in datalist:
        data_vector.append(list(map(to_num, list(ii))))
    # 归一化
    if normal == 1:
        return np.asarray(data_vector) / (len(words_redic) / 2) - 1
    return np.array(data_vector)


def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    # print("sequences",sequences)
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
    # print("lengths",lengths)
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    sample_shape = tuple()
    for s in sequences:
        # print("s",np.asarray(s).shape[1:])
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)

    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths


# 样本数据预处理（用于训练）
def getbacthdata(batchx, charmap):
    batchx = ch_to_v(batchx, charmap, 0)
    sampletpad, sampletlengths = pad_sequences(batchx)  # 都填充为最大长度
    zero = np.zeros([len(batchx), 1])
    tarsentence = np.concatenate((sampletpad[:, 1:], zero), axis=1)
    return np.asarray(sampletpad, np.int32), np.asarray(tarsentence, np.int32), sampletlengths


inv_charmap, charmap = make_dictionary()
# print("inv_charmap",inv_charmap,"charmap",charmap)
vocab_size = len(charmap)

DATA_DIR = './仿冒APPLE样本.txt'  # 定义载入的样本路径
input_text = []
f = open(DATA_DIR, encoding='UTF-8')
for i in f:
    a = i.replace('\n', '').split("\t")
    t = filter(lambda x: len(x) > 0, a)  # 没有内容的过滤掉
    # print("t",t)
    input_text = input_text + list(t)
    # print("input_text",input_text)
    # print(input_text[-2:])

input_text, target_text, sampletlengths = getbacthdata(input_text, charmap)
print("input_text", input_text.shape)
print("target_text", target_text.shape)

max_length = len(input_text[0])
learning_rate = 0.001
# the embedding dimension 
embedding_dim = 256
# number of RNN (here GRU) units
units = 1024
# batch size 
BATCH_SIZE = 6
dataset = tf.data.Dataset.from_tensor_slices((input_text, target_text)).shuffle(1000)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, batch_size):
        super(Model, self).__init__()
        self.units = units
        self.batch_sz = batch_size

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.mcell_fw = [tf.contrib.rnn.GRUCell(units) for _ in range(3)]

        self.mcell_bw = [tf.contrib.rnn.GRUCell(units) for _ in range(3)]

        # if tf.test.is_gpu_available():
        #   self.gru = tf.keras.layers.CuDNNGRU(self.units,
        #                                       return_sequences=True,
        #                                       return_state=True,
        #                                       recurrent_initializer='glorot_uniform')
        # else:
        #   self.gru = tf.keras.layers.GRU(self.units,
        #                                  return_sequences=True,
        #                                  return_state=True,
        #                                  recurrent_activation='sigmoid',
        #                                  recurrent_initializer='glorot_uniform')

        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden_fw, hidden_bw):
        x = self.embedding(x)
        # output shape == (batch_size, max_length, hidden_size)
        # states shape == (batch_size, hidden_size)

        # states variable to preserve the state of the model
        # this will be used to pass at every step to the model while training
        output, states_fw, states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(self.mcell_fw, self.mcell_bw,
                                                                                          x,
                                                                                          initial_states_fw=hidden_fw,
                                                                                          initial_states_bw=hidden_bw,
                                                                                          dtype=tf.float32)

        # output, states = self.gru(x, initial_state=hidden)

        # reshaping the output so that we can pass it to the Dense layer
        # after reshaping the shape is (batch_size * max_length, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # The dense layer will output predictions for every time_steps(max_length)
        # output shape after the dense layer == (max_length * batch_size, vocab_size)
        x = self.fc(output)

        return x, states_fw, states_bw


model = Model(vocab_size, embedding_dim, units, BATCH_SIZE)
optimizer = tf.train.AdamOptimizer()


# using sparse_softmax_cross_entropy so that we don't have to create one-hot vectors
def loss_function(real, preds):
    return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
latest_cpkt = tf.train.latest_checkpoint(checkpoint_dir)
if latest_cpkt:
    print('Using latest checkpoint at ' + latest_cpkt)
    checkpoint.restore(latest_cpkt)
else:
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

def train():
    EPOCHS = 20

    for epoch in range(EPOCHS):
        start = time.time()

        # initializing the hidden state at the start of every epoch
        totaloss = []
        for (batch, (inp, target)) in enumerate(dataset):
            hidden_fw = model.reset_states()
            hidden_bw = model.reset_states()
            with tf.GradientTape() as tape:
                # feeding the hidden state back into the model
                # This is the interesting step
                predictions, hidden_fw, hidden_bw = model(inp, hidden_fw, hidden_bw)

                # reshaping the target because that's how the
                # loss function expects it
                target = tf.reshape(target, (-1,))
                loss = loss_function(target, predictions)
                totaloss.append(loss)

            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             loss))
        # saving (checkpoint) the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1, np.mean(totaloss)))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Evaluation step(generating text using the model learned)

def genertor():
    for iii in range(20):

        # You can change the start string to experiment
        # start_string = 'w'
        # input_eval = [charmap[s] for s in start_string]
        # input_eval = tf.expand_dims(input_eval, 0)
        # print(input_eval)
        input_eval = input_text[np.random.randint(len(input_text))][0]
        start_string = inv_charmap[input_eval]
        # print(start_string)

        input_eval = tf.expand_dims([input_eval], 0)
        # print(input_eval)

        # empty string to store our results
        text_generated = ''

        # low temperatures results in more predictable text.
        # higher temperatures results in more surprising text
        # experiment to find the best setting
        temperature = 1.0

        # hidden state shape == (batch_size, number of rnn units); here batch size == 1

        hidden_fw = [tf.zeros((1, units)), tf.zeros((1, units)), tf.zeros((1, units))]
        hidden_bw = [tf.zeros((1, units)), tf.zeros((1, units)), tf.zeros((1, units))]

        # hidden_fw=None
        # hidden_bw=None

        for i in range(max_length):
            predictions, hidden_fw, hidden_bw = model(input_eval, hidden_fw, hidden_bw)
            hidden_fw=list(hidden_fw)
            hidden_bw=list(hidden_bw)
            # using a multinomial distribution to predict the word returned by the model
            predictions = predictions / temperature
            predicted_id = tf.multinomial(predictions, num_samples=1)[0][0].numpy()
            if predicted_id == 0:
                break

            # We pass the predicted word as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated += inv_charmap[predicted_id]

        print(start_string + text_generated)


if __name__ == '__main__':
    train()
    genertor()
