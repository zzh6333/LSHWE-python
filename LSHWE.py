from nltk.corpus import stopwords
stop = set(stopwords.words('english')) #stop words
from nltk.stem.porter import PorterStemmer
import re
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import tensorflow as tf
from lshash import LSHash

window_size = 2
learning_rate = 0.01
training_epoch = 50
hashsize = 30
dim = 25
friends = 5
rarewords_threshold = 2
input_file = '##############################'
output_file = "##############################"

taglist = []
def removestopwords(text):
    after_remove = list()
    total_words = 0
    clean_sentence = []
    text = re.sub("[\~|`|\!|\@|\$|\%|\^|\&|\*|\(|\)|\-|\_|\+|\=|\||\'|\|\[|\]|\{|\}|\;|\:|\"|\,|\<|\.|\>|\/|\?]", "",
                   text)
    for word in text.split():
        if word not in stop:
            if re.search('http', word) or re.search('www', word):
                continue
            elif re.search('\#', word):
                if word not in taglist:
                    taglist.append(word)
                word = re.sub("\#\S+", "tagid" + str(taglist.index(word)), word)
                clean_sentence.append(str.lower(word))
            else:
                word = re.sub(r'@\w+', "username", word)
                clean_sentence.append(str.lower(word))
    total_words += clean_sentence.__len__()
    after_remove.append(" ".join(after_remove))
    return clean_sentence

def stemming(text):
    porter_stemmer = PorterStemmer()
    sentence = []
    for word in text:
        word = porter_stemmer.stem(word)
        sentence.append(str(word))
    return sentence

text_x = []
with open(input_file) as labelfile:
    for line in labelfile:
        text = line.split("\n")[0]
        text_removestop = removestopwords(str(text))
        wordstem = stemming(text_removestop)
        text_x.append(wordstem)

def dict(x):
    dic = []
    counts = []
    for item in x:
        for word in item:
            if word in dic:
                index = dic.index(word)
                counts[index] += 1
            else:
                dic.append(word)
                counts.append(1)
    return dic, counts

dic, counts = dict(text_x)
wordcount = len(dic)
print "total words:",wordcount

rarewords = []
for i in range(wordcount):
    if counts[i] < rarewords_threshold:
        rarewords.append(i)
print "rare words:", len(rarewords)

cooc_matr = np.zeros([wordcount, wordcount])
for item in text_x:
    for i in range(len(item)):
        cenword = item[i]
        cen_id = dic.index(cenword)
        for j in range(i - window_size, i + window_size + 1):
            if j in range(len(item)):
                val = window_size + 1 - abs(i - j)
                coword = item[j]
                co_id = dic.index(coword)
                cooc_matr[cen_id][co_id] += val

for i in range(wordcount):
    cooc_matr[i] = cooc_matr[i] / cooc_matr[i][i] * 1.0

lsh = LSHash(hashsize, wordcount)
for i in range(len(cooc_matr)):
    lsh.index(cooc_matr[i])

friendslist = []
sim_matr = []
for i in range(len(rarewords)):
    result = lsh.query(cooc_matr[rarewords[i]], num_results=friends, distance_func='euclidean')
    length = len(result)
    fri = []
    sim = []
    for j in range(length):
        fri.append(str(np.array(result[j][0])))
        sim.append(float(result[j][1]))
    if length < friends:
        count = friends - length
        for i in range(count):
            fri.append(str(8))
            sim.append(-1)
    friendslist.append(fri)
    sim_matr.append(sim)

X = tf.placeholder("float", [wordcount, wordcount])
sim_true = tf.placeholder("float", [len(rarewords), friends])
sim_pred = tf.placeholder("float", [len(rarewords), friends])
n_hidden_1 = 256
n_hidden_2 = dim

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([wordcount, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, wordcount])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([wordcount])),
}

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

cooc_true = X
encoder_op = encoder(X)
cooc_pred = decoder(encoder_op)
cost = tf.reduce_mean(tf.pow(cooc_true - cooc_pred, 2))
cost_min = cost + tf.reduce_mean(tf.pow(sim_true - sim_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_min)

if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.graph.finalize()
    sess.run(init)
    for epoch in range(training_epoch):
        word_list = sess.run(encoder_op, feed_dict={X: cooc_matr})

        lsh = LSHash(hashsize, dim)
        for item in word_list:
            lsh.index(item)

        cos = []
        for i in range(len(rarewords)):
            result = lsh.query(word_list[rarewords[i]], num_results=friends, distance_func='euclidean')
            cosin = -1 * np.ones(friends)
            for item in result:
                id = str(np.array(item[0]))
                for j in range(friends):
                    if id == friendslist[i][j]:
                        cosin[j] = float(item[1])
            cos.append(cosin)

        cost_fin, _ = sess.run([cost_min, optimizer], feed_dict={sim_pred: cos, sim_true: sim_matr, X: cooc_matr})
        print "epoch:",epoch, ",cost=", "{:.9f}".format(cost_fin)
    print("Optimization Finished!")
    encoder_result = sess.run(encoder_op, feed_dict={X: cooc_matr})

with open(output_file, "w") as f:
    for i in range(wordcount):
        f.write(str(dic[i]) + " ")
        for item in encoder_result[i]:
            f.write(str(item) + " ")
        f.write("\n")
