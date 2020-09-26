# ### Motion Deblurring using CNN (tensorflow 1.x)
# Uniform motion deblurring을 Convolutional Neural Network(CNN)을 이용하여 구현해 본다.
# Idea 1) blurred - CNN - deblurred

# Jupyter notebook에서 matplotlib 사용할 때, 필요
# get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.data import Dataset, Iterator
from PIL import Image
print ("Package Loaded")

# Hyperparameter
batch_size = 64
training_rate = 0.000001
epochs = 30000

# input_parser 정의 (input & ground-truth)
def input_parser(img_path, label_path):
    reader = tf.WholeFileReader()
    img_file = tf.read_file(img_path)
    img_decoded = tf.cast(tf.image.decode_image(img_file,channels=1),tf.float32)
    label_file =tf.read_file(label_path)
    label_decoded = tf.cast(tf.image.decode_image(label_file,channels=1),tf.float32)
    return img_decoded, label_decoded

# training & validation 경로
train_imgs = tf.constant([("./dataset/train_in/%d_bi.png" % i) for i in range(1,40001)])
label_imgs = tf.constant([("./dataset/train_out/%d_gt.png" % i) for i in range(1,40001)])

dataset = tf.data.Dataset.from_tensor_slices((train_imgs,label_imgs))
dataset = dataset.map(input_parser)
dataset = dataset.shuffle(buffer_size=20000)
dataset = dataset.repeat()
dataset = dataset.batch(batch_size)

iterator = Iterator.from_structure(dataset.output_types,dataset.output_shapes)
training_init_op = iterator.make_initializer(dataset)
next_element = iterator.get_next()

#valid_imgs = tf.constant([("./dataset/validation_bp/%d_bi.png" % i) for i in range(1,501)])
#valid_label_imgs = tf.constant([("./dataset/validation_gp/%d_gt.png" % i) for i in range(1,501)])

#dataset2 = Dataset.from_tensor_slices((valid_imgs,valid_label_imgs))
#dataset2= dataset2.map(input_parser)
#dataset2 = dataset2.shuffle(buffer_size=20000)
#dataset2 = dataset2.repeat()
#dataset2 = dataset2.batch(batch_size)

#iterator2 = Iterator.from_structure(dataset2.output_types,dataset2.output_shapes)
#training_init_op2 = iterator2.make_initializer(dataset2)
#next_element2 = iterator2.get_next()

# CNN model
# 1. 입력층, 출력층 및 13개의 은닉층으로 구성
# 2. 각 층의 커널의 크기는 처음 두층은 128, 나머지 은닉층은 64, 출력층은 1을 사용
inputs = tf.placeholder(tf.float32, (None, 64, 64, 1), name='inputs')
labels = tf.placeholder(tf.float32, (None, 64, 64, 1), name='labels')

conv1 = tf.layers.conv2d(inputs, 128, (3,3), padding='same', activation=None)
conv1 = tf.nn.relu(conv1)

conv2 = tf.layers.conv2d(conv1, 128, (3,3), padding='same', activation=None)
conv2 = tf.nn.relu(conv2)

conv3 = tf.layers.conv2d(conv2, 64, (3,3), padding='same', activation=None)
conv3 = tf.nn.relu(conv3)

conv4 = tf.layers.conv2d(conv3, 64, (3,3), padding='same', activation=None)
conv4 = tf.nn.relu(conv4)

conv5 = tf.layers.conv2d(conv4, 64, (3,3), padding='same', activation=None)
conv5 = tf.nn.relu(conv5)

conv6 = tf.layers.conv2d(conv5, 64, (3,3), padding='same', activation=None)
conv6 = tf.nn.relu(conv6)

conv7 = tf.layers.conv2d(conv6, 64, (3,3), padding='same', activation=None)
conv7 = tf.nn.relu(conv7)

conv8 = tf.layers.conv2d(conv7, 64, (3,3), padding='same', activation=None)
conv8 = tf.nn.relu(conv8)

conv9 = tf.layers.conv2d(conv8, 64, (3,3), padding='same', activation=None)
conv9 = tf.nn.relu(conv9)

#conv10 = tf.layers.conv2d(conv9, 64, (3,3), padding='same', activation=None)
#conv10 = tf.nn.relu(conv10)

#conv11 = tf.layers.conv2d(conv10, 64, (3,3), padding='same', activation=None)
#conv11 = tf.nn.relu(conv11)

#conv12 = tf.layers.conv2d(conv11, 64, (3,3), padding='same', activation=None)
#conv12 = tf.nn.relu(conv12)

#conv13 = tf.layers.conv2d(conv12, 64, (3,3), padding='same', activation=None)
#conv13 = tf.nn.relu(conv13)

logits = tf.layers.conv2d(conv9, 1, (3,3), padding='same', activation=None)
logits = tf.identity(logits, name='logits')

# Loss function & Optimization
costc=tf.losses.mean_squared_error(labels,logits)
optc = tf.train.AdamOptimizer(training_rate).minimize(costc)

# Weight 저장 경로
save_file = './a/model_gs1.ckpt'
saver = tf.train.Saver()

# Learning
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(training_init_op)
    #sess.run(training_init_op2)
    
    for e in range(epochs):
        input_tensor,label_tensor = sess.run(next_element)
        inpss = input_tensor.reshape((-1, 64, 64, 1))
        labss = label_tensor.reshape((-1, 64, 64, 1))
        batch_cost, _ = sess.run([costc, optc], feed_dict = {inputs: inpss, labels: labss})
        if (e%100 == 0):                        
            print("Epoch: {}/{}...".format(e+1, epochs), "loss: {:.6f} ".format(batch_cost))
    # 학습 종료 후 가중치 저장
    saver.save(sess, save_file)

# 1장 Test
# Blur된 영상에 Guided filtering or Shock filtering을 전처리한 영상을 입력으로 사용한다.

# file 경로
test_imgs = tf.constant("./dataset/test_bp/Bblur_1.png")
ground_imgs = tf.constant("./dataset/test_bp/gt_1.png")
base_t = tf.constant("./dataset/test_bp/Bblur_1.png")

img_filec = tf.read_file(test_imgs)
img_decodedc = tf.cast(tf.image.decode_image(img_filec, channels=1),tf.float32)

img_fileg = tf.read_file(ground_imgs)
img_decodedg = tf.cast(tf.image.decode_image(img_fileg, channels=1),tf.float32)

img_fileb = tf.read_file(base_t)
img_decodedb = tf.cast(tf.image.decode_image(img_fileb, channels=1),tf.float32)

# 테스트 전 영상 조정하기
with tf.Session() as sess:
    test_input = sess.run(img_decodedc)
    ground_input = sess.run(img_decodedg)
    base_input = sess.run(img_decodedb)
    sess.close()

img = Image.open("./dataset/test_bp/Bblur_1.png")
test_n, test_m = img.size
base_tt = Image.open("./dataset/test_bp/Bblur_1.png")
print ("height: {} width: {}".format(test_m, test_n))

test_input = test_input.reshape((test_m, test_n))
ground_input = ground_input.reshape((512,512))

# test 입력 영상 출력
plt.imshow(test_input, cmap='Greys_r')
plt.show()

# 출력 구하기
# 테스트 영상을 Neural Network에 입력하여 출력을 구한다.
save_model_path = './a/model_gs1.ckpt'
loaded_graph = tf.Graph()

with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph(save_model_path + '.meta')
    loader.restore(sess, save_model_path)
    
    inputs = loaded_graph.get_tensor_by_name('inputs:0')
    labels = loaded_graph.get_tensor_by_name('labels:0')
    logits = loaded_graph.get_tensor_by_name('logits:0')
    
    # padding
    p_n = 64 - test_n%64
    p_m = 64 - test_n%64
    if (test_n%64 == 0):
        p_n = 0
    if (test_m%64 == 0):
        p_m = 0
    
    result_img = np.zeros([test_m+p_m, test_n+p_n])
    #(위,아래) (왼 ,오)
    test_inputp = np.pad(test_input,((16,16+p_m),(16,16+p_n)),'reflect')
    
    b_n = int((test_n+p_n)/32)
    b_m = int((test_m+p_m)/32)
    
    print ("height: {} width: {}".format(b_m, b_n))
    
    for i in range(b_n):
        for j in range(b_m):
            image_patch = np.zeros([64,64])
            for x in range(64):
                for y in range(64):
                    image_patch[y, x] = test_inputp[j*32+y,i*32+x]
            
            image_patch = image_patch.reshape((-1,64,64,1))
            resultc = sess.run(logits, feed_dict = {inputs: image_patch}) 
            resulta = resultc.reshape((64,64))
                       
            for x in range(32):
                for y in range(32):
                    result_img[(j*32)+y,(i*32)+x] = resulta[y+16,x+16]
                    
    result_imgs = np.zeros([test_m, test_n])
    for i in range(test_n):
        for j in range(test_m):
            result_imgs[j, i] = result_img[j,i]    
            
    plt.imshow(result_imgs, cmap='Greys_r')

# 결과 영상 출력
# 순서대로 결과영상, 테스트 입력 영상(shock filtering 후 영상), 디블러된 영상
plt.imshow(ground_input, cmap='Greys_r')
plt.show()
plt.imshow(test_input, cmap='Greys_r')
plt.show()
plt.imshow(result_imgs.reshape((test_m, test_n)), cmap='Greys_r')
plt.show()

# 결과 영상 저장
im = Image.fromarray(result_imgs)
if im.mode != 'RGB':
    im = im.convert('RGB')
im.save("./dataset/test_bp/result3_1.png")

# 테스트 영상 전체 저장
for da in range(51,81):
    for db in range(1,9):
        
        test_imgs = tf.constant("./dataset/pre/Bblur_%d_%d.png" %(da, db))
        img_filec = tf.read_file(test_imgs)
        img_decodedc = tf.cast(tf.image.decode_image(img_filec,channels=1),tf.float32)

        with tf.Session() as sess:
            test_input  = sess.run(img_decodedc)
            sess.close()

        img = Image.open("./dataset/pre/Bblur_%d_%d.png" %(da, db))
        test_n, test_m = img.size
        base_tt = Image.open("./dataset/test_bp/Bblur_1.png")

        print ("height: {} width: {}".format(test_m, test_n))

        test_input = test_input.reshape((test_m, test_n))

        save_model_path = './a/model_gs1.ckpt'
        loaded_graph = tf.Graph()


        with tf.Session(graph=loaded_graph) as sess:
            loader = tf.train.import_meta_graph(save_model_path + '.meta')
            loader.restore(sess, save_model_path)

            inputs = loaded_graph.get_tensor_by_name('inputs:0')
            labels = loaded_graph.get_tensor_by_name('labels:0')
            logits = loaded_graph.get_tensor_by_name('logits:0')

            # padding
            p_n = 64 - test_n%64
            p_m = 64 - test_n%64
            if (test_n%64 == 0):
                p_n = 0
            if (test_m%64 == 0):
                p_m = 0

            result_img = np.zeros([test_m+p_m, test_n+p_n])
            #(위,아래) (왼 ,오)
            test_inputp = np.pad(test_input,((16,16+p_m),(16,16+p_n)),'reflect')

            b_n = int((test_n+p_n)/32)
            b_m = int((test_m+p_m)/32)

            print ("height: {} width: {}".format(b_m, b_n))

            for i in range(b_n):
                for j in range(b_m):
                    image_patch = np.zeros([64,64])
                    for x in range(64):
                        for y in range(64):
                            image_patch[y, x] = test_inputp[j * 32 + y, i * 32 + x]

                    image_patch = image_patch.reshape((-1, 64, 64, 1))
                    resultc = sess.run(logits, feed_dict = {inputs: image_patch}) 
                    resulta = resultc.reshape((64, 64))

                    for x in range(32):
                        for y in range(32):
                            result_img[(j * 32) + y, (i * 32) + x] = resulta[y + 16, x + 16]

            result_imgs = np.zeros([test_m, test_n])
            for i in range(test_n):
                for j in range(test_m):
                    result_imgs[j, i] = result_img[j, i]

            im = Image.fromarray(result_imgs)
            if im.mode != 'RGB':
                im = im.convert('RGB')

            im.save("./dataset/CNN/cnn_%d_%d.png" %(da, db))