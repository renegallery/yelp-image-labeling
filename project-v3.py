""" Northwestern MSiA 490-30 Project code, Spring 2016
    This code demonstrates the use of transfer learning to speed up
    the training process for a convolutional neural network.
    author: Kedi Wu, Luyao Yang, Yilei Li, Yung Jen Kung
"""    
import os, numpy as np, random, time
import csv
from keras.optimizers import SGD, RMSprop, Adagrad, Adam

# Base settings
basepath = "train_crop_64"
outpath  = "predictions"
epoch_size = 8000
test_size  = 2000
imnorm   = 1.0          # Image normalization factor

batch_size, nb_epoch = 32, 100

# User tweakable settings
cnnlayers     = 4        # Total groups of CNN layers to create
vggxfer       = False     # Enables VGG weight transfer
vgglayers     = 2        # Number of first layers to load VGG weights into
freeze_conv   = True     # Freeze convolutional layers
fclayersize   = 512      # Size of fully connected layers
fclayers      = 1        # Number of fully connected layers
fcdropout     = 0.5      # Dropout factor for fully connected layers
normalize     = True     # Normalize input to zero mean
batchnorm     = True     # Batch normalization
loadmodel     = True     # Save/load models to reduce training time
savemodel     = True     # Save/load models to reduce training time
grid_size     = 3        # Combine this many photos into each
vis_filter    = False    # Show filter viz?
label_num     = 3        # how many labels to train

optimizer = Adagrad() #SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#%% Load train data
def load_link(path):
    with open(path) as linkfile:
        reader = csv.reader(linkfile)
        mydict = {row[0]:[row[i+2] for i in range(label_num)] for row in reader}
    return mydict
    
def load_group(path):
    mydict = {}
    with open(path) as linkfile:
        reader = csv.reader(linkfile)
        for row in reader:
            key = row[1]
            if key in mydict:
                mydict[key].append(row[0])
            else:
                mydict[key] = [row[0]]
    return mydict
        
def load_data(basepath, reference, restaurant, number=4000, grid_size=3):
    """Loads image data using folder names as class names
       Beware: make sure all images are the same size, or resize them manually"""
    import scipy.misc
    xdata, ydata = [], [[] for i in range(label_num)]
    obj_classes = [ ["Not Good for Lunch", "Good for Lunch"],
                    ["Not Good for Dinner", "Good for Dinner"],
                    ["Not Take Reservation", "Take Reservation"]]
    unitkeys = list(restaurant.keys())
    unit = [random.choice(unitkeys) for i in range(number)]
    for business in unit:
        if len(restaurant[business]) < (grid_size**2): continue
        photo_ids = random.sample(restaurant[business], grid_size**2)
        images = [scipy.misc.imread(os.path.join(basepath, photo + '.jpg')) \
            for photo in photo_ids]
        #images = [np.swapaxes(im, 0, 2) for im in images]
        rows = []
        for i in range(grid_size):
            rows.append(np.hstack(images[(i*grid_size):(i*grid_size+3)]))
        combined = np.vstack(rows)
        combined = np.swapaxes(combined, 0, 2)
        
        xdata.append(combined)
        [ydata[i].append(reference[photo_ids[0]]) for i in range(label_num)]
        if int(time.time()*1000) % 250 == 0:
            print("Progress: " + str(len(xdata)) + " shape= " + str(combined.shape), end='\r')
    
    print("Loaded %d samples" % len(xdata))
    shuffle_ind = list(range(len(xdata)))
    random.shuffle(shuffle_ind)
    return np.array(xdata, dtype='float32')[shuffle_ind] / imnorm, np.array(ydata, dtype='float32')[shuffle_ind], obj_classes

# Load test data
reference = load_link("train_photo_linked.csv")
units = load_group("train_photo_linked.csv")
from keras.utils import np_utils
X_test, y_test, obj_classes = load_data(basepath, reference, units, test_size, grid_size)
if normalize: 
    X_test = X_test - X_test.mean()
    X_test = X_test / X_test.std()
img_channels, img_rows, img_cols = X_test.shape[1:]
Y_test  = [np_utils.to_categorical(y_test[i], len(obj_classes[i])) for i in range(label_num)]

#%% Define a VGG-compatible model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K

model = Sequential()

# This layer is used for visualizing filters. Don't remove it.
model.add(ZeroPadding2D((1, 1), batch_input_shape=(1, 3, img_rows, img_cols)))
first_layer = model.layers[-1]
input_img = first_layer.input

# VGG net definition starts here. Change the cnnlayers to set how many layers to transfer
if cnnlayers >= 1:
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1', input_shape=(img_channels, img_rows, img_cols)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))

if cnnlayers >= 2:
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))

if cnnlayers >=3:
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))

if cnnlayers >= 4:
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))

if cnnlayers >= 5:
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# VGG net definition ends here
# All layers past this convert the VGG convolutional "code" into a classification

# Flatten and normalize data here, we don't know the data distribution of
# the preloaded training weights, so batch normalization helps fix slowdowns
model.add(Flatten())
from keras.layers.normalization import BatchNormalization
if batchnorm: model.add(BatchNormalization())

submodels = []
for i in range(label_num):
    submodel = Sequential()
    submodel.add(model)
    for l in range(1, fclayers + 1):
        submodel.add(Dense(fclayersize, name='fc%d' % l))
        submodel.add(Activation('relu'))
        submodel.add(Dropout(fcdropout)) # Modify dropout as necessary
    
    if batchnorm: submodel.add(BatchNormalization())
    submodel.add(Dense(len(obj_classes)))
    submodel.add(Activation('softmax'))
    submodel.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    submodels.append(submodel)

#%% Load existing VGG weights
if vggxfer:
    weights_file = "vgg16_weights.h5" # Pretrained VGG weights
    if os.path.exists(weights_file):
        print("Found existing weights file, loading data...")
        import h5py
        f = h5py.File(weights_file)
        if 'layer_names' in f.attrs.keys(): print("Weights file has:", f.attrs['layer_names'])
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers) or 'conv' not in model.layers[k].name:
                #print("Skipping layer:", k, model.layers[k].name if k < len(model.layers) else "<none>")
                # Commented layers are skipped                  
                continue
            # only load requested number of layers            
            if int(model.layers[k].name[4]) > vgglayers:
                continue
            print("Transferring layer:", k, model.layers[k].name, model.layers[k].output_shape)        
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            
            model.layers[k].set_weights(weights)
            
            if freeze_conv and int(model.layers[k].name[4]) <= vgglayers: 
                model.layers[k].trainable = False
        f.close()

# Print model summary
model.summary()
print("Trainable layers:", model.trainable_weights)

#%% Visualization code
import matplotlib.pyplot as plt
layer_dict = dict([(layer.name, layer) for layer in model.layers])

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x = x*0.1 + 0.5
    x = np.clip(x, 0, 1) * 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def viz_filter_max(layer_name, filter_index=0, max_steps=150):
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, filter_index, :, :])
    grads = K.gradients(loss, input_img)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([input_img], [loss, grads])
    step = 1e-0
    input_img_data = np.random.random((1, 3, img_rows, img_cols)) * 20 + 128.
    tm = time.time()
    for i in range(max_steps):
        if (time.time() - tm > 5) and (i % 10 == 0): print(i, '/', max_steps)
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        if time.time() - tm > 1:
            plt.text(0.1, 0.1, "Filter viz timeout", color='red')
            break
    img = input_img_data[0]
    img = deprocess_image(img)
    fig = plt.imshow(img)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    return layer_output

def viz_losses(losses, scores, accuracies, epoch=0):
    f, (pl1, pl2) = plt.subplots(1, 2, sharey=False, figsize=(12, 5))
    pl1.plot(np.log(losses), label='Train')
    pl1.plot(np.log(scores), label='Test')
    pl2.plot(accuracies, label='Test accuracies')
    pl1.set_title("Log Loss")
    pl2.set_title("Test Accuracy")
    pl1.legend()
    plt.show()

def viz_filters(nbfilters=3):
    for layer_name in sorted(layer_dict.keys()):
        if not hasattr(layer_dict[layer_name], 'nb_filter'): continue
        nfilters = layer_dict[layer_name].nb_filter
        print("Layer", layer_name, "has", nfilters, "filters")
        plt.subplots(1, nbfilters)
        for j in range(nbfilters):
            plt.subplot(1, nbfilters, j + 1)
            viz_filter_max(layer_name, random.randint(0, nfilters-1))
        plt.show()

def test_prediction(im=None, y=None):
    t_img = random.randint(0, len(X_train) - 1)
    if im is None: im, y = X_train[t_img], Y_train[t_img]
    plt.imshow((im.T - im.min()) / (im.max() - im.min()))
    plt.show()
    pred = model.predict_proba(np.expand_dims(im, 0))
    cls = np.argmax(y)
    print("Actual: %s(%d)" % (obj_classes[cls], cls))
    for cls in list(reversed(np.argsort(pred)[0]))[:5]:
        conf = float(pred[0, cls])/pred.sum()
        print("    predicted: %010s(%d), confidence=%0.2f [%-10s]" % (obj_classes[cls], cls, conf, "*" * int(10*conf)))
    return pred
    
#%% Image data augmentation 
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    featurewise_center=False,               # set input mean to 0 over the dataset
    samplewise_center=False,                # set each sample mean to 0
    featurewise_std_normalization=False,    # divide inputs by std of the dataset
    samplewise_std_normalization=False,     # divide each input by its std
    zca_whitening=False,                    # apply ZCA whitening
    rotation_range=0,                       # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,                  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,                 # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,                  # randomly flip images
    vertical_flip=False)                    # randomly flip images
datagen.fit(X_test)

#%% Training code
if loadmodel and os.path.exists("model.h5"): model.load_weights("model.h5")
#from keras.utils import generic_utils
losses, scores, accuracies = [], [], []
for e in range(nb_epoch):
    print('---- Epoch', e, ' ----')
    # Load data subset if needed
    X_train, y_train, obj_classes = load_data(basepath, reference, units, epoch_size, grid_size)
    Y_train = np_utils.to_categorical(y_train, len(obj_classes))
    if normalize: 
        X_train = X_train - X_train.mean()
        X_train = X_train / X_train.std()

    print('Sample proportion = %.3f train, %.3f test' % 
        (Y_train[:,1].sum() / Y_train[:,1].size, Y_test[:,1].sum() / Y_test[:,1].size))
    
    print('Training...')      
    loss = model.fit_generator(datagen.flow(X_train, Y_train, shuffle=True,
                    batch_size=batch_size),
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch=1,
                    validation_data=(X_test, Y_test))               
    if savemodel: model.save_weights("model.h5", overwrite=True)
    
    losses.append(loss.history['loss']), scores.append(loss.history['val_loss']), accuracies.append(loss.history['val_acc'])
    viz_losses(losses, scores, accuracies, e)
    t_ind = random.randint(0, len(X_train) - 1)
    test_prediction(X_train[t_ind], Y_train[t_ind])
    if e % 20 == 0 and vis_filter:
        try:
            print("Visualizing filters, press CTRL-C to stop...")
            viz_filters()
        except KeyboardInterrupt:
            pass

#%%
"""
# Use datagen.flow() below for test or train images
# prediction_label = True to use predictions as labels
# mode = 'color' as default, 'gray' for grayscale, 'both' for color next to gray
def write_predictions(images, prediction_label=True, mode='color'):
    if not os.path.exists(outpath): os.makedirs(outpath)
    for i, (bx, by) in enumerate(images):
        for im, y in zip(bx, by):
            print(im.shape)
            im = (im.T - im.min()) / (im.max() - im.min())
            if mode == 'color': plt.imshow(im)
            elif mode == 'gray': plt.imshow(im, cmap='gray')
            elif mode == 'both':
                plt.subplots(1, 2)
                plt.subplot(1, 2, 1)
                plt.imshow(im)
                plt.subplot(1, 2, 2)
                plt.imshow(im, cmap='gray')
            pred = model.predict_proba(np.expand_dims(im, 0))
            cls = np.argmax(pred) if label == 'prediction' else np.argmax(y)
            conf = int(100*float(pred[0, cls])/pred.sum())        
            clsname = obj_classes[cls]
            clspath = os.path.join(outpath, clsname)
            if not os.path.exists(clspath): os.makedirs(clspath)
            plt.savefig(os.path.join(clspath, "%d-conf-%d.png" % (i, conf)))

write_predictions(datagen.flow(X_test, Y_test), prediction_label=False, mode='color')
"""
    