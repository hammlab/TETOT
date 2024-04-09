import sys
import os
import numpy as np
import tensorflow as tf
from utils import eval_accuracy_disc_hinge, load_VLCS_source_TT, load_VLCS_targets, eval_entropy_disc
from utils import  marginal_WD1, marginal_WD2, conditional_WD1, conditional_WD2, pseudo_conditional_WD1, pseudo_conditional_WD2
from models import classificationNN, representationNN
import argparse
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
import socket
from torchvision import transforms
import imagenet_c_corruptions
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description='Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--METHOD', type=str, default="ERM", help='ERM')
parser.add_argument('--SOURCE', type=str, default="0", help='Source Model')
parser.add_argument('--TARGET', type=str, default="1", help='Target dataset')
parser.add_argument('--TRIAL', type=str, default="0", help='0')
args = parser.parse_args()

METHOD = args.METHOD
TRIAL = args.TRIAL
REP_DIM = 128

SRCS = [int(args.SOURCE)]
TRGS = [int(args.TARGET)]

print(SRCS, TRGS)

CHECKPOINT_PATH = "./checkpoints/vanilla_dg_resnet50_S_DG_" + METHOD + "_source_" + str(SRCS[0]) + "_trial_" + str(TRIAL) + "_" + str(REP_DIM)

if not os.path.exists(CHECKPOINT_PATH):
    sys.exit("No Model:"+CHECKPOINT_PATH)


BATCH_SIZE = 50
NUM_CLASSES = 5

HEIGHT = 224
WIDTH = 224
NCH = 3


# Load Dataset
print("Loading data")
src_train_data, src_train_labels, src_test_data, src_test_labels = load_VLCS_source_TT(SRCS)
target_data, target_labels = load_VLCS_targets(TRGS)

print("Loaded")
src_X_train = [item for sublist in src_train_data for item in sublist]
src_Y_train = [item for sublist in src_train_labels for item in sublist]

src_X_test = [item for sublist in src_test_data for item in sublist]
src_Y_test = [item for sublist in src_test_labels for item in sublist]

trg_X = [item for sublist in target_data for item in sublist]
trg_Y = [item for sublist in target_labels for item in sublist]

src_X_train = np.array(src_X_train, dtype=np.float32).reshape([-1, HEIGHT, WIDTH, NCH])
src_X_test = np.array(src_X_test, dtype=np.float32).reshape([-1, HEIGHT, WIDTH, NCH])
trg_X = np.array(trg_X, dtype=np.float32).reshape([-1, HEIGHT, WIDTH, NCH])

src_X_train = preprocess_input(src_X_train * 255)
src_X_test = preprocess_input(src_X_test * 255)

src_Y_train = tf.keras.utils.to_categorical(src_Y_train, NUM_CLASSES)
src_Y_test = tf.keras.utils.to_categorical(src_Y_test, NUM_CLASSES)
trg_Y = tf.keras.utils.to_categorical(trg_Y, NUM_CLASSES)

base_model = ResNet50(weights='imagenet', include_top=False, pooling="avg")
encoder = representationNN(2048, REP_DIM)
classifier = classificationNN(REP_DIM, NUM_CLASSES)

ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

ckpt = tf.train.Checkpoint(base_model = base_model, encoder = encoder, classifier = classifier)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=1) 
ckpt.restore(ckpt_manager.latest_checkpoint)


nb_batches_src = int(len(src_X_train)/BATCH_SIZE)
if len(src_X_train)%BATCH_SIZE!=0:
    nb_batches_src+=1

for batch in range(nb_batches_src):
    ind_batch = range(BATCH_SIZE * batch, min(BATCH_SIZE * (1+batch), len(src_X_train)))
    if batch == 0:
        encoded_source_images = encoder(base_model(src_X_train[ind_batch], training=False), training=False).numpy()
    else:
        encoded_source_images = np.concatenate([encoded_source_images, encoder(base_model(src_X_train[ind_batch], training=False), training=False).numpy()])
       
############## Corrupted Source Std Scaler ##############
src_scaler = StandardScaler()
src_scaler.fit(encoded_source_images)
encoded_source_images = src_scaler.transform(encoded_source_images)
############################

for corruption in imagenet_c_corruptions.CORRUPTIONS:
    
    print(corruption, CHECKPOINT_PATH)
    
    trg_corrupted_acc = []
    pseudo_1_conditional_w1_distances = []
    
    for severity in range(1, 6):
        
        corruption_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: corruption(x, severity)),
            transforms.ToTensor()])
        
        corrupted_images = []
        for i in range(len(trg_X)):
            new_img = corruption_transform(np.uint8(trg_X[i]*255)).permute(1,2,0).cpu().numpy()/255.
            corrupted_images.append(new_img)
        
        corrupted_images = preprocess_input(255 * np.array(corrupted_images))
        corrupted_images_labels = trg_Y


        nb_batches_trg = int(len(corrupted_images)/BATCH_SIZE)
        if len(corrupted_images)%BATCH_SIZE!=0:
            nb_batches_trg+=1
        
        for batch in range(nb_batches_trg):
            ind_batch = range(BATCH_SIZE * batch, min(BATCH_SIZE * (1+batch), len(corrupted_images)))
            if batch == 0:
                encoded_target_images = encoder(base_model(corrupted_images[ind_batch], training=False), training=False).numpy()
                target_pseudo_labels = classifier(encoder(base_model(corrupted_images[ind_batch], training=False), training=False).numpy(), training=False).numpy()
            else:
                encoded_target_images = np.concatenate([encoded_target_images, encoder(base_model(corrupted_images[ind_batch], training=False), training=False).numpy()])
                target_pseudo_labels = np.concatenate([target_pseudo_labels, classifier(encoder(base_model(corrupted_images[ind_batch], training=False), training=False).numpy(), training=False).numpy()])
          
        target_pseudo_labels = tf.nn.softmax(target_pseudo_labels).numpy()                
        
        trg_accuracy, trg_loss = eval_accuracy_disc_hinge(encoded_target_images, trg_Y, classifier)
        trg_entropy = eval_entropy_disc(encoded_target_images, classifier)
        
        trg_corrupted_acc.append(trg_accuracy)
        
        ############## Corrupted Target Std Scaler ##############
        corrupted_trg_scaler = StandardScaler()
        corrupted_trg_scaler.fit(encoded_target_images)
        encoded_target_images = corrupted_trg_scaler.transform(encoded_target_images)
        ############################
        
        # W1
        pseudo_1_conditional_WD1_val = pseudo_conditional_WD1(encoded_source_images, src_Y_train, encoded_target_images, target_pseudo_labels, 1E0)
        pseudo_1_conditional_w1_distances.append(pseudo_1_conditional_WD1_val)
        
        if "original" in str(corruption):
            break
        
    print(np.array2string(np.array(trg_corrupted_acc), separator=',')) 
    print(np.array2string(np.array(pseudo_1_conditional_w1_distances), separator=','))
    
print(args)