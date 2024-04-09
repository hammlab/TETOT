import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
#import ot
from scipy.spatial.distance import cdist 
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
import socket
from sklearn.preprocessing import StandardScaler
import ot
import scipy

tf.keras.backend.set_floatx('float32')
ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

HEIGHT = 224
WIDTH = 224
NCH = 3

def marginal_WD1(src_features, trg_features):
    
    feature_cost =  scipy.spatial.distance.cdist(src_features, trg_features, metric='euclidean')
    wd1 = ot.emd2(ot.unif(len(src_features)), ot.unif(len(trg_features)), feature_cost, numItermax=500000)
    return wd1

def marginal_WD2(src_features, trg_features):
        
    feature_cost =  scipy.spatial.distance.cdist(src_features, trg_features, metric='sqeuclidean')
    wd2 = ot.emd2(ot.unif(len(src_features)), ot.unif(len(trg_features)), feature_cost, numItermax=500000)
    return np.sqrt(wd2)

def conditional_WD1(src_features, src_labels, trg_features, trg_labels, NUM_CLASSES):
    
    wd_num = 0
    for k in range(NUM_CLASSES):
        idx_S = np.argwhere(np.argmax(src_labels, 1) == k).flatten()
        idx_T = np.argwhere(np.argmax(trg_labels, 1) == k).flatten()
        
        C = scipy.spatial.distance.cdist(src_features[idx_S], trg_features[idx_T], metric='euclidean')
            
        gamma = ot.emd(ot.unif(len(idx_S)), ot.unif(len(idx_T)), C)
        wd_num += np.sum(gamma * C) * (len(idx_S) / len(src_labels)) 
        
    WDs = wd_num
    return WDs

def conditional_WD2(src_features, src_labels, trg_features, trg_labels, NUM_CLASSES):
    
    wd_num = 0
    for k in range(NUM_CLASSES):
        idx_S = np.argwhere(np.argmax(src_labels, 1) == k).flatten()
        idx_T = np.argwhere(np.argmax(trg_labels, 1) == k).flatten()
        
        C = scipy.spatial.distance.cdist(src_features[idx_S], trg_features[idx_T], metric='sqeuclidean')
            
        gamma = ot.emd(ot.unif(len(idx_S)), ot.unif(len(idx_T)), C)
        wd_num += np.sum(gamma * C) * (len(idx_S) / len(src_labels)) 
        
    WDs = np.sqrt(wd_num)
    return WDs

def pseudo_conditional_WD1(src_features, src_labels, trg_features, trg_labels, LAMBDA):
    
    C_features = scipy.spatial.distance.cdist(src_features, trg_features, metric='euclidean')
    C_labels = scipy.spatial.distance.cdist(src_labels, trg_labels, metric='euclidean')
    
    C = C_features + LAMBDA * C_labels
    wd1 = ot.emd2(ot.unif(len(src_features)), ot.unif(len(trg_features)), C, numItermax=500000)
    
    return wd1
    

def pseudo_conditional_WD2(src_features, src_labels, trg_features, trg_labels, LAMBDA):
    
    C_features = scipy.spatial.distance.cdist(src_features, trg_features, metric='sqeuclidean')
    C_labels = scipy.spatial.distance.cdist(src_labels, trg_labels, metric='sqeuclidean')
    
    C = C_features + LAMBDA * C_labels
    wd2 = ot.emd2(ot.unif(len(src_features)), ot.unif(len(trg_features)), C, numItermax=500000)
    
    return np.sqrt(wd2)

def aug(x_in):
    x_in = tf.image.random_flip_left_right(x_in)
    x_in = tf.image.random_brightness(x_in, 0.3)
    x_in = tf.image.random_contrast(x_in, 1-0.3, 1+0.3)
    x_in = tf.image.random_saturation(x_in,1-0.3,1+0.3)
    x_in = tf.image.random_hue(x_in, 0.3).numpy()
    return x_in


def torch_bernoulli_(p, size):
    return (torch.rand(size) < p).float()

def torch_xor_(a, b):
    return (a - b).abs()

def load_PACS(sources, targets):
    
    root = "path/to/data/PACS"

    environments = [f.name for f in os.scandir(root) if f.is_dir()]
    environments = sorted(environments)
    #['art_painting', 'cartoon', 'photo', 'sketch']
    
    common_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    
    
    src_data = []
    src_labels = []
    src_test_data = []
    src_test_labels = []
    
    target_data = []
    target_labels = []

    for i, environment in enumerate(environments):
    
        if i in sources or i in targets:
            print(environment)
            path = os.path.join(root, environment)
            
            env_dataset = ImageFolder(path, transform=common_transform)
            
        
            loader = DataLoader(env_dataset, len(env_dataset))
            dataset_array = next(iter(loader))[0].permute(0, 2, 3, 1).numpy()
            dataset_labels_array = next(iter(loader))[1].numpy()
            
            
            if i in sources:
                print("\n\n\n Labels:", len(np.unique(dataset_labels_array)))
                for label in range(len(np.unique(dataset_labels_array))):
                    label_indices = np.argwhere(dataset_labels_array == label).flatten()
                    if label == 0:
                        train_indices = label_indices[:int(0.9 * len(label_indices))]
                        test_indices = label_indices[int(0.9 * len(label_indices)):]
                    else:
                        train_indices = np.concatenate([train_indices, label_indices[:int(0.9 * len(label_indices))]])
                        test_indices = np.concatenate([test_indices, label_indices[int(0.9 * len(label_indices)):]])
                        
                idxes = np.arange(len(train_indices))
                np.random.shuffle(idxes)
                train_indices = train_indices[idxes]
                
                
                src_data.append(dataset_array[train_indices])
                src_labels.append(dataset_labels_array[train_indices])
                src_test_data.append(dataset_array[test_indices])
                src_test_labels.append(dataset_labels_array[test_indices])
            
            elif i in targets:
                target_data.append(dataset_array)
                target_labels.append(dataset_labels_array)
        

    return src_data, src_labels, src_test_data, src_test_labels, target_data, target_labels

def load_PACS_sources(sources, BATCH_SIZE, val_split=True):
    
    root = "path/to/data/PACS"

    environments = [f.name for f in os.scandir(root) if f.is_dir()]
    environments = sorted(environments)
    #['art_painting', 'cartoon', 'photo', 'sketch']
    
    augment_transform = transforms.Compose([
        #transforms.Resize((224,224)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomGrayscale(0.1),
        transforms.ToTensor(),
        #transforms.Normalize(
        #    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    src_data_loaders = []
    val_data = []
    val_labels = []
    
    for i, environment in enumerate(environments):
    
        if i in sources:
            print("source:", environment)
            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path, transform=augment_transform)
            dataset_size = len(env_dataset)
            
            if val_split:
                train_dataset, valid_dataset = torch.utils.data.random_split(env_dataset, (int(0.9*dataset_size), dataset_size - int(0.9*dataset_size)))
            else:
                train_dataset = env_dataset
            
            loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            src_data_loaders.append(loader)
            
            if val_split:
                val_loader = DataLoader(valid_dataset, len(valid_dataset))
                val_data_array = next(iter(val_loader))[0].permute(0, 2, 3, 1).numpy()
                val_labels_array = next(iter(val_loader))[1].numpy()
                
                val_data.append(val_data_array)
                val_labels.append(val_labels_array)
    
        
    return src_data_loaders, val_data, val_labels
            
    

def load_PACS_sources_onetime(sources):
    
    root = "path/to/data/PACS"

    environments = [f.name for f in os.scandir(root) if f.is_dir()]
    environments = sorted(environments)
    
    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomGrayscale(0.1),
        transforms.ToTensor(),
    ])
    
    source_data = []
    source_labels = []
    val_data = []
    val_labels = []

    for i, environment in enumerate(environments):
    
        if i in sources:
            print("Source:", environment)
            path = os.path.join(root, environment)
            
            env_dataset = ImageFolder(path, transform=augment_transform)
            dataset_size = len(env_dataset)
            train_dataset, valid_dataset = torch.utils.data.random_split(env_dataset, (int(0.8*dataset_size), dataset_size - int(0.8*dataset_size)))
            
            train_loader = DataLoader(train_dataset, len(env_dataset))
            train_dataset_array = next(iter(train_loader))[0].permute(0, 2, 3, 1).numpy()
            train_dataset_labels_array = next(iter(train_loader))[1].numpy()
            
            source_data.append(train_dataset_array)
            source_labels.append(train_dataset_labels_array)
            
            val_loader = DataLoader(valid_dataset, len(valid_dataset))
            val_data_array = next(iter(val_loader))[0].permute(0, 2, 3, 1).numpy()
            val_labels_array = next(iter(val_loader))[1].numpy()
            
            val_data.append(val_data_array)
            val_labels.append(val_labels_array)
            
    return source_data, source_labels, val_data, val_labels


def load_PACS_targets(targets):
    
    root = "path/to/data/PACS"

    environments = [f.name for f in os.scandir(root) if f.is_dir()]
    environments = sorted(environments)
    
    common_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    
    target_data = []
    target_labels = []

    for i, environment in enumerate(environments):
    
        if i in targets:
            print("target:", environment)
            path = os.path.join(root, environment)
            
            env_dataset = ImageFolder(path, transform=common_transform)
            loader = DataLoader(env_dataset, len(env_dataset))
            dataset_array = next(iter(loader))[0].permute(0, 2, 3, 1).numpy()
            dataset_labels_array = next(iter(loader))[1].numpy()
            
            target_data.append(dataset_array)
            target_labels.append(dataset_labels_array)

    return target_data, target_labels


def eval_accuracy(x_test, y_test, base_model, encoder, classifier):
    correct = 0
    points = 0
    loss = 0
    batch_size = 20
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        rep = encoder(base_model(x_test[ind_batch], training=False), training=False)
        pred = classifier(rep, training=False)
        
        correct += np.sum(np.argmax(pred,1) == np.argmax(y_test[ind_batch],1))
        points += len(ind_batch)
        loss += np.sum(ce_loss_none(y_test[ind_batch], pred).numpy())
    
    return (correct / np.float32(points))*100., loss/ np.float32(points)

def eval_accuracy_pytorch(x_test, y_test, model, device):
    correct = 0
    points = 0
    loss = 0
    batch_size = 50
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    
    
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        data = torch.tensor(x_test[ind_batch]).to(device)
        pred = model(data)
        
        correct += np.sum(np.argmax(pred,1) == np.argmax(y_test[ind_batch],1))
        points += len(ind_batch)
        loss += np.sum(ce_loss_none(y_test[ind_batch], pred).numpy())
    
    return (correct / np.float32(points))*100., loss/ np.float32(points)

def eval_accuracy_all(x_test, y_test, classifier):
    correct = 0
    points = 0
    loss = 0
    batch_size = 50
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        pred = classifier(x_test[ind_batch], training=False)
        
        correct += np.sum(np.argmax(pred,1) == np.argmax(y_test[ind_batch],1))
        points += len(ind_batch)
        loss += np.sum(ce_loss_none(y_test[ind_batch], pred).numpy())
    
    return (correct / np.float32(points))*100., loss/ np.float32(points)

def eval_accuracy_disc_hinge(x_test, y_test, classifier):
    correct = 0
    points = 0
    loss = 0
    batch_size = 50
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        pred = tf.nn.softmax(classifier(x_test[ind_batch], training=False))
        
        correct += np.sum(np.argmax(pred,1) == np.argmax(y_test[ind_batch],1))
        points += len(ind_batch)
        
        real = tf.reduce_sum(y_test[ind_batch] * pred, 1)
        other = tf.reduce_max((1 - y_test[ind_batch]) * pred - (y_test[ind_batch] * 10000), 1)
        
        loss += np.sum(tf.nn.relu(other - real + 0.1).numpy())
        
    return (correct / np.float32(points))*100., loss/np.float32(points)

def eval_entropy_disc(x_test, y_test, classifier):
    points = 0
    entropy = 0
    batch_size = 50
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        pred_softmax = tf.nn.softmax(classifier(x_test[ind_batch], training=False))
        pred_log_softmax = tf.nn.log_softmax(classifier(x_test[ind_batch], training=False))
        
        entropy += tf.reduce_sum(-tf.reduce_sum(pred_softmax * pred_log_softmax, axis=-1)).numpy()
        
        points += len(ind_batch)
    
    return entropy/np.float32(points)



def plot_images(data, filename, rows=3, cols = 10):
    
    fig = plt.figure(1,(128, 10.))
    grid = ImageGrid(fig, 121, nrows_ncols=(rows, cols), axes_pad = 0.01)
    for i in range(rows*cols):
        grid[i].imshow(data[i])
        grid[i].axis('off')  
    plt.savefig('Plots/'+str(filename)+'.pdf', bbox_inches='tight')
    plt.close()


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def mini_batch_class_balanced(label, sample_size=20):
    ''' sample the mini-batch with class balanced
    '''
    label = np.argmax(label, axis=1)

    n_class = len(np.unique(label))
    index = []
    for i in range(n_class):
        
        s_index = np.argwhere(label==i).flatten()
        np.random.shuffle(s_index)
        index.append(s_index[:sample_size])

    index = [item for sublist in index for item in sublist]
    index = np.array(index, dtype=int)
    return index


def restore_original_image_from_array_vgg(x, data_format='channels_last'):
    mean = [103.939, 116.779, 123.68]

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] += mean[0]
            x[1, :, :] += mean[1]
            x[2, :, :] += mean[2]
        else:
            x[:, 0, :, :] += mean[0]
            x[:, 1, :, :] += mean[1]
            x[:, 2, :, :] += mean[2]
    else:
        x[..., 0] += mean[0]
        x[..., 1] += mean[1]
        x[..., 2] += mean[2]

    if data_format == 'channels_first':
        # 'BGR'->'RGB'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'BGR'->'RGB'
        x = x[..., ::-1]

    return np.array(np.clip(x/255., 0, 1))
