
import time,datetime
import numpy as np
from collections import Counter
import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torch.nn import Sequential
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader

def onehot_matrix(samples_vec, num_classes):
    """
    >>> onehot_matrix(np.array([1, 0, 3]), 4)
    [[ 0.  1.  0.  0.]
     [ 1.  0.  0.  0.]
     [ 0.  0.  0.  1.]]

    >>> onehot_matrix(np.array([2, 2, 0]), 3)
    [[ 0.  0.  1.]
     [ 0.  0.  1.]
     [ 1.  0.  0.]]

    Ref: http://bit.ly/1VSKbuc
    """
    num_samples = samples_vec.shape[0]

    onehot = np.zeros(shape=(num_samples, num_classes))
    onehot[range(0, num_samples), samples_vec] = 1

    return onehot

def segment_based_evaluation(y_pred, segment_ids, segment2label):
    """
    @argments:
    y_pred: predicted labels of frames
    segment_ids: segment id of frames
    segment2label: mapping from segment id to label
    """
    seg_pred = {}
    for frame_id, seg_id in enumerate(segment_ids):
        if seg_id not in seg_pred:
            seg_pred[seg_id] = []
        seg_pred[seg_id].append(y_pred[frame_id])

    ncorrect = 0
    for seg_id in seg_pred.keys():
        predicted = seg_pred[seg_id]
        c = Counter(predicted)
        predicted_label = c.most_common()[0][0] # take the majority voting

        if predicted_label == segment2label[seg_id]:
            ncorrect += 1

    accuracy = ncorrect/len(segment2label)
    print('Segment-based Accuracy using %d testing samples: %f' % (len(segment2label), accuracy))

device = ('cuda:2' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        ### BEGIN YOUR CODE (10 points)
        # 特征维度、隐层神经元、
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=32*2, out_features=32), # 双向LSTM
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=32, out_features=7)
        )
        ### END YOUR CODE

    def get_optimizer(self):
        ### BEGIN YOUR CODE (5 points)
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-2)
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0) # T_max表示周期的1/2 eta_min表示学习率变化的最小值
        ### END YOUR CODE
        return optimizer
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:,-1,:] # 最后一个时刻的结果
        x = self.fc(x)
        return x
    

class PhonemeClassifier(object):
    def __init__(self):
        unique_phonemes = ['CL', 'SF', 'VS', 'WF', 'ST', 'NF', "q"]
        self.labels = unique_phonemes
        self.train_epoch = 20

    def label_to_ids(self, y):
        y_ = [self.labels.index(label) for label in y]
        return y_

    def id_to_label(self, y):
        y_ = [self.labels[i] for i in y]
        return y_

    
    def train(self, X_train, y_train):
        y_train = self.label_to_ids(y_train)
        y_train = np.asarray(y_train)
        
        ### BEGIN YOUR CODE (15 points)
        X_train = torch.from_numpy(X_train).float().reshape((-1, 39, 1))
        y_train = torch.Tensor(y_train).long()
        
        dataset = TensorDataset(X_train, y_train)
        dataset = DataLoader(dataset, batch_size=4096, shuffle=True)
        
        self.model = Model()
        self.model = self.model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        self.optimizer = self.model.get_optimizer()
        self.model.train()
        for epoch in range(self.train_epoch):
            total_loss = 0
            for data in dataset:
                x, y = data
                x, y = x.to(device), y.to(device)
                y_pred = self.model(x)
                loss = loss_fn(y_pred, y)
                total_loss += loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        ### END YOUR CODE

    def test(self, X_test, y_test):
        """
        @arguments:
        X_test: #frames, #features (39 for mfcc)
        y_test: frame labels
        """
        ### BEGIN YOUR CODE (5 points)
        X_test = torch.from_numpy(X_test).float().reshape((-1, 39, 1))
        X_test = X_test.to(device)
        self.model.eval()
        label_pred = self.model(X_test)
        out_classes = torch.argmax(label_pred, axis=1)
        ### END YOUR CODE
        
        out_classes = self.id_to_label(out_classes) # from id to string
        out_classes = np.asarray(out_classes)
        acc = sum(out_classes == y_test) * 1.0 / len(out_classes)
        print('Frame-based Accuracy using %d testing samples: %f' % (X_test.shape[0], acc))

        return out_classes
