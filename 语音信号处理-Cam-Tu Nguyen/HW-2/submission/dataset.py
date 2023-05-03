
import glob
import pickle
import librosa
import scipy
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn import preprocessing
import ipdb

unique_classes = ['CL', 'SF', 'VS', 'WF', 'ST', 'NF', "q"]

def read_data_folder(data_path):
    """
    @return
    wav_files: list of file paths to WAV files in the train or val folder.
    labels_files: ist of file paths to PHNCLS files in the train or val folder.
    """
    # get all the WAV and PHNCLS in the folder data_path
    wav_files = sorted(glob.glob(data_path + "/*.WAV"))
    labels_files = sorted(glob.glob(data_path + "/*.PHNCLS"))
    return wav_files, labels_files

# 提取segment labels和representations
def extract_features(wavfile, label_file, first_seg_id=0, stanalysis=MFCCAnalysis()):
    """
    Extract segment labels and representations.
    
    @arguments:
    wavfile: path to wav file
    label_file: path to PHNCLS file
    first_seg_id: segment_id of the first segment of the current file.
                  When you process a list of files, you may want segment id to increase globally.

    @returns:
    X: #frames, #features
    y: #frames

    frame2seg: mapping from frame id to segment id
    y_seg: segment labels (segment-based groundtruth)
    """
    #  mfccs_and_deltas, hop_length, n_fft    
    X_st, hop_length, window_len = stanalysis.perform(wavfile)
    # print(hop_length, window_len) # 220 441

    seg_labels = {}
    point_seg_ids = []
    with open(label_file, 'r') as f:
        for line in f.readlines():
            start_frame, end_frame, label = line.split(' ')
            start_frame = int(start_frame)
            end_frame = int(end_frame)
            
            label = label.strip()
            segment_id = len(seg_labels) + first_seg_id
            seg_labels[segment_id] = label
            
            phn_frames = end_frame - start_frame
            # point_seg_ids stores segment ids for every sample point.
            point_seg_ids.extend([segment_id]*phn_frames) 
            

    X = []
    y = []
    frame_seg_ids = []
    curr_frame = curr_hop = 0
    
    while (curr_frame < (len(point_seg_ids) - window_len)):
        ### BEGIN YOUR CODE (10 points)
        
        # extract the segment ids for the sample points within the frame 
        # from curr_frame to curr_frame + window_len
        
        # Since one frame may overlap with more than one segment, 
        # sample points within the frame may be assigned with multiple segment ids.
        # We get the major segment id as the segment id corresponding to the current frame.
        # 主元素问题，采用摩尔投票算法，时间复杂度为 O(n)
        count = 0
        for i in range(curr_frame, curr_frame + window_len):
            if count:
                if point_seg_ids[i] == major:
                    count += 1
                else:
                    count -= 1
            else:
                major = point_seg_ids[i]
                count += 1
        segment_id = major
        ### END YOUR CODE

        label = seg_labels[segment_id]
        y.append(label)
        X.append(X_st[curr_hop,:])
        frame_seg_ids.append(segment_id)
        
        curr_hop += 1
        curr_frame += hop_length
    
    return X, y, frame_seg_ids, seg_labels


def prepare_data(wavfiles, label_files, stanalysis=MFCCAnalysis()):
    X = []
    y = []
    segment_ids = []
    seg2labels = {}
    
    file_seg_id = 0
    for i in tqdm(range(len(wavfiles))):
        wavfile = wavfiles[i]
        label_file = label_files[i]
        x_, y_, seg_ids_, seg_labels_ = extract_features(
            wavfile, label_file, first_seg_id=file_seg_id, stanalysis=stanalysis)

        file_seg_id += len(seg_labels_)
        for k,v in seg_labels_.items():
            seg2labels[k] = v

        X.append(x_)
        y.extend(y_)
        segment_ids.extend(seg_ids_)
        

    X = np.concatenate(X)
    return X, y, segment_ids, seg2labels
