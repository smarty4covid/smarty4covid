import os 
import sys
import librosa
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from scipy import spatial
from scipy import signal
import statistics
import itertools 
import scipy as sp
from scipy.interpolate import interp1d
from librosa import power_to_db,amplitude_to_db,stft
from audio_type_classifier_inference import *

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering


from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering


sr = 48000
#### model

ms = Multitimescale(m_small_pth='audio_type_short.h5',m_large_pth='audio_type_long.h5')
def get_prediction_mask(filename):
    predictions = ms.predict(filename, return_seq = True)
    masks = [np.zeros(predictions.shape[0]),np.zeros(predictions.shape[0]),np.zeros(predictions.shape[0])]
    for i, pred in enumerate(predictions):
        masks[np.argmax(pred)][i] = 1
        
    return masks


### compression 
def arctan_compressor(x, factor=2):
    constant = np.linspace(-1, 1, 1000)
    transfer = np.arctan(factor * constant)
    transfer /= np.abs(transfer).max()
    return apply_transfer(x, transfer)

def apply_transfer(signal, transfer, interpolation='linear'):
    constant = np.linspace(-1, 1, len(transfer))
    interpolator = interp1d(constant, transfer, interpolation)
    return interpolator(signal)
###

def get_not_silent_parts(filename, pad = 0, top_db = 10):
    y, sr = librosa.load(filename, sr = None) # read file
#     y = arctan_compressor(y, factor = 5)
    
    masks = get_prediction_mask(filename)
    mask = masks[1]
    mask_points = get_points_from_mask(mask)
    const = len(y)/len(mask)

    for s, e in mask_points:
        if (e - s) * const < 3*sr:
            mask[s:e] = 0

    for i in range (len(y)):
        j = int (i / const)
        if mask[j] == 0:
            y[i] = 0
    clips = librosa.effects.split(y, top_db=top_db, frame_length = sr//100) # find the clips where we dont have silence
    pad = int (pad * sr)
    points = []
    
    y_masked = []
    for c in clips:
        if c[1] - c[0] < (sr // 2):
            continue
        start_index = max(0, c[0] - pad)
        end_index = min(len ( y ) ,  c[1] + pad)
        data = y[start_index: end_index]
        points.append ((start_index, end_index))
        y_masked.append(data)
    return y, points, y_masked


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def get_vector_of_part(y):
    _, _, Sxx = signal.spectrogram(y, sr)
    S_dB = power_to_db(Sxx, ref=np.max)
    S_dB = (S_dB+80)/80
    return normalize_data(np.sum(S_dB, axis = 1))


def find_outlier(a):
    a = np.array(a)
    mean, stdev = np.mean(a, axis=0), np.std(a, axis=0)
    # Mean: [811.2  76.4  88. ]
    # Std: [152.97764543   6.85857128  29.04479299]


    ## Find Outliers
    outliers = ((np.abs(a[:,0] - mean[0]) > stdev[0])
                * (np.abs(a[:,1] - mean[1]) > stdev[1])
                * (np.abs(a[:,2] - mean[2]) > stdev[2]))
    ## Result
#     print(outliers)
    
    
    
def get_masks(filename, top_db = 50, algo = "AffinityPropagation"):
    y, points, y_masked = get_not_silent_parts(filename, top_db = top_db)

    vectors = []
    for part in y_masked:
        vector = get_vector_of_part(part)
        vectors.append(vector)
        
        
    exhale = np.zeros(len(y))
    inhale = np.zeros(len(y))
    
    if len(vectors) < 2:
        return y, inhale, exhale
    
    if algo == "kmeans":
        try:
            num_cl = 2
            clustering = KMeans(n_clusters=num_cl, random_state=0).fit(np.array(vectors))
            predictions = clustering.labels_
        except:
            return y, inhale, exhale
    elif algo == "AffinityPropagation":
        try:
            clustering = AffinityPropagation(random_state=0).fit(np.array(vectors))
            num_cl = len(clustering.cluster_centers_)
            predictions = clustering.labels_
        except:
            return y, inhale, exhale
        
    elif algo == "AgglomerativeClustering":
        num_cl = 2
        clustering = AgglomerativeClustering().fit(np.array(vectors))
        
    elif algo == "SpectralClustering":
        from sklearn.cluster import SpectralClustering
        sc = SpectralClustering(2, affinity='precomputed', n_init=100, assign_labels='discretize')
        dists = np.zeros((len(vectors), len(vectors)))
        for i in range (len(vectors)):
            for j in range (len(vectors)):
                if i != j:
                    dists[i][j] = spatial.distance.cosine(vectors[i], vectors[j])
        predictions = sc.fit_predict(dists)  
        num_cl = 2
        
    
    if num_cl == 1 or num_cl == 2:
        label_0 = 0
        label_1 = 1
    else:
        labels = {i:0 for i in range (num_cl)}
        for i in range (num_cl):
            labels[i] += list(predictions).count(i)
        labels = [[k, v] for k, v in sorted(labels.items(), key=lambda item: item[1], reverse = True)]
        label_0 = labels[0][0]
        label_1 = labels[1][0]
#     print(f"Estimated number of clusters: {num_cl}, label_0 = {label_0}, label_1 = {label_1}")
    
    
    for i, label in enumerate(predictions):
        if label == label_0:
            inhale[points[i][0]: points[i][1]] = 1
        elif label == label_1:
            exhale[points[i][0]: points[i][1]] = 1
    return y, inhale, exhale


#     plt.plot(y)
#     plt.plot(inhale)
#     plt.plot(exhale)


def get_points_from_mask(mask):
    points = []
    last_s = 0
    for i, s in enumerate(mask):
        if last_s == 0 and s == 1:
            starting_point = i

        elif last_s == 1 and s == 0:
            points.append((starting_point, i))

        elif last_s == 0 and s == 0:
            starting_point = 0

        last_s = s
    return points



def simple_breath_rate(inhale, exhale, add_contraints = False):
    brs = []
    inhale_points = get_points_from_mask(inhale)
    exhale_points = get_points_from_mask(exhale)
    for i in range (len(inhale_points) -1):
        br = 60 / ((inhale_points[i + 1][0] - inhale_points[i][0]) / sr)
        brs.append(br)
        br = 60 / ((inhale_points[i + 1][1] - inhale_points[i][1]) / sr)
        brs.append(br)


    for i in range (len(exhale_points) -1):
        br = 60 / ((exhale_points[i + 1][0] - exhale_points[i][0]) / sr)
        brs.append(br)
        br = 60 / ((exhale_points[i + 1][1] - exhale_points[i][1]) / sr)
        brs.append(br)    

#     print (brs)
    return np.mean(brs)

def filter_parts(signal):
    mask = np.zeros((len(signal)))
    points  = get_points_from_mask(signal)
    
    for start, end in points:
        length = (end - start) / sr
        if length > 0.2 and length < 15:
            mask[start: end] = 1
            
    return mask 

def breath_rate_constraints(inhale, exhale, add_contraints = False):
    brs = []
    
    # filter parts if a inhale or exhale is to short ot to big
    inhale = filter_parts(inhale)
    exhale = filter_parts(exhale)
    
    inhale_points = get_points_from_mask(inhale)
    exhale_points = get_points_from_mask(exhale)
    for i in range (len(inhale_points) -1):
        
        br = 60 / ((inhale_points[i + 1][0] - inhale_points[i][0]) / sr)
        brs.append(br)
        br = 60 / ((inhale_points[i + 1][1] - inhale_points[i][1]) / sr)
        brs.append(br)
        
        

    for i in range (len(exhale_points) -1):
        br = 60 / ((exhale_points[i + 1][0] - exhale_points[i][0]) / sr)
        brs.append(br)
        br = 60 / ((exhale_points[i + 1][1] - exhale_points[i][1]) / sr)
        brs.append(br)    

    return np.mean(brs)



def contraints(start, end, signal):
    length = (end - start) / sr
        
    if length < 1 and length > 20:
        return False
    
    if length < 2 and len(get_points_from_mask(signal[start: end])) > 0: # must have an exhale point
        return False
    
    if length < 5 and len(get_points_from_mask(signal[start: end])) == 0: # must have an exhale point
        return False

    if length > 15 and len(get_points_from_mask(signal[start: end])) == 0:
        return False
    
    if len(get_points_from_mask(signal[start: end])) > 1:
        return False
    
    return True

def single_breath_rate(start, end, signal):
    length = (end - start) / sr
    if not contraints(start, end, signal):
        return -1
    return (60 / length)


def differential_filtering(inhale, exhale):
    
    breath_rates = []
    inhale = filter_parts(inhale)
    exhale = filter_parts(exhale)
    
    inhale_points = get_points_from_mask(inhale)
    exhale_points = get_points_from_mask(exhale)
    
    count = np.zeros(len(inhale))
#     print (inhale_points, exhale_points)
    
    ll = []
    for i in range (len(inhale_points) -1):
        if (inhale_points[i + 1][0] - inhale_points[i][1]) < sr:
            continue
            
        breath_rate = single_breath_rate(inhale_points[i][0], inhale_points[i + 1][0], exhale)
        if breath_rate > 0:
#             count[inhale_points[i][0]: inhale_points[i + 1][0]] = 1
#             ll.append([inhale_points[i][0], inhale_points[i + 1][0], breath_rate])
            breath_rates.append(breath_rate)
        
        breath_rate = single_breath_rate(inhale_points[i][1], inhale_points[i + 1][1], exhale)
        if breath_rate > 0:
#             count[inhale_points[i][1]: inhale_points[i + 1][1]] = 1
#             ll.append([inhale_points[i][1], inhale_points[i + 1][1], breath_rate])
            breath_rates.append(breath_rate)
            
            
    for i in range (len(exhale_points) -1):
        if (exhale_points[i + 1][0] - exhale_points[i][1]) < sr:
            continue
            
        breath_rate = single_breath_rate(exhale_points[i][0], exhale_points[i + 1][0], inhale)
        if breath_rate > 0:
#             count[exhale_points[i][0]: exhale_points[i + 1][0]] = 1
#             ll.append([exhale_points[i][0], exhale_points[i + 1][0], breath_rate])
            breath_rates.append(breath_rate)
        
        breath_rate = single_breath_rate(exhale_points[i][1], exhale_points[i + 1][1], inhale)
        if breath_rate > 0:
#             count[exhale_points[i][1]: exhale_points[i + 1][1]] = 1
#             ll.append([exhale_points[i][1], exhale_points[i + 1][1], breath_rate])
            breath_rates.append(breath_rate)
        
#     breath_rates = [b for b in breath_rates if b > 10]
#     print ("Mean: ", np.mean(breath_rates))
#     print (ll)
    return breath_rates


def get_breath_rate(filename):
    y, inhale, exhale = get_masks(filename)
    inhale = filter_parts(inhale)
    exhale = filter_parts(exhale)
    if sum(inhale) != 0 or sum(exhale) != 0:
        pr = differential_filtering(inhale, exhale)
        if pr == []:
            return np.nan
        l = np.array([int(p) for p in pr])
        pr = l[(l>np.quantile(l,0.1)) & (l<np.quantile(l,0.9))].tolist()
        return np.mean(pr)
    return np.nan





def parse_files(folder_name):
    """
    Parsing folder and return a dictonary of filenames -> breath_rate
    """
    resp = []
    patient_ids = os.listdir(folder_name)
    for pid in tqdm(patient_ids):
        qids = [f for f in os.listdir(os.path.join(folder_name, pid)) if "questionnaire" in f and ".json" in f]
        for qid in qids:
            filename = os.path.join(folder_name, pid, qid)
            with open(filename, "r") as f:
                data = json.load(f)
            if "BreathsPerMinute" not in data or data["BreathsPerMinute"] == None:
                continue
                
                
            breath_file = [f for f in os.listdir(os.path.join(folder_name, pid)) if data["QuestionnaireId"] in f and "breath_1" in f][0]
            breath_file = os.path.join(folder_name, pid, breath_file)
            resp.append([breath_file, data["BreathsPerMinute"]])
    return resp



def annotate_parts(signal, signal_1, signal_2, breath_area):
    points_1 = get_points_from_mask(signal_1)
    points_2 = get_points_from_mask(signal_2)
    
    m1, m2 = [], []
    d1, d2 = [], []
    for s, e in points_1:
        if sum(breath_area[s:e]) != 0:
            m1 = list(signal[s: e])
            d1.append((e - s)/sr)
            
    for s, e in points_2:
        if sum(breath_area[s:e]) != 0:
            m2 = list(signal[s: e])
            d2.append((e - s)/sr)

    if np.mean(m1) > np.mean(m2):
        return signal_1, signal_2
    else:
        exhale_to_inhale, inhale_to_exhale = metrics_extra(signal_2, signal_1)
        return signal_2, signal_1
    

def length_filter(signal, threshold = 1):
    points = get_points_from_mask(signal)
    for s, e in points:
        if (e - s) / sr < threshold:
            signal[s: e] = 0
    return signal


def normal_point_contraints(start, end):
    s1, e1 = start
    s2, e2 = end
    if (s2 - e1)/sr < 0.5 or (s2 - e1)/sr > 5:
        return False
    if (e2 - s1)/sr < 1 or (e2 - s1)/sr > 15:
        return False
    if (e2 - e1)/sr < 2 or (e2 - e1)/sr > 20:
        return False
    if (s2 - s1)/sr < 1 or (s2 - s1)/sr > 20:
        return False
    return True
    
    
def normal_pattern(inhale, exhale):
    inhale_points = get_points_from_mask(inhale)
    exhale_points = get_points_from_mask(exhale)
    
    breath_area = np.zeros(len(inhale))
    breath_rate = []
    for i in range (len(inhale_points)-1):
        s1, e1 = inhale_points[i]
        s2, e2 = inhale_points [i + 1]
        
        if normal_point_contraints(inhale_points[i], inhale_points[i + 1]) and len(get_points_from_mask(exhale[e1:s2])) == 1:
            breath_rate.append(points_to_breath_rate(s1, s2))
            breath_rate.append(points_to_breath_rate(e1, e2))
            breath_area[s1: e2] = -1
#             print (s1, e1, s2, e2)
    
    for i in range (len(exhale_points)-1):
        s1, e1 = exhale_points[i]
        s2, e2 = exhale_points [i + 1]
        
        if normal_point_contraints(exhale_points[i], exhale_points[i + 1]) and len(get_points_from_mask(inhale[e1:s2])) == 1:
            breath_rate.append(points_to_breath_rate(s1, s2))
            breath_rate.append(points_to_breath_rate(e1, e2))
            breath_area[s1: e2] = -1
#             print (s1, e1, s2, e2)
    
#     print (breath_rate, np.mean(breath_rate))
    return breath_area, breath_rate

def exhaleonly_point_contraints(start, end):
    s1, e1 = start
    s2, e2 = end
    if (s2 - e1)/sr < 1 or (s2 - e1)/sr > 5:
        return False
    if (e2 - s1)/sr < 1 or (e2 - s1)/sr > 15:
        return False
    if (e2 - e1)/sr < 2 or (e2 - e1)/sr > 20:
        return False
    if (s2 - s1)/sr < 1 or (s2 - s1)/sr > 20:
        return False
    return True
    
    
def only_exhale_patterns(inhale, exhale):
    inhale_points = get_points_from_mask(inhale)
    exhale_points = get_points_from_mask(exhale)
    
    breath_area = np.zeros(len(inhale))
    breath_rate = []
    
    for i in range (len(inhale_points)-1):
        s1, e1 = inhale_points[i]
        s2, e2 = inhale_points [i + 1]
        
        if exhaleonly_point_contraints(inhale_points[i], inhale_points[i + 1]) and len(get_points_from_mask(exhale[e1:s2])) == 0:
            breath_rate.append(points_to_breath_rate(s1, s2))
            breath_rate.append(points_to_breath_rate(e1, e2))
            breath_area[s1: e2] = -1
    
    for i in range (len(exhale_points)-1):
        s1, e1 = exhale_points[i]
        s2, e2 = exhale_points [i + 1]
        
        if exhaleonly_point_contraints(exhale_points[i], exhale_points[i + 1]) and len(get_points_from_mask(inhale[e1:s2])) == 0:
            breath_rate.append(points_to_breath_rate(s1, s2))
            breath_rate.append(points_to_breath_rate(e1, e2))
            breath_area[s1: e2] = -1
    
#     print (breath_rate, np.mean(breath_rate))
    return breath_area, breath_rate


def points_to_breath_rate(start, end):
    return 60/ ((end - start) / sr)


def get_valid_parts(filename):
    y, inhale, exhale = get_masks(filename)
    # contraints 
    # lengths of inhales or exhales, longer than 1 sec
    inhale = length_filter(inhale, 0.5)
    exhale = length_filter(exhale, 0.5)
    
    breath_area1, breath_rate1 = normal_pattern(inhale, exhale)
    breath_area2, breath_rate2 = only_exhale_patterns(inhale, exhale)
    conf = ((len(breath_rate1) + len(breath_rate2)),  np.std(breath_rate1 + breath_rate2))
    return y, inhale, exhale, breath_area1 + breath_area2, np.mean(breath_rate1 + breath_rate2), conf

def metrics_extra(exhale, inhale):
    inhale_points = get_points_from_mask(inhale)
    exhale_points = get_points_from_mask(exhale)
    
    exhale_to_inhale = []
    inhale_to_exhale = []
    for i in range (len(inhale_points)-1):
        s1, e1 = inhale_points[i]
        s2, e2 = inhale_points [i + 1]
        
        if normal_point_contraints(inhale_points[i], inhale_points[i + 1]) and len(get_points_from_mask(exhale[e1:s2])) == 1:
            inhale_to_exhale.append((s2 - e1) / sr)
            
    for i in range (len(exhale_points)-1):
        s1, e1 = exhale_points[i]
        s2, e2 = exhale_points [i + 1]
        
        if normal_point_contraints(exhale_points[i], exhale_points[i + 1]) and len(get_points_from_mask(inhale[e1:s2])) == 1:
            exhale_to_inhale.append((s2 - e1) / sr)
    
    return exhale_to_inhale, inhale_to_exhale 

def metrics(signal, signal_1, signal_2, breath_area):
    points_1 = get_points_from_mask(signal_1)
    points_2 = get_points_from_mask(signal_2)
    
    m1, m2 = [], []
    d1, d2 = [], []
    for s, e in points_1:
        if sum(breath_area[s:e]) != 0:
            m1 = list(signal[s: e])
            d1.append((e - s)/sr)
    for s, e in points_2:
        if sum(breath_area[s:e]) != 0:
            m2 = list(signal[s: e])
            d2.append((e - s)/sr)

    if np.mean(m1) > np.mean(m2):
        exhale_to_inhale, inhale_to_exhale = metrics_extra(signal_1, signal_2)
        return np.mean(d1), np.mean(d2), np.mean(exhale_to_inhale), np.mean(inhale_to_exhale)
    else:
        exhale_to_inhale, inhale_to_exhale = metrics_extra(signal_2, signal_1)
        return np.mean(d2), np.mean(d1), np.mean(exhale_to_inhale), np.mean(inhale_to_exhale)
    

def normal_breathing_features(inhale, exhale):
    inhale_points = get_points_from_mask(inhale)
    exhale_points = get_points_from_mask(exhale)

    breath_area = np.zeros(len(inhale))
    
    features = {
        "inhale/exhale": [],
        "inhale/total": [],
        "breath_packs": []
    }
    for i in range (len(inhale_points)-1):
        s1, e1 = inhale_points[i]
        s2, e2 = inhale_points [i + 1]
        
        
        exhale_point = get_points_from_mask(exhale[e1:s2])
        if normal_point_contraints(inhale_points[i], inhale_points[i + 1]) and len(exhale_point) == 1:
            exhale_start, exhale_end = list(exhale_point[0])
            
            inhale_dur = e1 - s1
            exhale_dur = exhale_end - exhale_start
            
            features["inhale/exhale"].append(inhale_dur / exhale_dur)
            features["inhale/total"].append(inhale_dur / (s2-s1))
            features["breath_packs"].append([s1, e1, e1 + exhale_start, e1 + exhale_end, s2, e2])
    
    for i in range (len(exhale_points)-1):
        s1, e1 = exhale_points[i]
        s2, e2 = exhale_points [i + 1]
        
        
        inhale_point = get_points_from_mask(inhale[e1:s2])
        if normal_point_contraints(exhale_points[i], exhale_points[i + 1]) and len(inhale_point) == 1:
            inhale_start, inhale_end = list(inhale_point[0])
            
            exhale_dur = e1 - s1
            inhale_dur = inhale_end - inhale_start
            
            features["inhale/exhale"].append(inhale_dur / exhale_dur)
            features["inhale/total"].append(inhale_dur / (s2-s1))
            features["breath_packs"].append([s1, e1, e1 + inhale_start, e1 + inhale_end, s2, e2])

    return features
    