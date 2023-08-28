import pandas as pd
import json
from tqdm import tqdm
import os
import librosa
from librosa.feature import melspectrogram
from librosa import power_to_db,amplitude_to_db,stft
from librosa.effects import trim

import soundfile as sf
import numpy as np
from PIL import Image
from audioread.exceptions import NoBackendError

from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

from tensorflow.keras.layers import Dense, Activation, Add, Flatten, Input, Conv2D, Dropout,\
    AveragePooling2D, Concatenate, MaxPool2D, BatchNormalization, Reshape, Conv1D, MaxPool1D, Average,Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras
from tensorflow.keras import backend as K

import pickle




def is_cough_j(j_id):
    j_file = '/media/eddie/Big_Drive/COUGHVID/'+j_id+'.json'
    with open(j_file,'r') as f:
        di = json.load(f)
    if float(di['cough_detected'])>0.8:
        return True
    else:
        return False

class Multitimescale:
    def __init__(self, seq_len_small=128, seq_len_large=1024, m_small_pth = None, m_large_pth= None,sr=48000):
        self.seq_len_small=seq_len_small
        self.seq_len_large=seq_len_large
        self.sr = sr
        if m_small_pth is not None:
            self.m_small = load_model(m_small_pth)
            self.m_large = load_model(m_large_pth)

    def extract_features(self,audio_file,sr=48000):
        ## Name to save the spectrogram
        fname = audio_file.split('/')[-1][:-4]+'.png'

        audio,sr = librosa.load(audio_file,sr=sr)
        audio = trim(audio)[0]
        
        S = melspectrogram(audio,sr)
        S_dB = power_to_db(S, ref=np.max)
        S_dB = (S_dB+80)/80
        return S_dB

    def load_spectrogram(self,spec_file):
        img = np.asarray(Image.open(spec_file))/255.0
        return img

    def predict(self,inp,return_seq=False):
        if type(inp)==str:
            if '.png' in inp:
                x = self.load_spectrogram(inp)
            else:
                x = self.extract_features(inp)
        else:
            x = inp


        if x.shape[1]>self.seq_len_small:
            x_small = np.zeros((x.shape[1]-self.seq_len_small,x.shape[0],self.seq_len_small))
            for i in range(x.shape[1]-self.seq_len_small):
                x_small[i] = x[:,i:i+self.seq_len_small]
        else:
            x_small = np.zeros((1,x.shape[0],self.seq_len_small))
            x_small[0,:,:x.shape[1]] = x
        

        if x.shape[1]>self.seq_len_large:
            x_large = np.zeros((x.shape[1]-self.seq_len_large,x.shape[0],self.seq_len_large))
            for i in range(x.shape[1]-self.seq_len_large):
                x_large[i] = x[:,i:i+self.seq_len_large]
        else:
            x_large = np.zeros((1,x.shape[0],self.seq_len_large))
            x_large[0,:,:x.shape[1]] = x

        if return_seq:
            pred_small = np.zeros((x.shape[1],3))
            pred_small[self.seq_len_small//2:-self.seq_len_small//2] = self.m_small.predict(x_small)
            pred_large = np.zeros((x.shape[1],3))
            pred_large[self.seq_len_large//2:-self.seq_len_large//2] = self.m_large.predict(x_large)
            return pred_large+pred_small
        else:
            return np.mean(self.m_small.predict(x_small),axis=0)+np.mean(self.m_large.predict(x_large),axis=0)

    def predict_coughvid(self,coughvid_dir,valid=False,save_file = 'coughvid_multiscale_valid.csv'):
        if valid:
            fnames = [os.path.join(coughvid_dir,f) for f in os.listdir(coughvid_dir) if is_cough_j(f[:-4])]
        else:
            fnames = [os.path.join(coughvid_dir,f) for f in os.listdir(coughvid_dir)]

        files = list()
        preds_cough = list()
        preds_breath =list()
        preds_speech = list()
        for f in tqdm(fnames):
            p = self.predict(f)
            #p = np.mean(p,axis=0)
            preds_cough.append(p[0])
            preds_breath.append(p[1])
            preds_speech.append(p[2])

        df = pd.DataFrame()
        df['file']=files
        df['pred_cough']=preds_cough
        df['pred_breath']=preds_breath
        df['pred_speech']=preds_speech
        df.to_csv(save_file)

    def predict_coswara(self,coswara_dir,save_file='coswara_multiscale.csv'):
        ddir = coswara_dir
        types = dict()
        types['breath'] = os.path.join(ddir,'breathing')
        types['cough'] = os.path.join(ddir,'cough')
        types['speech'] = os.path.join(ddir,'speech')

        files = list()
        preds_cough = list()
        preds_breath =list()
        preds_speech = list()
        ground_truth = list()

        for k in types:
            for f in tqdm(os.listdir(types[k])):
                mel_file = os.path.join(types[k],f)
                p = self.predict(mel_file)
                files.append(mel_file)
                preds_cough.append(p[0])
                preds_breath.append(p[1])
                preds_speech.append(p[2])
                ground_truth.append(k)
        df = pd.DataFrame()
        df['file']=files
        df['pred_cough']=preds_cough
        df['pred_breath']=preds_breath
        df['pred_speech']=preds_speech
        df['ground_truth']=ground_truth
        df.to_csv(save_file)


#ms = Multitimescale(m_small_pth='audio_type_short.h5',m_large_pth='audio_type_long.h5')
