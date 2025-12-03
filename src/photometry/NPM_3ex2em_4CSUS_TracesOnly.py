# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:34:14 2023

For 4CS-US Probablisic Pavlovian Conditioning

TrialType_
1:CS1 Rewarded
2-10:CS1 UnRewarded
11-15:CS2 Rewarded
16-20CS2 UnRewarded
21-29:CS3 Rewarded
30: CS3 UnRewarded
31-39: CS4 Airpuff
40: CS4 Airpuff Omission 


@author: kenta.hagihara
"""
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import csv
import glob
import re
from scipy.optimize import curve_fit
import json
import pandas as pd
from scipy.stats import sem

import notebooks.BehaviorAnalysis.General.PreprocessingFunctions_v2 as pf

#plt.close('all')
SaveDir = r'C:\Users\jason.lee\Desktop\PdCO\photometry_results'

AnalDir = r'\\allen\aind\scratch\KentaHagihara_InternalTransfer\Pavlovian2upload\behavior_791005_2025-06-25_11-27-33'
FIP_Dir = os.path.join(AnalDir, 'fib')
behav_Dir = os.path.join(AnalDir, 'behavior')

# for visualization
Roi2Vis = [0]    
AllPlot = 0

# params for pre-processing
nFrame2cut = 100  # crop initial n frames
sampling_rate = 20  # individual channel (not total)
kernelSize = 1  # median filter
degree = 4  # polyfit
b_percentile = 0.70  # To calculate F0, median of bottom x%

StimPeriod = 0.5  # sec for visualization
preW = 100  # nframes for PSTH
LickWindow = 5.0  # sec window length for Consummatory/Omission licks

#%%
try: 
    file1 = glob.glob(FIP_Dir + os.sep + "FIP_DataIso_*")[0]
    
except Exception as e:
    file1 = glob.glob(FIP_Dir[0:-8] + 'fib' + os.sep + "FIP_DataIso_*")[0]
    file2 = glob.glob(FIP_Dir[0:-8] + 'fib' + os.sep + "FIP_DataG_*")[0]
    file3 = glob.glob(FIP_Dir[0:-8] + 'fib' + os.sep + "FIP_DataR_*")[0]   
    subjectID = FIP_Dir.split("\\")[3]
    
else:
    file2 = glob.glob(FIP_Dir + os.sep + "FIP_DataG_*")[0]
    file3 = glob.glob(FIP_Dir + os.sep + "FIP_DataR_*")[0]
    subjectID = FIP_Dir.split("\\")[3]

with open(file1) as f:
    reader = csv.reader(f)
    datatemp = np.array([row for row in reader])
    data1 = datatemp[1:,:].astype(np.float32)
    # del datatemp
    
with open(file2) as f:
    reader = csv.reader(f)
    datatemp = np.array([row for row in reader])
    data2 = datatemp[1:,:].astype(np.float32)
    # del datatemp
    
with open(file3) as f:
    reader = csv.reader(f)
    datatemp = np.array([row for row in reader])
    data3 = datatemp[1:,:].astype(np.float32)
    # del datatemp


TSfiles = glob.glob(behav_Dir + os.sep + "TS_*")
TSdict = {}

for file_i in range(len(TSfiles)):
    fullpath_i = glob.glob(behav_Dir + os.sep + "TS_*")[file_i]
    file_i_name = os.path.basename(fullpath_i)
    match = re.search(r'TS_(.*?)_', file_i_name)
    key = match.group(1)
    
    with open(fullpath_i, newline='') as file:
        csv_reader = csv.reader(file)
        try:
            has_header = csv.Sniffer().has_header(file.read(1024))
        except:
            has_header = False
                
    with open(fullpath_i) as f:
        reader = csv.reader(f)
        if has_header:
            next(reader) # skip header
        datatemp = np.array([row for row in reader])
        TSdict[key] = datatemp.astype(np.float32)
        

#%%
# in case acquisition halted accidentally
Length = np.amin([len(data1),len(data2),len(data3)])

data1 = data1[0:Length] # iso       Time*[TS,ROI0,ROI1,ROI2,..]
data2 = data2[0:Length] # signal
data3 = data3[0:Length] # Stim

PMts = data2[:,0] #SignalTS
time_seconds = np.arange(len(data1)) /sampling_rate
#%% Preprocess
Ctrl_dF_F = np.zeros((data1.shape[0],data1.shape[1]))
G_dF_F = np.zeros((data1.shape[0],data1.shape[1]))
R_dF_F = np.zeros((data1.shape[0],data1.shape[1]))

for ii in range(data2.shape[1]-1):
    Ctrl_dF_F[:, ii] = pf.tc_preprocess(data1[:,ii+1], nFrame2cut, kernelSize, sampling_rate, degree, b_percentile)
    G_dF_F[:, ii] = pf.tc_preprocess(data2[:,ii+1] , nFrame2cut, kernelSize, sampling_rate, degree, b_percentile)
    R_dF_F[:, ii] = pf.tc_preprocess(data3[:,ii+1] , nFrame2cut, kernelSize, sampling_rate, degree, b_percentile)

#%% Stim Prep
TSFramesdict = {}

for k in TSdict:
    Temp = TSdict[k]
    TempFrames = np.empty(len(Temp))
    for ii in range(len(Temp)):
        idx = np.argmin(np.abs(data1[:,0] - Temp[ii,0]))
        TempFrames[ii] = idx
    
    TSFramesdict[k] = TempFrames

RewardFrames = TSFramesdict['Reward']
AirpuffFrames = TSFramesdict['Airpuff']
CS1Frames = TSFramesdict['CS1']
CS2Frames = TSFramesdict['CS2']
CS3Frames = TSFramesdict['CS3']
CS4Frames = TSFramesdict['CS4']
LickFrames = TSFramesdict['Lick']

##
figT = plt.figure('Summary:' + AnalDir,figsize=(12, 12))
gs = gridspec.GridSpec(16,12)



#%% PSTH functions (for multiple traces)
def PSTHmaker(TC, Stims, preW, postW):
    cnt = 0
    for ii in range(len(Stims)):
        if Stims[ii] - preW >= 0 and  Stims[ii] + postW < len(TC):
            A = int(Stims[ii]-preW) 
            B = int(Stims[ii]+postW)  
            if cnt == 0:
                PSTHout = TC[A:B,:]
                cnt = 1
            else:
                PSTHout = np.dstack([PSTHout,TC[A:B,:]])
        else:
            if cnt == 0:
                PSTHout = np.zeros(preW+postW)
                cnt = 1
            #else:
                #PSTHout = np.dstack([PSTHout, np.zeros(preW+postW)])
    return PSTHout

#%%
def PSTHplot(PSTH, MainColor, SubColor, LabelStr):
    plt.plot(np.arange(np.shape(PSTH)[1])/20 - preW/sampling_rate, np.mean(PSTH.T,axis=1),label=LabelStr,color = MainColor, linewidth=0.5)
    #plt.plot(np.arange(np.shape(PSTH)[1])/20 - 5, np.mean(PSTH.T,axis=1) + np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0]),color = SubColor, linestyle = "dotted")
    #plt.plot(np.arange(np.shape(PSTH)[1])/20 - 5, np.mean(PSTH.T,axis=1) - np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0]),color = SubColor, linestyle = "dotted")
    y11 =  np.mean(PSTH.T,axis=1) + np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0])
    y22 =  np.mean(PSTH.T,axis=1) - np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0])
    plt.fill_between(np.arange(np.shape(PSTH)[1])/20 - preW/sampling_rate, y11, y22, facecolor=SubColor, alpha=0.5)


#%% PSTH baseline subtraction (multi)
#dim0:trial, dim1:time
def PSTH_baseline(PSTH, preW):

    for ii in range(np.shape(PSTH)[2]):    
        Trace_this = PSTH[:, :, ii]
        Trace_this_base = Trace_this[0:preW,:]
        Trace_this_subtracted = Trace_this - np.mean(Trace_this_base,axis=0)
        if ii == 0:
            PSTHbase = Trace_this_subtracted
        else:
            PSTHbase = np.dstack([PSTHbase,Trace_this_subtracted]) 
    return PSTHbase


#%% Rew+/Rew-
#% Trial csv handing

if bool(glob.glob(behav_Dir + os.sep + "TrialN_*")) == True:
    file_TrialMat = glob.glob(behav_Dir + os.sep + "TrialN_*")[0]
    df = pd.read_csv(file_TrialMat)

    Mat_CS1=np.where((df['TrialType']<=10) & (df['TrialType']>=1))[0]
    Mat_CS2=np.where((df['TrialType']<=20) & (df['TrialType']>=11))[0]
    Mat_CS3=np.where((df['TrialType']<=30) & (df['TrialType']>=21))[0]
    Mat_CS4=np.where(df['TrialType']>30)[0]
    
    Mat_CS1R=np.where(df['TrialType']==1)[0]
    Mat_CS1UR=np.where((df['TrialType']<=10) & (df['TrialType']>=2))[0]
    Mat_CS2R=np.where((df['TrialType']<=15) & (df['TrialType']>=11))[0]
    Mat_CS2UR=np.where((df['TrialType']<=20) & (df['TrialType']>=16))[0]
    Mat_CS3R=np.where((df['TrialType']<=29) & (df['TrialType']>=21))[0]
    Mat_CS3UR=np.where(df['TrialType']==30)[0]
    Mat_CS4P=np.where((df['TrialType']<=39) & (df['TrialType']>=31))[0]
    Mat_CS4UP=np.where(df['TrialType']==40)[0]    
    
    RewardedCS1ind = np.where(np.isin(Mat_CS1, Mat_CS1R))[0]
    RewardedCS2ind = np.where(np.isin(Mat_CS2, Mat_CS2R))[0]
    RewardedCS3ind = np.where(np.isin(Mat_CS3, Mat_CS3R))[0]
    AirpuffCS4ind = np.where(np.isin(Mat_CS4, Mat_CS4P))[0]
    UnRewardedCS1ind = np.where(np.isin(Mat_CS1, Mat_CS1UR))[0]
    UnRewardedCS2ind = np.where(np.isin(Mat_CS2, Mat_CS2UR))[0]
    UnRewardedCS3ind = np.where(np.isin(Mat_CS3, Mat_CS3UR))[0]    
    UnAirpuffCS4ind = np.where(np.isin(Mat_CS4, Mat_CS4UP))[0]

#%%
Psth_G_CS1R = PSTHmaker(G_dF_F*100, CS1Frames[RewardedCS1ind.astype(int)], 100, 300)
Psth_R_CS1R = PSTHmaker(R_dF_F*100, CS1Frames[RewardedCS1ind.astype(int)], 100, 300)
Psth_C_CS1R = PSTHmaker(Ctrl_dF_F*100, CS1Frames[RewardedCS1ind.astype(int)], 100, 300)
Psth_G_CS1R_base = PSTH_baseline(Psth_G_CS1R, 100)
Psth_R_CS1R_base = PSTH_baseline(Psth_R_CS1R, 100)
Psth_C_CS1R_base = PSTH_baseline(Psth_C_CS1R, 100)  

Psth_G_CS1UR = PSTHmaker(G_dF_F*100, CS1Frames[UnRewardedCS1ind.astype(int)], 100, 300)
Psth_R_CS1UR = PSTHmaker(R_dF_F*100, CS1Frames[UnRewardedCS1ind.astype(int)], 100, 300)
Psth_C_CS1UR = PSTHmaker(Ctrl_dF_F*100, CS1Frames[UnRewardedCS1ind.astype(int)], 100, 300)
Psth_G_CS1UR_base = PSTH_baseline(Psth_G_CS1UR, 100)
Psth_R_CS1UR_base = PSTH_baseline(Psth_R_CS1UR, 100)
Psth_C_CS1UR_base = PSTH_baseline(Psth_C_CS1UR, 100)

Psth_G_CS2R = PSTHmaker(G_dF_F*100, CS2Frames[RewardedCS2ind.astype(int)], 100, 300)
Psth_R_CS2R = PSTHmaker(R_dF_F*100, CS2Frames[RewardedCS2ind.astype(int)], 100, 300)
Psth_C_CS2R = PSTHmaker(Ctrl_dF_F*100, CS2Frames[RewardedCS2ind.astype(int)], 100, 300)
Psth_G_CS2R_base = PSTH_baseline(Psth_G_CS2R, 100)
Psth_R_CS2R_base = PSTH_baseline(Psth_R_CS2R, 100)
Psth_C_CS2R_base = PSTH_baseline(Psth_C_CS2R, 100)  

Psth_G_CS2UR = PSTHmaker(G_dF_F*100, CS2Frames[UnRewardedCS2ind.astype(int)], 100, 300)
Psth_R_CS2UR = PSTHmaker(R_dF_F*100, CS2Frames[UnRewardedCS2ind.astype(int)], 100, 300)
Psth_C_CS2UR = PSTHmaker(Ctrl_dF_F*100, CS2Frames[UnRewardedCS2ind.astype(int)], 100, 300)
Psth_G_CS2UR_base = PSTH_baseline(Psth_G_CS2UR, 100)
Psth_R_CS2UR_base = PSTH_baseline(Psth_R_CS2UR, 100)
Psth_C_CS2UR_base = PSTH_baseline(Psth_C_CS2UR, 100)

Psth_G_CS3R = PSTHmaker(G_dF_F*100, CS3Frames[RewardedCS3ind.astype(int)], 100, 300)
Psth_R_CS3R = PSTHmaker(R_dF_F*100, CS3Frames[RewardedCS3ind.astype(int)], 100, 300)
Psth_C_CS3R = PSTHmaker(Ctrl_dF_F*100, CS3Frames[RewardedCS3ind.astype(int)], 100, 300)
Psth_G_CS3R_base = PSTH_baseline(Psth_G_CS3R, 100)
Psth_R_CS3R_base = PSTH_baseline(Psth_R_CS3R, 100)
Psth_C_CS3R_base = PSTH_baseline(Psth_C_CS3R, 100)  

Psth_G_CS3UR = PSTHmaker(G_dF_F*100, CS3Frames[UnRewardedCS3ind.astype(int)], 100, 300)
Psth_R_CS3UR = PSTHmaker(R_dF_F*100, CS3Frames[UnRewardedCS3ind.astype(int)], 100, 300)
Psth_C_CS3UR = PSTHmaker(Ctrl_dF_F*100, CS3Frames[UnRewardedCS3ind.astype(int)], 100, 300)
Psth_G_CS3UR_base = PSTH_baseline(Psth_G_CS3UR, 100)
Psth_R_CS3UR_base = PSTH_baseline(Psth_R_CS3UR, 100)
Psth_C_CS3UR_base = PSTH_baseline(Psth_C_CS3UR, 100)

Psth_G_CS4P = PSTHmaker(G_dF_F*100, CS4Frames[AirpuffCS4ind.astype(int)], 100, 300)
Psth_R_CS4P = PSTHmaker(R_dF_F*100, CS4Frames[AirpuffCS4ind.astype(int)], 100, 300)
Psth_C_CS4P = PSTHmaker(Ctrl_dF_F*100, CS4Frames[AirpuffCS4ind.astype(int)], 100, 300)
Psth_G_CS4P_base = PSTH_baseline(Psth_G_CS4P, 100)
Psth_R_CS4P_base = PSTH_baseline(Psth_R_CS4P, 100)
Psth_C_CS4P_base = PSTH_baseline(Psth_C_CS4P, 100)  

Psth_G_CS4UP = PSTHmaker(G_dF_F*100, CS4Frames[UnAirpuffCS4ind.astype(int)], 100, 300)
Psth_R_CS4UP = PSTHmaker(R_dF_F*100, CS4Frames[UnAirpuffCS4ind.astype(int)], 100, 300)
Psth_C_CS4UP = PSTHmaker(Ctrl_dF_F*100, CS4Frames[UnAirpuffCS4ind.astype(int)], 100, 300)
Psth_G_CS4UP_base = PSTH_baseline(Psth_G_CS4UP, 100)
Psth_R_CS4UP_base = PSTH_baseline(Psth_R_CS4UP, 100)
Psth_C_CS4UP_base = PSTH_baseline(Psth_C_CS4UP, 100)
        
##
ymin = np.empty(len(Roi2Vis))
ymax = np.empty(len(Roi2Vis))
yminR = np.empty(len(Roi2Vis))
ymaxR = np.empty(len(Roi2Vis))

for ii in range(len(Roi2Vis)):
    ymax[ii]=np.max([np.max(np.mean(Psth_G_CS1R_base[:,Roi2Vis[ii],:],axis=1)),
    np.max(np.mean(Psth_G_CS1UR_base[:,Roi2Vis[ii],:],axis=1)),
    np.max(np.mean(Psth_G_CS2R_base[:,Roi2Vis[ii],:],axis=1)),
    np.max(np.mean(Psth_G_CS2UR_base[:,Roi2Vis[ii],:],axis=1)),
    np.max(np.mean(Psth_G_CS3R_base[:,Roi2Vis[ii],:],axis=1)),
    np.max(np.mean(Psth_G_CS3UR_base[:,Roi2Vis[ii],:],axis=1)),
    np.max(np.mean(Psth_G_CS4P_base[:,Roi2Vis[ii],:],axis=1)),
    np.max(np.mean(Psth_G_CS4UP_base[:,Roi2Vis[ii],:],axis=1))])
    
    ymin[ii]=np.min([np.min(np.mean(Psth_G_CS1R_base[:,Roi2Vis[ii],:],axis=1)),
    np.min(np.mean(Psth_G_CS1UR_base[:,Roi2Vis[ii],:],axis=1)),
    np.min(np.mean(Psth_G_CS2R_base[:,Roi2Vis[ii],:],axis=1)),
    np.min(np.mean(Psth_G_CS2UR_base[:,Roi2Vis[ii],:],axis=1)),
    np.min(np.mean(Psth_G_CS3R_base[:,Roi2Vis[ii],:],axis=1)),
    np.min(np.mean(Psth_G_CS3UR_base[:,Roi2Vis[ii],:],axis=1)),
    np.min(np.mean(Psth_G_CS4P_base[:,Roi2Vis[ii],:],axis=1)),
    np.min(np.mean(Psth_G_CS4UP_base[:,Roi2Vis[ii],:],axis=1))])
    
    ymaxR[ii]=np.max([np.max(np.mean(Psth_R_CS1R_base[:,Roi2Vis[ii],:],axis=1)),
    np.max(np.mean(Psth_R_CS1UR_base[:,Roi2Vis[ii],:],axis=1)),
    np.max(np.mean(Psth_R_CS2R_base[:,Roi2Vis[ii],:],axis=1)),
    np.max(np.mean(Psth_R_CS2UR_base[:,Roi2Vis[ii],:],axis=1)),
    np.max(np.mean(Psth_R_CS3R_base[:,Roi2Vis[ii],:],axis=1)),
    np.max(np.mean(Psth_R_CS3UR_base[:,Roi2Vis[ii],:],axis=1)),
    np.max(np.mean(Psth_R_CS4P_base[:,Roi2Vis[ii],:],axis=1)),
    np.max(np.mean(Psth_R_CS4UP_base[:,Roi2Vis[ii],:],axis=1))])
    
    yminR[ii]=np.min([np.min(np.mean(Psth_R_CS1R_base[:,Roi2Vis[ii],:],axis=1)),
    np.min(np.mean(Psth_R_CS1UR_base[:,Roi2Vis[ii],:],axis=1)),
    np.min(np.mean(Psth_R_CS2R_base[:,Roi2Vis[ii],:],axis=1)),
    np.min(np.mean(Psth_R_CS2UR_base[:,Roi2Vis[ii],:],axis=1)),
    np.min(np.mean(Psth_R_CS3R_base[:,Roi2Vis[ii],:],axis=1)),
    np.min(np.mean(Psth_R_CS3UR_base[:,Roi2Vis[ii],:],axis=1)),
    np.min(np.mean(Psth_R_CS4P_base[:,Roi2Vis[ii],:],axis=1)),
    np.min(np.mean(Psth_R_CS4UP_base[:,Roi2Vis[ii],:],axis=1))])
    
figT=plt.figure('Summary:' + AnalDir)

for ii in range(len(Roi2Vis)):
    plt.subplot(2,2,1)
    PSTHplot(Psth_C_CS1R_base[:,Roi2Vis[ii],:].T, [0, 0, 1, 0.2], [0, 0, 0.5, 0.1], "Iso_R+")
    PSTHplot(Psth_R_CS1UR_base[:,Roi2Vis[ii],:].T, "m", "darkmagenta", "R-")
    PSTHplot(Psth_C_CS1UR_base[:,Roi2Vis[ii],:].T, [0, 0, 0, 0.2],[0, 0, 0, 0.1], "Iso_R-")    
    PSTHplot(Psth_R_CS1R_base[:,Roi2Vis[ii],:].T, "g", "darkgreen", "R+")
    plt.ylim([yminR[ii]*1.1, ymaxR[ii]*1.1])
    plt.xlim([-5,15])
    plt.grid(True)
    plt.title("CS1(10%Rew) all trials, Red_ROI: " + str(ii))
    plt.xlabel('Time - Tone (s)')
    plt.ylabel('dF/F%')
    plt.axvspan(0, 1.0, color = [1, 0, 0, 0.4])
    plt.axvspan(2.0, 2.5, color = [0, 0, 1, 0.4])
#plt.legend()

    
for ii in range(len(Roi2Vis)):
    plt.subplot(2,2,2)
    PSTHplot(Psth_C_CS1R_base[:,Roi2Vis[ii],:].T, [0, 0, 1, 0.2], [0, 0, 0.5, 0.1], "Iso_R+")
    PSTHplot(Psth_R_CS2UR_base[:,Roi2Vis[ii],:].T, "m", "darkmagenta", "R-")
    PSTHplot(Psth_C_CS1UR_base[:,Roi2Vis[ii],:].T, [0, 0, 0, 0.2],[0, 0, 0, 0.1], "Iso_R-")  
    PSTHplot(Psth_R_CS2R_base[:,Roi2Vis[ii],:].T, "g", "darkgreen", "R+")
    plt.ylim([yminR[ii]*1.1, ymaxR[ii]*1.1])
    plt.xlim([-5,15])
    plt.grid(True)
    plt.title("CS2(50%Rew) all trials, Red_ROI: " + str(ii))
    plt.xlabel('Time - Tone (s)')
    plt.axvspan(0, 1.0, color = [0, 1, 0, 0.4])
    plt.axvspan(2.0, 2.5, color = [0, 0, 1, 0.4])
    
for ii in range(len(Roi2Vis)):
    plt.subplot(2,2,3)
    PSTHplot(Psth_C_CS1R_base[:,Roi2Vis[ii],:].T, [0, 0, 1, 0.2], [0, 0, 0.5, 0.1], "Iso_R+")
    PSTHplot(Psth_R_CS3UR_base[:,Roi2Vis[ii],:].T, "m", "darkmagenta", "R-")
    PSTHplot(Psth_C_CS1UR_base[:,Roi2Vis[ii],:].T, [0, 0, 0, 0.2],[0, 0, 0, 0.1], "Iso_R-")    
    PSTHplot(Psth_R_CS3R_base[:,Roi2Vis[ii],:].T, "g", "darkgreen", "R+")
    plt.ylim([yminR[ii]*1.1, ymaxR[ii]*1.1])
    plt.grid(True)
    plt.title("CS3(90%Rew) all trials, Red_ROI: " + str(ii))
    plt.xlabel('Time - Tone (s)')
    plt.axvspan(0, 1.0, color = [1, 0, 1, 0.4])
    plt.axvspan(2.0, 2.5, color = [0, 0, 1, 0.4])
    
for ii in range(len(Roi2Vis)):
    plt.subplot(2,2,4)
    PSTHplot(Psth_C_CS1R_base[:,Roi2Vis[ii],:].T, [0, 0, 1, 0.2], [0, 0, 0.5, 0.1], "Iso_Puff+")
    PSTHplot(Psth_R_CS4UP_base[:,Roi2Vis[ii],:].T, "m", "darkmagenta", "Puff-")
    PSTHplot(Psth_C_CS1UR_base[:,Roi2Vis[ii],:].T, [0, 0, 0, 0.2],[0, 0, 0, 0.1], "Iso_Puff-")  
    PSTHplot(Psth_R_CS4P_base[:,Roi2Vis[ii],:].T, "g", "darkgreen", "Puff+")  
    plt.ylim([yminR[ii]*1.1, ymaxR[ii]*1.1])
    plt.grid(True)
    plt.title("CS4(90%Puff) all trials, Red_ROI: " + str(ii))
    plt.xlabel('Time - Tone (s)')
    plt.axvspan(0, 1.0, color = [0.3, 0.3, 0.3, 0.4])
    plt.axvspan(2.0, 2.5, color = [0, 0, 0, 0.4])
plt.legend(loc='upper right', fontsize='small')

plt.subplots_adjust(hspace = 0.5, wspace=0.25)
plt.subplots_adjust(hspace = 2, wspace=1)
plt.tight_layout()

#%%
print('TotalTrial: ' + str(np.sum([len(CS1Frames),len(CS2Frames),len(CS3Frames)])))
print('CS1Trial: ' + str(len(CS1Frames)))
print('CS1 Rewarded:' + str(len(RewardedCS1ind)) + ' (' + str(np.single(100*len(RewardedCS1ind)/len(CS1Frames))) + '%)') 
print('CS2Trial: ' + str(len(CS2Frames)))
print('CS2 Rewarded:' + str(len(RewardedCS2ind)) + ' (' + str(np.single(100*len(RewardedCS2ind)/len(CS2Frames))) + '%)') 
print('CS3Trial: ' + str(len(CS1Frames)))
print('CS3 Rewarded:' + str(len(RewardedCS3ind)) + ' (' + str(np.single(100*len(RewardedCS3ind)/len(CS3Frames))) + '%)') 



#%% Save fig
aID=os.path.basename(AnalDir)
aDate=os.path.basename(os.path.dirname(AnalDir))
plt.savefig(SaveDir + os.sep + 'RedTraces_' + aID + '_' + aDate + '.pdf')
plt.savefig(SaveDir + os.sep + 'RedTraces_' + aID + '_' + aDate + '.jpg')
#if bool(glob.glob(AnalDir + os.sep + "TrialN_*")) == True:

plt.show()

#%% JL 250701 
# === Standalone Individual Trial Plots: First 5 Trials of CS1–CS4 ===
cs_labels = ['CS1(10%Rew)', 'CS2(50%Rew)', 'CS3(90%Rew)', 'CS4(90%Puff)']
cs_frames = [CS1Frames, CS2Frames, CS3Frames, CS4Frames]
cue_colors = [[1, 0, 0, 0.2], [0, 1, 0, 0.2], [1, 0, 1, 0.2], [0.3, 0.3, 0.3, 0.2]]  # cue span color
event_colors = ['blue', 'blue', 'blue', 'black']  # reward/puff span color

plt.figure(figsize=(12, 10))

for idx, (cs_label, frames, cue_color, event_color) in enumerate(zip(cs_labels, cs_frames, cue_colors, event_colors)):
    # Get baseline-subtracted PSTH for first 5 trials
    psth_trials = PSTHmaker(R_dF_F*100, frames[:5], preW, 300)
    psth_trials = PSTH_baseline(psth_trials, preW)

    ax = plt.subplot(2, 2, idx + 1)
    for trial_i in range(min(5, psth_trials.shape[2])):
        ax.plot(
            np.arange(psth_trials.shape[0]) / sampling_rate - preW / sampling_rate,
            psth_trials[:, Roi2Vis[0], trial_i],
            label=f'Trial {trial_i+1}',
            alpha=0.6
        )

    ax.set_title(f'{cs_label} - First 5 Trials (ROI {Roi2Vis[0]})')
    ax.set_xlabel('Time - Tone (s)')
    ax.set_ylabel('dF/F (%)')
    ax.axvspan(0, 1.0, color=cue_color)
    ax.axvspan(2.0, 2.5, color=event_color, alpha=0.2)
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()


#%% 250701 JL
# === First vs. Last 5 Trials of CS4 (Airpuff) ===

first5 = CS4Frames[:5]
last5 = CS4Frames[-5:]

# Green signal
psth_first5 = PSTHmaker(R_dF_F*100, first5, preW, 300)
psth_last5  = PSTHmaker(R_dF_F*100, last5,  preW, 300)

# Baseline-subtract
psth_first5 = PSTH_baseline(psth_first5, preW)
psth_last5  = PSTH_baseline(psth_last5, preW)

# Plotting
plt.figure(figsize=(8, 6))
time_axis = np.arange(psth_first5.shape[0])/sampling_rate - preW/sampling_rate

for i in range(min(5, psth_first5.shape[2])):
    plt.plot(time_axis, psth_first5[:, Roi2Vis[0], i], color='red', alpha=0.5, label='First 5' if i==0 else None)

for i in range(min(5, psth_last5.shape[2])):
    plt.plot(time_axis, psth_last5[:, Roi2Vis[0], i], color='blue', alpha=0.5, label='Last 5' if i==0 else None)

plt.title(f'CS4(90%Puff) – ROI {Roi2Vis[0]}: First vs. Last 5 Trials')
plt.xlabel('Time - Tone (s)')
plt.ylabel('dF/F (%)')
plt.axvspan(0, 1.0, color=[0.3, 0.3, 0.3, 0.3], label='CS4 Cue')
plt.axvspan(2.0, 2.5, color='black', alpha=0.2, label='Airpuff')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


