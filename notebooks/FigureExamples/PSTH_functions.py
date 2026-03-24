#%% PSTH functions (for multiple traces)

import numpy as np
import matplotlib.pyplot as plt 

def PSTHmaker(TC, Stims, preW, postW):
    cnt = 0  
    if Stims - preW >= 0 and  Stims + postW < len(TC):
        A = int(Stims-preW) 
        B = int(Stims+postW)  
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

def PSTHmaker_multi(TC, Stims, preW, postW):
    TC = np.asarray(TC)
    if TC.ndim == 1:
        TC = TC[:, None]  # ensure (T, C)
    Stims = np.asarray(Stims, int)

    T, C = TC.shape
    W = preW + postW
    PSTHout = None

    for s in Stims:
        a, b = s - preW, s + postW
        if 0 <= a and b <= T:
            seg = TC[a:b, :]                             # (W, C)
        else:
            # --- NaN padding instead of zeros ---
            seg = np.full((W, C), np.nan, dtype=float)   # (W, C) filled with NaN
            ai, bi = max(0, a), min(T, b)               # overlap with valid range
            if ai < bi:
                seg[(ai - a):(bi - a), :] = TC[ai:bi, :]

        PSTHout = seg if PSTHout is None else np.dstack([PSTHout, seg])

    if PSTHout is None:
        raise ValueError("No events given.")
    return PSTHout  # shape: (W, C, trials)


#%%
def PSTHplot(PSTH, preW, sampling_rate, MainColor, SubColor, LabelStr):
    plt.plot(np.arange(np.shape(PSTH)[1])/20 - preW/sampling_rate, np.mean(PSTH.T,axis=1),label=LabelStr,color = MainColor, linewidth=0.5)
    #plt.plot(np.arange(np.shape(PSTH)[1])/20 - 5, np.mean(PSTH.T,axis=1) + np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0]),color = SubColor, linestyle = "dotted")
    #plt.plot(np.arange(np.shape(PSTH)[1])/20 - 5, np.mean(PSTH.T,axis=1) - np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0]),color = SubColor, linestyle = "dotted")
    # y11 =  np.mean(PSTH.T,axis=1) + np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0])
    # y22 =  np.mean(PSTH.T,axis=1) - np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0])
    # plt.fill_between(np.arange(np.shape(PSTH)[1])/20 - preW/sampling_rate, y11, y22, facecolor=SubColor, alpha=0.5)

def PSTHplot_multi(PSTH_w_by_trials, preW, sampling_rate, color, subcolor, label):
    W = PSTH_w_by_trials.shape[0]
    t_rel = (np.arange(W) - preW) / sampling_rate
    y = np.nanmean(PSTH_w_by_trials, axis=1)  # mean across trials
    sem = np.nanstd(PSTH_w_by_trials, axis=1, ddof=1) / np.sqrt(np.sum(~np.isnan(PSTH_w_by_trials), axis=1))

    plt.plot(t_rel, y, color=color, lw=0.8, label=label)
    plt.fill_between(t_rel, y - sem, y + sem, alpha=0.3, facecolor=subcolor, linewidth=0)

#%% PSTH baseline subtraction (multi)
#dim0:trial, dim1:time
def PSTH_baseline(PSTH, preW):
   
    Trace_this = PSTH[:, :]
    Trace_this_base = Trace_this[0:preW,:]
    Trace_this_subtracted = Trace_this - np.mean(Trace_this_base,axis=0)

    PSTHbase = Trace_this_subtracted

    return PSTHbase

def generate_psth_multi(Ctrl_dF_F, G_df_F, R_df_F, roi_index, stim_frames, preW, postW, fs):
    C = PSTHmaker_multi(Ctrl_dF_F[:, roi_index]*100, stim_frames, preW, postW)
    G = PSTHmaker_multi(G_df_F[:,  roi_index]*100, stim_frames, preW, postW)
    R = PSTHmaker_multi(R_df_F[:,  roi_index]*100, stim_frames, preW, postW)
    # baseline (per trial)
    C = C - np.nanmean(C[:preW, :, :], axis=0, keepdims=True)
    G = G - np.nanmean(G[:preW, :, :], axis=0, keepdims=True)
    R = R - np.nanmean(R[:preW, :, :], axis=0, keepdims=True)
    # time axis tied to THIS windowing
    W = preW + postW
    t_rel = (np.arange(W) - preW) / fs
    return C, G, R, t_rel
