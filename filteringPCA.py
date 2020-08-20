import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from sklearn.decomposition import PCA
from scipy import stats
def interpolate(traj, origin, processed):
    traj_resized = []
    for i in range(traj.shape[1]):
        # subtle nuance in dsize - (1, newSize)
        resize = cv2.resize(traj[:,i], (1,int(traj.shape[0]*processed / origin)),interpolation=cv2.INTER_CUBIC)
        traj_resized.append(resize)
    traj_resized = np.squeeze(np.array(traj_resized))
    traj_resized = np.swapaxes(traj_resized,0,1)
    return traj_resized

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    output = []
    for i in range(data.shape[1]):
        y = lfilter(b, a, data[:,i])
        output.append(y)
    output = np.array(output).reshape(data.shape)
    return output

## Define parameters (take from output of MotionTracking.py)- 
fps_cam = 25
fs = 91

#Processing - mode filtering
traj = np.squeeze(np.load('extracted_files/traj_p13.npy'))
traj = traj[:,:,0]
traj_diff = np.diff(traj, axis=0)
traj_diff_max = np.max(abs(traj_diff), axis=0).astype(int)
traj_mode = stats.mode(traj_diff_max)[0][0]
good_features = np.where(traj_diff_max <= traj_mode)[0]

traj = traj[:,good_features]

# traj = traj - traj[0,:] 
for i in range(traj.shape[1]):
    plt.plot(traj[0:100,i])
plt.show()

# print(traj.shape)
# Cubic interpolation (fps of the video in file is 28)
traj_resampled = interpolate(traj, fps_cam, fs)
# print(traj_resampled.shape)
# Pass band filtering using 5th order butterworth filter
lowcut = 0.75
highcut = 2
traj_filtered = np.fft.rfft(traj_resampled, axis=0)
# print(traj_filtered.shape)
# traj_filtered = cv2.dft(traj_resampled, axis=0)
low_lim = np.floor(2*traj_filtered.shape[0]*lowcut/fs).astype(int)
up_lim  = np.ceil(2*traj_filtered.shape[0]*highcut/fs).astype(int)
# print(low_lim)
# print(up_lim)
traj_filtered[0:low_lim,:] = 0 
traj_filtered[up_lim:,:] = 0
traj_filtered = np.fft.irfft(traj_filtered, axis=0)
# traj_filtered = cv2.idft(traj_filtered,axis=0)
# print(traj_filtered.shape)
#butter_bandpass_filter(traj_resampled, lowcut=0.8, highcut=3, fs=fs)
plt.plot(traj_filtered)
plt.show()
# initialize PCA object
pca = PCA(traj_filtered.shape[1])

# remove the top 25% trajectories with highest L2 norm
alpha = 0.25
norm_mat = np.linalg.norm(traj_filtered,axis=1)
# print(norm_mat.shape)
idx = (-norm_mat).argsort()[int(alpha*traj_filtered.shape[0]):]
traj_pca_inp = traj_filtered[idx,:]
# print(traj_pca_inp.shape)
pca.fit(traj_pca_inp)
# print(pca.components_.shape)

# Generate PCA projections 
signal = pca.transform(traj_filtered)

# signal = np.matmul(traj_filtered, pca.components_.T)
np.save('signal.npy', signal)

# Compute which principal vector is most periodic
periodicity = []
freq_arr = []
for k in range(3):
    id = k
    # sig = signal[:,id] - np.mean(signal[:,id])
    sig = signal[:,id]
    sig_fft = abs(cv2.dft(sig))
    # print(sig_fft)
    max_val = np.max(sig_fft)
    # print(max_val)
    id_max = np.argmax(sig_fft)
    # print(id_max)
    freq = fs * (id_max) / (2*sig_fft.shape[0])
    freq_arr.append(freq)
    harmonic = 2*freq
    id_harmonic = np.floor(sig_fft.shape[0]*harmonic/fs).astype(int)
    # print(id_harmonic)
    # compute periodicity as mentioned in paper 
    periodicity.append((sig_fft[id_max] + sig_fft[id_harmonic] )/sum(sig_fft))
    
# print(sig_fft.shape)
sig_periodic = np.argmax(np.array(periodicity))
# print(periodicity)
heart_beat = freq_arr[sig_periodic]
print("most periodic signal id: ", sig_periodic)
print("heart beat: ", 60*heart_beat, "bpm")

plt.plot(signal[:,sig_periodic])
plt.show()




# id = 0
# val = -100
# for id in range(5):
#     sig = signal[:,id] - np.mean(signal[:,id])
#     sig_fft = abs(cv2.dft(sig))**2
#     id_freq = np.argmax(sig_fft)
#     print(sig_fft[id_freq] / sum(sig_fft))
#     if(sig_fft[id_freq] / sum(sig_fft) > val):
#         val = sig_fft[id_freq] / sum(sig_fft)
#         id_max = id_freq
# # the factor 8 is the approximate stop band of the butterworth filter maybe, not sure
# print(sig_fft.shape[0])
# heart_beat= id_max*fps_cam/sig_fft.shape[0]
# print(sig_fft.shape[0])
# print("heart beat: ",60*heart_beat, "bpm")
