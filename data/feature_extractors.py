import numpy as np
import librosa
import librosa.display


def audio_to_image_extractor(root_dir ,file_name,n_mels,hop_length,n_fft,fmax
,fmin,sampling_rate, gain, bias, eps,power, time_constant,cst=5, top_db=80.):

    '''audio_to_image_extractor
    Convert audio to image
    '''

    row_sound, sr = librosa.load(f"{root_dir}/{file_name}",sr= sampling_rate)
    sound = np.zeros((cst*sr,))

    if row_sound.shape[0] < cst*sr:
        sound[:row_sound.shape[0]] = row_sound[:]
    else:
        sound[:] = row_sound[:cst*sr]

    spec = librosa.feature.melspectrogram(sound, 
                                        sr=sampling_rate,
                                        n_mels=n_mels,
                                        hop_length=hop_length,
                                        n_fft=n_fft,
                                        fmin=fmin,
                                        fmax=fmax)
    spec_pcen = librosa.core.pcen(spec, 
                                    sr= sampling_rate,
                                    hop_length= hop_length,
                                    gain= gain,
                                    bias= bias,
                                    power= power,
                                    time_constant = time_constant,
                                    eps = eps)
    spec_pcen = spec_pcen.astype(np.float32)

    return spec_pcen

def spec_to_image(spec, eps=1e-6):

    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_img = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    
    return spec_img.astype(np.uint8)

def save_spec_image(spec_img, fname):
    cv2.imwrite(fname, spec_img)