import numpy as np
import librosa
import librosa.display


def audio_tree_feature_extractor(root_dir, wav_file, hop_length):
  """audio_tree_feature_extractor
  Extracts a dictionary of audio features
  """

  signal, sr = librosa.load(f"{root_dir}{wav_file}")
  signal, _ = librosa.effects.trim(signal)

  chroma = librosa.feature.chroma_stft(signal, sr=sr, hop_length=hop_length)
  mfcc   = librosa.feature.mfcc(signal, sr = sr, n_mfcc=20)
  tempo, _ = librosa.beat.beat_track(signal)

  spectral_centroid = librosa.feature.spectral_centroid(signal, sr=sr, hop_length=hop_length)[0]
  spectral_bandwidth = librosa.feature.spectral_bandwidth(signal, sr=sr, hop_length=hop_length)[0]
  roll_off = librosa.feature.spectral_rolloff(signal, sr=sr,hop_length=hop_length)[0]

  harmony = librosa.effects.harmonic(signal)
  zero_crossing_rate = librosa.feature.zero_crossing_rate(signal)

  rms = librosa.feature.rms(signal)

  features  = {"chroma_stft_mean" : chroma.mean(),
              "chroma_stft_var"  : chroma.var(),
              "tempo" : tempo,
              "spectra_centroid_mean" : spectral_centroid.mean(),
              "spectra_centroid_var" : spectral_centroid.var(),
              "spectra_bandwith_mean" : spectral_bandwidth.mean(),
              "spectra_bandwoth_var" : spectral_bandwidth.var(),
              "roll_off_mean" : roll_off.mean(),
              "roll_off_var" : roll_off.var(),
              "harmony_mean" : harmony.mean(),
              "harmony_var" : harmony.var(),
              "zero_crossing_rate_mean" : zero_crossing_rate.mean(),
              "zero_crossing_rate_var" : zero_crossing_rate.var(),
              "rms_mean" : rms.mean(),
              "rms_var" : rms.var(),
              }


  mfccs_dict = [{f"mfcc{i}_mean": vec.mean() , f"mfcc{i}_var": vec.var()} for i, vec in enumerate(mfcc)]

  for kv in mfccs_dict:
    features.update(kv)

  return features


def audio_to_image_extractor(root_dir ,file_name,n_mels,hop_length,n_fft,fmax
,fmin,sampling_rate, gain, bias, eps,power, time_constant,cst=5, top_db=80.):

    '''audio_to_image_extractor
    Convert audio to image
    '''

    row_sound, sr = librosa.load(f"{root_dir}{file_name}",sr= sampling_rate)
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