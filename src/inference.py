import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

from data.feature_extractors import *

from data.dataloaders import AudioDataset
from src.model import *
import cv2
from scipy.stats.mstats import gmean


device = 'cuda' if torch.cuda.is_available() else 'cpu'

mapper = joblib.load("./data/label_mapper.joblib")


def inference(model, states, test_loader, device):

  model.to(device)

  probabilities = []
  for i, data in enumerate(test_loader):
    images = data["spec"].to(device)
    avg_preds = []
    for state in states:
      model.load_state_dict(state['model'])
      model.eval()
      with torch.no_grad():
        y_preds = model(images)
      avg_preds.append(y_preds.softmax(1).to('cpu').numpy())

    avg_preds = gmean(avg_preds, axis=0)

    probabilities.append(avg_preds)
  return np.concatenate(probabilities)

  def audio_to_image(file_name,n_mels,hop_length,n_fft,fmax,fmin,sampling_rate, gain, bias, eps,power, time_constant,cst=5, top_db=80.):

    '''audio_to_image
    Convert audio to image
    '''

    row_sound, sr = librosa.load(f"{file_name}",sr= sampling_rate)
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


  def main(args):

    pretrained_models = {
    f"{args.model_name}" : [f'{args.trained_models}/{args.model_name}_fold{fold}_best.pth' for fold in args.use_folds] }

    states = [torch.load(f) for f in pretrained_models[f"{args.model_name}"]]

    if args.batch_predict is not None:

        test_df = pd.read_csv(args.test_csv)
        test_ds = AudioDataset(df = test_df,  size= (*args.size))
        testloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

        model = AudioModel(arch_name = args.model_name,pretrained=args.pretrained, Family =args.family)

        predictions = inference(model, states, testloader, device)

        predictions_df = pd.DataFrame()
        predictions_df["Image_id"] = test_df["filename"].values
        predictions_df["ground_truth"] = test_df["label"].values
        predictions_df['image_path_id'] = args.root_dir + predictions_df['Image_id'].astype(str)

        predictions_df["genre"] = predictions_df["label"].map(mapper)
        predictions_df["genre"] = predictions_df["genre"].apply(lambda x: x.replace("genre_", ""))

        acc = sum(predictions_df['ground_truth'] == predictions_df['genre'])/predictions_df.shape[0])*100

        print(f"Model Accuracy : {(acc} % ")

        torch.cuda.empty_cache()
        try:
            del(model)
            del(states)
        except:
            pass
        gc.collect()

        return predictions_df, acc


    else:

        fn = args.audio_path

        spec = audio_to_image(file_name=fn,\
        n_mels=CFG.n_mels,hop_length=CFG.hop_length,n_fft=CFG.n_fft,fmax=CFG.fmax,\
        fmin=CFG.fmin,sampling_rate=CFG.sampling_rate, gain=CFG.gain, bias=CFG.bias,\
         eps=CFG.eps,power=CFG.power, time_constant=CFG.time_constant,cst=30, top_db=80.)

        spec = spec_to_image(spec)

        spec = cv2.resize(spec, self.size)

        spec = torch.tensor(spec, dtype=torch.float).unsqueeze(0)

        model = AudioModel(arch_name = args.model_name,pretrained=args.pretrained, Family =args.family).to(device)

        avg_preds = []
        for state in states:
            model.load_state_dict(state['model'])
            model.eval()
            with torch.no_grad():
                y_preds = model(spec)
            avg_preds.append(y_preds.softmax(1).to('cpu').numpy())

        avg_preds = gmean(avg_preds, axis=0)

        avg_preds = np.squeeze(avg_preds)

        pred_label = np.argmax(avg_preds, 1)

        predicted_genre = mapper[pred_label]

        torch.cuda.empty_cache()
        try:
            del(model)
            del(states)
        except:
            pass
        gc.collect()

        return predicted_genre

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inference Runner")

    parser.add_argument("--test_csv", type = str,  default = './data/raw/test.csv', help = "path/to/test csv")
    parser.add_argument("--root_dir" , type = str,    default = './data/spectrograms/', help = "path/to/spectrograms")
    parser.add_argument("--debug", action = "store_true", default=None, help = "run in debug mode")
    parser.add_argument("--sampling_rate", type =int, default=22050, help="sampling rate")
    parser.add_argument("--num_workers", type =int, default=2, help="number of workers")
    parser.add_argument("--model_name", type =str, default="densenet201", help="model name")
    parser.add_argument("--family", type =str, default="Densenet201", choices =["Densenet201", "Densenet161", "tf_efficientnet_b4_ns"], help="model family")
    parser.add_argument("--size", type =tuple, default=(500, 230), help="image size")
    parser.add_argument("--batch_size", type = int, default=32, help = "batch size")
    parser.add_argument("--duration", type =int, default=30, help="duration")
    parser.add_argument("--hop_length", type =int, default=0, help="hop length")
    parser.add_argument("--fmin", type =int, default=20, help="min frequency")
    parser.add_argument("--fmax", type =int, default=0, help="max frequency")
    parser.add_argument("--eps", type =float, default=1e-9, help="epsilon")
    parser.add_argument("--n_mels", type =int, default=92,help="n_mels")
    parser.add_argument("--n_fft", type =int, default=0, help="nfft")
    parser.add_argument("--padmode", type =str, default="constant", help="padding mode")
    parser.add_argument("--samples", type =int, default=0, help="samples")
    parser.add_argument("--gain", type =float, default=0.6, help="gain")
    parser.add_argument("--bias", type =float, default=0.1, help="bias")
    parser.add_argument("--power", type =float, default=0.2, help="power")
    parser.add_argument("--time_constant", type =float, default=0.4, help="time_constant")
    parser.add_argument("--batch_predict", action="store_true", default=None, help = "flag to predict a batch of files in csv format")

    args = parser.parse_args()

    args.hop_length  = args.duration * 128 if not args.hop_length else args.hop_length
    args.samples = args.sampling_rate * args.duration if not args.samples else args.samples
    args.fmax = args.sampling_rate // 2 if not args.fmax else args.samples

    main(args)








