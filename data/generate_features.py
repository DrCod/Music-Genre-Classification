import numpy as np
import pandas as pd
import librosa
import librosa.display
import IPython.display as ipd
from tqdm.auto import tqdm


from feature_extractors import audio_to_image_extractor

class Config:
    root_dir = "./raw"
    sampling_rate = 22050
    duration = 30 # sec
    hop_length = 128*duration # to make time steps 128
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 92
    n_fft = n_mels * 20
    padmode = 'constant'
    samples = sampling_rate * duration
    ####################### PCEN PARAMS
    gain          = 0.6
    bias          = 0.1 
    power         = 0.2 
    time_constant = 0.4 
    eps           = 1e-9
    skip_files = ["jazz.00054.wav"]



def generate_specs(df):

    #Training specs
    for row in tqdm(df.values):
    sound_path = row[0] #this corresponds to 'file_name'
    if sound_path not in CFG.skip_files:
        spec_name = sound_path.replace(".wav", ".jpg") #this corresponds to 'spec_name'

        spec = audio_to_image_extractor(root_dir=CFG.root_dir ,file_name=sound_path,n_mels=CFG.n_mels,hop_length=CFG.hop_length,n_fft=CFG.n_fft,fmax=CFG.fmax
,fmin=CFG.fmin,sampling_rate=CFG.sampling_rate, gain=CFG.gain, bias=CFG.bias, eps=CFG.eps,power=CFG.power, time_constant=CFG.time_constant,cst=30, top_db=80.)

        spec = spec_to_image(spec)
        save_spec_image(spec, spec_name)
    else:
        continue


def main(args):
    print(f"Generating training features ...")
    df = pd.read_csv(args.filename)
    generate_specs(df)
    print("Done!")

        

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Feature Generator")
    parser.add_argument("--filename", type = str, default=f"{CFG.root}/train.csv.csv", help="path/to/csv file")

    args = parser.parse_args()


    main(args)