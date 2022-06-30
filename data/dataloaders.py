import torch
from torch.utils.data import Dataset
import cv2
import joblib

import numpy as np




class AudioDataset(Dataset):
  def __init__(self, df, task='train', size=(300, 230), **kwargs):
    super(AudioDataset, self).__init__()
    self.df = df
    self.task = task
    self.size = size
    mapper  = joblib.load(f"./data/label_mapper.joblib")
    self.c = len(mapper)
    self.classes = list(mapper.values())

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):

    fn = self.df.loc[idx, 'spec_name']

    spec = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    spec = cv2.resize(spec, self.size)

    output = {
        'spec': torch.tensor(spec, dtype=torch.float).unsqueeze(0),
    }

    if self.task=='train':
      output.update({'label': torch.tensor(np.argmax(self.df.iloc[idx,3:-1].values)) })

    return output
