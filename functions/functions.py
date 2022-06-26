
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

import numpy as np

SEED_VAL = 42

def get_score(y_true, y_pred):
    """
    logloss score metric
    """
    return log_loss(y_true, y_pred)


def create_folds(df, tgt, n_folds):

    df['fold'] = 0

    df = df.sample(frac=1).reset_index(drop=True)

    fold = StratifiedKFold(n_splits = n_folds, random_state=SEED_VAL,shuffle=True)
    for i, (tr, vr) in enumerate(fold.split(df.index, df[tgt])):
        df.loc[vr, 'fold'] = int(i)

    return df

class EarlyStopping:
    # source : https://www.kaggle.com/code/yusufmuhammedraji/pytorch-cv-earlystopping-lrscheduler

    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        checkpoint = {"config": Config, "model_state_dict": model.state_dict()}

        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


