import logging
import torch
import numpy as np
import parselmouth
from parselmouth.praat import call


def get_formant(wav_path, f0, formant_num=1):
    sound = parselmouth.Sound(wav_path)
    formant = call(sound, "To Formant (burg)...", 0.0, 5.0, 5500.0, 0.025, 50.0)
    frames = call(formant, "List all frame times")
    num_formants = []
    for time in frames:
        tmp = call(formant, "Get value at time", formant_num, time, "hertz", "Linear")
        num_formants.append(tmp)

    # force the frame numbers of num_formants equal mel
    if len(f0) > len(num_formants):
        padded_len  = len(f0) - len(num_formants)
        num_formants = np.pad(num_formants, (padded_len, 0), 'constant', constant_values=(0, 0))
    else:
        num_formants = num_formants[:len(f0)]
    return num_formants


def log(f0):
    '''
    '''
    nonzeros_indices = np.nonzero(f0)
    lf0 = f0.copy()
    lf0[nonzeros_indices] = np.log(f0[nonzeros_indices]) # for f0(Hz), lf0 > 0 when f0 != 0
    return lf0


def norm_f0(f0):
    nonzeros_indices = np.nonzero(f0)
    lf0 = f0.copy()
    lf0[nonzeros_indices] = np.log(f0[nonzeros_indices]) # for f0(Hz), lf0 > 0 when f0 != 0
    mean, std = np.mean(lf0[nonzeros_indices]), np.std(lf0[nonzeros_indices])
    lf0[nonzeros_indices] = (lf0[nonzeros_indices] - mean) / (std + 1e-8)
    return lf0



def get_logger(filename, verbosity=1, name=None):
    '''
    '''
    level_dict = {0: logging.DEBUG,
                  1: logging.INFO,
                  2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger()
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, 'w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def try_gup(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f"cuda:{i}")
    return torch.device('cpu')


def try_all_gpu():
    devices = [
        torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
    ]
    return devices if devices else [torch.device['cpu']]


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def to_eval(all_models):
    for m in all_models:
        m.eval()


def to_train(all_models):
    for m in all_models:
        m.train()
