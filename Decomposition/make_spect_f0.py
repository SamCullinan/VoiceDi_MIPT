import os
import sys
import pickle
import numpy as np

from scipy import signal
from librosa.filters import mel
from numpy.random import RandomState
from pysptk import sptk
from utils import butter_highpass
from utils import speaker_normalization
from utils import pySTFT

import librosa

from resemblyzer import VoiceEncoder, preprocess_wav

encoder = VoiceEncoder()

mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)

spk2gen = pickle.load(open("assets/spk2gen.pkl", "rb"))


# Modify as needed

rootDir = "assets/vctk_full"
targetDir_f0 = "assets/vctk_full_raptf0"
targetDir = "assets/vctk_full_spmel"

dirName, subdirList, _ = next(os.walk(rootDir))
print("Found directory: %s" % dirName)

for subdir in sorted(subdirList):
    print(subdir)

    if not os.path.exists(os.path.join(targetDir, subdir)):
        os.makedirs(os.path.join(targetDir, subdir))
    if not os.path.exists(os.path.join(targetDir_f0, subdir)):
        os.makedirs(os.path.join(targetDir_f0, subdir))
    if not os.path.exists(os.path.join(targetDir_emb, subdir)):
        os.makedirs(os.path.join(targetDir_emb, subdir))
    _, _, fileList = next(os.walk(os.path.join(dirName, subdir)))

    if spk2gen[subdir] == "M":
        lo, hi = 50, 250
    elif spk2gen[subdir] == "F":
        lo, hi = 100, 600
    else:
        raise ValueError

    prng = RandomState(int(subdir[1:]))
    uttr_embs = []
    for fileName in sorted(fileList):
        # read audio file
        ## x, fs = sf.read(os.path.join(dirName,subdir,fileName))
        if fileName[-4:] != ".wav":  # del
            continue
        x, fs = librosa.load(os.path.join(dirName, subdir, fileName), sr=16000)
        assert fs == 16000
        if x.shape[0] % 256 == 0:
            x = np.concatenate((x, np.array([1e-06])), axis=0)
        y = signal.filtfilt(b, a, x)
        wav = y * 0.96 + (prng.rand(y.shape[0]) - 0.5) * 1e-06

        # compute speaker embedding
        uttr_emb = encoder.embed_utterance(preprocess_wav(wav, source_sr=16000))
        uttr_embs.append(uttr_emb)

        # compute spectrogram
        D = pySTFT(wav).T
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = (D_db + 100) / 100

        # extract f0
        f0_rapt = sptk.rapt(
            wav.astype(np.float32) * 32768, fs, 256, min=lo, max=hi, otype=2
        )
        index_nonzero = f0_rapt != -1e10
        mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(
            f0_rapt[index_nonzero]
        )
        f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)

        assert len(S) == len(f0_rapt)

        np.save(
            os.path.join(targetDir, subdir, fileName[:-4]),
            S.astype(np.float32),
            allow_pickle=False,
        )
        np.save(
            os.path.join(targetDir_f0, subdir, fileName[:-4]),
            f0_norm.astype(np.float32),
            allow_pickle=False,
        )
        np.save(
            os.path.join(targetDir_emb, subdir, fileName[:-4]),
            uttr_emb.astype(np.float32),
            allow_pickle=False,
        )

    spkr_emb = np.array(uttr_embs).mean(axis=0)
    np.save(
        os.path.join(targetDir_emb, subdir, subdir + "_avg"),
        spkr_emb.astype(np.float32),
        allow_pickle=False,
    )
