import os
import sys
import pickle
import numpy as np
from scipy import signal
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav
import pandas as pd


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


meta = pd.read_csv(
    "/vol/bitbucket/apg416/cv-corpus-5.1-2020-06-22/en/validated.tsv", sep="\t"
)
meta = meta[
    ((meta["gender"] == "male") | (meta["gender"] == "female"))
    & (meta["segment"] != "Singleword Benchmark")
]
meta = meta.drop_duplicates(subset=["client_id"])
metaf, metam = meta[meta["gender"] == "female"], meta[meta["gender"] == "male"].sample(
    2807
)
meta = pd.concat([metaf, metam], ignore_index=True)
meta = meta.filter((["path", "gender"]))
meta.gender = meta.gender.replace({"male": 0, "female": 1})

encoder = VoiceEncoder()
b, a = butter_highpass(30, 16000, order=5)

# Modify as needed
rootDir = "/vol/bitbucket/apg416/cv-corpus-5.1-2020-06-22/en/clips"
targetDir_emb = "/vol/bitbucket/apg416/embs_covo"

if not os.path.exists(targetDir_emb):
    os.makedirs(targetDir_emb)

dirName = rootDir
# fileList = meta.path
print("Found directory: %s" % dirName)

drop_idx = []
# for fileName in fileList:
for idx, fileName in enumerate(meta.path):
    print(fileName)

    # read audio file
    try:
        x, fs = librosa.load(os.path.join(dirName, fileName), sr=16000)
    except FileNotFoundError:
        print("FileNotFoundError")
        drop_idx.append(idx)
        continue
    assert fs == 16000
    if x.shape[0] % 256 == 0:
        x = np.concatenate((x, np.array([1e-06])), axis=0)
    y = signal.filtfilt(b, a, x)
    wav = y * 0.96 + (np.random.rand(y.shape[0]) - 0.5) * 1e-06

    # compute speaker embedding
    uttr_emb = encoder.embed_utterance(preprocess_wav(wav, source_sr=16000))

    np.save(
        os.path.join(targetDir_emb, fileName[:-4]),
        uttr_emb.astype(np.float32),
        allow_pickle=False,
    )

meta.drop(drop_idx, inplace=True)
meta.path = meta.path.apply(lambda p: os.path.join(targetDir_emb, p[:-4] + ".npy"))
meta.to_csv(os.path.join(targetDir_emb, "meta_cv.csv"), index=False)
