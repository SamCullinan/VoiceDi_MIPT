import os
import sys
import numpy as np
from scipy import signal
from librosa.filters import mel
from pysptk import sptk

sys.path.insert(1, "/vol/bitbucket/apg416/project/decomposition")
from utils import butter_highpass
from utils import speaker_normalization
from utils import pySTFT
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav
import pandas as pd
import re

data_dir = "/vol/bitbucket/apg416/MSc/IEMOCAP"
project_dir = "/vol/bitbucket/apg416/project"

info_line = re.compile(r"\[.+\]\n", re.IGNORECASE)

start_times, end_times, wav_file_names, emotions, vals, acts, doms, paths = (
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
)

for sess in range(1, 6):
    emo_evaluation_dir = os.path.join(
        data_dir, "IEMOCAP_full_release/Session{}/dialog/EmoEvaluation/".format(sess)
    )
    evaluation_files = [
        filename for filename in os.listdir(emo_evaluation_dir) if "Ses" == filename[:3]
    ]
    for file in evaluation_files:
        with open(emo_evaluation_dir + file) as f:
            content = f.read()
        info_lines = re.findall(info_line, content)
        for line in info_lines[1:]:  # the first line is a header
            start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split(
                "\t"
            )
            start_time, end_time = start_end_time[1:-1].split("-")
            val, act, dom = val_act_dom[1:-1].split(",")
            val, act, dom = float(val), float(act), float(dom)
            start_time, end_time = float(start_time), float(end_time)
            start_times.append(start_time)
            end_times.append(end_time)
            wav_file_names.append(wav_file_name)
            paths.append(
                os.path.join(
                    data_dir,
                    "IEMOCAP_full_release/Session{}/sentences/wav/".format(sess),
                    wav_file_name[:-5],
                    wav_file_name + ".wav",
                )
            )
            emotions.append(emotion)
            vals.append(val)
            acts.append(act)
            doms.append(dom)

meta = pd.DataFrame(
    columns=[
        "start_time",
        "end_time",
        "length",
        "wav_file",
        "emotion",
        "val",
        "act",
        "dom",
        "path_wav",
    ]
)

meta["start_time"] = start_times
meta["end_time"] = end_times
meta["length"] = meta["end_time"] - meta["start_time"]
meta["wav_file"] = wav_file_names
meta["emotion"] = emotions
meta["val"] = vals
meta["act"] = acts
meta["dom"] = doms
meta["path_wav"] = paths

lamb_spmel = lambda path_wav: path_wav[:-4].replace("wav", "spmel")
lamb_raptf0 = lambda path_wav: path_wav[:-4].replace("wav", "raptf0")
lamb_emb = lambda path_wav: os.path.join(
    os.path.dirname(os.path.dirname(path_wav).replace("wav", "emb")),
    path_wav[-8] + path_wav[62] + "_avg",
)

meta["spk_id"] = meta.wav_file.apply(lambda s: s[-4] + s[4])
meta["path_spmel"] = meta.path_wav.apply(lamb_spmel)
meta["path_raptf0"] = meta.path_wav.apply(lamb_raptf0)
meta["path_emb"] = meta.path_wav.apply(lamb_emb)

meta.to_csv(
    os.path.join(project_dir, "decomposition/assets/iemocap_meta.csv"), index=False
)
# ------------------------------------------------------------------------------------
encoder = VoiceEncoder()

mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)

speakers = meta.spk_id.unique()
groups = {}
for spk in speakers:
    groups[spk] = meta.groupby(meta.spk_id).get_group(spk)


for spk in speakers:

    print(spk)

    if spk[0] == "M":
        lo, hi = 50, 250
    elif spk[0] == "F":
        lo, hi = 100, 600
    else:
        raise ValueError

    uttr_embs = []
    for (path_wav, path_spmel, path_raptf0, path_emb) in zip(
        groups[spk].path_wav,
        groups[spk].path_spmel,
        groups[spk].path_raptf0,
        groups[spk].path_emb,
    ):

        print(path_wav)

        if not os.path.exists(os.path.dirname(path_spmel)):
            os.makedirs(os.path.dirname(path_spmel))
        if not os.path.exists(os.path.dirname(path_raptf0)):
            os.makedirs(os.path.dirname(path_raptf0))
        if not os.path.exists(os.path.dirname(path_emb)):
            os.makedirs(os.path.dirname(path_emb))

        # read audio file
        x, fs = librosa.load(path_wav, sr=16000)
        assert fs == 16000
        if x.shape[0] % 256 == 0:
            x = np.concatenate((x, np.array([1e-06])), axis=0)
        y = signal.filtfilt(b, a, x)
        wav = y * 0.96 + (np.random.rand(y.shape[0]) - 0.5) * 1e-06

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

        np.save(path_spmel, S.astype(np.float32), allow_pickle=False)
        np.save(path_raptf0, f0_norm.astype(np.float32), allow_pickle=False)

    spkr_emb = np.array(uttr_embs).mean(axis=0)
    np.save(path_emb, spkr_emb.astype(np.float32), allow_pickle=False)
