import os
import pickle
import numpy as np

rootDir = "assets/vctk_full/vctk_full_spmel"
targetDir_emb = "assets/vctk_full/vctk_full_emb"
dirName, subdirList, _ = next(os.walk(rootDir))
print("Found directory: %s" % dirName)


speakers = []
for speaker in sorted(subdirList):
    print("Processing speaker: %s" % speaker)
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName, speaker)))

    spkid = np.load(os.path.join(targetDir_emb, speaker, speaker + "_avg.npy"))
    utterances.append(spkid)

    # create file list
    for fileName in sorted(fileList):
        utterances.append(os.path.join(speaker, fileName))
    speakers.append(utterances)

with open(os.path.join(rootDir, "train.pkl"), "wb") as handle:
    pickle.dump(speakers, handle)
