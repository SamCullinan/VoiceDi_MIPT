import torch
import numpy as np
import os
from decomposition import Decomposition
from iemocap_loader import get_loader
from librosa.output import write_wav
import argparse
import sys
import matplotlib.pyplot as plt

# project_dir = "/vol/bitbucket/apg416/project/"


def main(config):
    import os
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # sys.path.insert(1, os.path.join(parent_dir, 'decomposition'))

    # project_dir = config.project_dir
    n_voices = config.n_voices
    n_samples = config.n_samples
    # sample_dir = config.sample_dir
    sample_dir = os.path.join(project_dir, 'samples')
    selected_emos = config.selected_emos
    save_spect = config.save_spect

    assert n_voices % 2 == 0

    ss_iter = 2800000
    ss_path = os.path.join(
        project_dir, "decomposition/run_full/models/{}-G.ckpt".format(ss_iter)
    )
    model = Decomposition(ss_path, freeze=True)
    model.cuda()

    sys.path.insert(1, os.path.join(project_dir, "waveglow"))
    waveglow_iter = 128000
    waveglow_path = os.path.join(
        project_dir, "waveglow/checkpoints/waveglow_{}".format(waveglow_iter)
    )
    waveglow = torch.load(waveglow_path)["model"]
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.cuda().eval()

    sys.path.insert(1, os.path.join(project_dir, "cvoicegan"))
    from model_wgan import Generator

    G = Generator(dim_z=256)
    # G_path = "/vol/bitbucket/apg416/project/cvoicegan/experiments/models/1000000-G.ckpt"
    G_path = os.path.join(
        project_dir, "cvoicegan/experiments/models/{}-G.ckpt".format(1000000)
    )
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    G.cuda()

    dirname_voicegan_wav = os.path.join(sample_dir, "1_VoiceGAN", "wav")
    dirname_neutral_wav = os.path.join(sample_dir, "2_FakeNeutral", "wav")

    if not os.path.exists(dirname_voicegan_wav):
        os.makedirs(dirname_voicegan_wav)
    if not os.path.exists(dirname_neutral_wav):
        os.makedirs(dirname_neutral_wav)

    if save_spect:
        dirname_voicegan_mel = os.path.join(sample_dir, "1_VoiceGAN", "mel")
        dirname_neutral_mel = os.path.join(sample_dir, "2_FakeNeutral", "mel")

        if not os.path.exists(dirname_voicegan_mel):
            os.makedirs(dirname_voicegan_mel)
        if not os.path.exists(dirname_neutral_mel):
            os.makedirs(dirname_neutral_mel)

    for emo in selected_emos:

        iemocap_loader = get_loader(
            selected_emos=[emo], mode="test", batch_size=n_samples
        )

        x_real_org, emb_org, f0_org, len_org, emo_org = next(iter(iemocap_loader))

        x_real_org = x_real_org.cuda()
        emb_org = emb_org.cuda()
        f0_org = f0_org.cuda()

        C, emb_org, codes_x, codes_f0, codes_2 = model(x_real_org, emb_org, f0_org)

        batch_len = x_real_org.size(0)
        z = torch.randn(batch_len * n_voices, 256).cuda()

        emb_fake = G(z, torch.randperm(batch_len * n_voices) % 2)

        x_voice_list = [
            model.decode(C, emb_trg, codes_x, codes_f0, codes_2)
            for emb_trg in emb_fake.chunk(n_voices)
        ]
        x_neutral_list = [model.decode(C, emb_org, codes_x, codes_f0 * 1e-1, codes_2)]

        for idx in range(x_real_org.size(0)):

            # Save fake voice samples
            path_wav = os.path.join(dirname_voicegan_wav, emo + str(idx) + ".wav")

            if save_spect:
                path_mel = os.path.join(dirname_voicegan_mel, emo + str(idx) + ".png")

            with torch.no_grad():
                s = torch.zeros((10, 80)).cuda()
                x_cat = [x_real_org[idx, : len_org[idx], :], s]
                for x_id in x_voice_list:
                    x_cat += [x_id[idx, : len_org[idx], :], s]
                x_cat = torch.cat(x_cat[:-1], dim=0)
                mel = x_cat.unsqueeze(0).transpose(2, 1)
                waveform = waveglow.infer(mel, sigma=1).detach().cpu().numpy()
            write_wav(path_wav, waveform.T, sr=16000)

            if save_spect:
                plt.imsave(
                    path_mel,
                    mel.squeeze(0).detach().cpu().numpy(),
                    origin="lower",
                    cmap="viridis",
                )

            del waveform, x_cat, mel

            # Save fake neutral samples
            path_wav = os.path.join(dirname_neutral_wav, emo + str(idx) + ".wav")

            if save_spect:
                path_mel = os.path.join(dirname_neutral_mel, emo + str(idx) + ".png")

            with torch.no_grad():
                s = torch.zeros((10, 80)).cuda()
                x_cat = [x_real_org[idx, : len_org[idx], :], s]
                for x_id in x_neutral_list:
                    x_cat += [x_id[idx, : len_org[idx], :], s]
                x_cat = torch.cat(x_cat[:-1], dim=0)
                mel = x_cat.unsqueeze(0).transpose(2, 1)
                waveform = waveglow.infer(mel, sigma=1).detach().cpu().numpy()
            write_wav(path_wav, waveform.T, sr=16000)

            if save_spect:
                plt.imsave(
                    path_mel,
                    mel.squeeze(0).detach().cpu().numpy(),
                    origin="lower",
                    cmap="viridis",
                )

            del waveform, x_cat, mel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_voices",
        type=int,
        default=4,
        help="number of generated voices, must be even",
    )
    parser.add_argument(
        "--n_samples", type=int, default=10, help="number of samples per emotion"
    )
    
    parser.add_argument(
        "--selected_emos",
        "--list",
        nargs="+",
        help="selected emotions from IEMOCAP dataset",
        default=["ang", "hap", "sad", "exc", "fru"],
    )
    parser.add_argument(
        "--save_spect", type=bool, default=False, help="perform voice swapping"
    )

    config = parser.parse_args()
    print(config)
    main(config)
