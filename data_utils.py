import os
import random
import torch
import torch.utils.data
from torch import nn
from tqdm import tqdm
from tools.log import logger
import commons
import torch.multiprocessing as mp
from mel_processing import spectrogram_torch, mel_spectrogram_torch
from utils import load_audio, load_filepaths_and_text
from text import cleaned_text_to_sequence
from config import config
from models import VQVAE
import torchaudio
from fish_speech.models.vqgan.modules.wavenet import WaveNet
from fish_speech.models.vqgan.spectrogram import LogMelSpectrogram
from fish_speech.models.vqgan.modules.fsq import DownsampleFiniteScalarQuantize
import numpy as np
"""Multi speaker version"""


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.spk_map = hparams.spk2id
        self.hparams = hparams

        self.use_mel_spec_posterior = getattr(
            hparams, "use_mel_posterior_encoder", False
        )
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)


        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        skipped = 0
        logger.info("Init dataset...")
        for audiopath, _, _, _ in tqdm(
            self.audiopaths_sid_text
        ):
            audiopaths_sid_text_new.append([audiopath])
            lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath = audiopath_sid_text[0]
        spec, wav = self.get_audio(audiopath)
        mel_feature = self.get_mel_feature(audiopath)
        return (spec, wav, mel_feature)

    def get_audio(self, filename):
        audio_array = load_audio(filename, self.sampling_rate)  # load_audio的方法是已经归一化到-1~1之间的，不用再/32768
        audio = torch.FloatTensor(audio_array)  # /32768
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(audio_norm, self.filter_length, self.sampling_rate, self.hop_length, self.win_length,
                                  center=False)
        spec = torch.squeeze(spec, 0)
        return spec, audio_norm
        
    def get_mel_feature(self, filename):
        rank = mp.current_process()._identity
        rank = rank[0] if len(rank) > 0 else 0
        gpu_id = rank % torch.cuda.device_count()
        #device = f"cuda:{gpu_id}"
        waveform, _ = torchaudio.load(filename, backend="sox")
        if waveform.shape[0] > 1:  # Check if the audio is not mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        audio = waveform.float().unsqueeze(0)

        # 获取音频长度
        audio_lengths = torch.tensor([audio.shape[1]])
        model = VQVAE(use_decoder=False)
        model.eval()

        decoded_mels = model(audio, audio_lengths)
        if np.random.rand() > 0.8:
            torch.cuda.empty_cache()
        return decoded_mels

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        )
        
        max_spec_len = max([x[0].size(1) for x in batch])
        max_spec_len = int(2 * ((max_spec_len // 2) + 1))
        max_wav_len = max([x[1].size(1) for x in batch])
        max_mel_feature_len = max([x[2].size(2) for x in batch])
        max_mel_feature_len = int(2 * ((max_mel_feature_len // 2) + 1))
        max_spec_len = max(max_spec_len, max_mel_feature_len)
        max_mel_feature_len = max_spec_len
        # text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        mel_feature_lengths = torch.LongTensor(len(batch))
        # sid = torch.LongTensor(len(batch))

        # text_padded = torch.LongTensor(len(batch), max_text_len)
        # tone_padded = torch.LongTensor(len(batch), max_text_len)
        # language_padded = torch.LongTensor(len(batch), max_text_len)
        # bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
        # en_bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
        # emo = torch.FloatTensor(len(batch), 512)

        spec_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        mel_feature_padded = torch.FloatTensor(len(batch), 768, max_mel_feature_len)
        
        # text_padded.zero_()
        # tone_padded.zero_()
        # language_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        mel_feature_padded.zero_()
        # bert_padded.zero_()
        # en_bert_padded.zero_()
        # emo.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            spec = row[0]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[1]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            mel_feature = row[2]
            mel_feature_padded[i, :, : mel_feature.size(2)] = mel_feature
            mel_feature_lengths[i] = mel_feature.size(2)

        return (
            mel_feature_padded,
            mel_feature_lengths,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths
        )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        try:
            for i in range(len(buckets) - 1, 0, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)
            assert all(len(bucket) > 0 for bucket in buckets)
        # When one bucket is not traversed
        except Exception as e:
            print("Bucket warning ", e)
            for i in range(len(buckets) - 1, -1, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            if len_bucket == 0:
                continue
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
