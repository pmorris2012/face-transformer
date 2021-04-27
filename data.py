import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from sklearn.model_selection import train_test_split
import sys
import random

SPECIAL_TOKENS = {
    "MASK": -1,
    "SEQ": -2
}

def collate_missing(batch):
    arrays = [item[0] for item in batch]
    seq_idxs = [item[1] for item in batch]
    seq_label_idxs = [item[2] for item in batch]
    label_arrays = [item[3] for item in batch]

    array_count = 0
    for i in range(len(batch)):
        seq_idxs[i][seq_idxs[i] >= 0] += array_count
        array_count += arrays[i].shape[0]

    return torch.cat(arrays), torch.stack(seq_idxs), torch.stack(seq_label_idxs), torch.cat(label_arrays)

class PretrainingDataset(Dataset):
    """Face Keypoints dataset."""

    def __init__(
        self, 
        root_dir,
        array_ext='.npy',
        framerate=30.0,
        sample_length=4.0, #seconds
        step_size=0.1, #seconds
        num_keypoints=68,
        special_tokens=SPECIAL_TOKENS,
        label_prob=0.15,
        label_keep_prob=0.10
    ):
        self.root_dir = Path(root_dir)
        self.array_ext = array_ext
        self.framerate = framerate
        self.sample_length = sample_length
        self.step_size = step_size
        self.num_keypoints = num_keypoints
        self.num_distances = count = torch.arange(self.num_keypoints).sum().item()
        self.special_tokens = special_tokens
        self.label_prob = label_prob
        self.label_keep_prob = label_keep_prob

        self._build_directory_index()

    def _build_directory_index(self):
        #list all files (frames) in each folder (video)
        self.directory_index = {}
        file_pattern = F"**/*{self.array_ext}"
        for file_path in self.root_dir.glob(file_pattern):
            folder_path = file_path.parent
            frame = file_path.stem
            if folder_path in self.directory_index:
                self.directory_index[folder_path].append(frame)
            else:
                self.directory_index[folder_path] = [frame]

        self.samples = []
        for path in self.directory_index:
            frames = self.directory_index[path]
            for stride_array in self._stride_video(frames):
                self.samples.append((path, stride_array))

    def _stride_video(self, frames):
        frames = sorted(frames, key=lambda frame: int(frame))
        int_frames = list(map(int, frames))
        min_frame = int(frames[0])
        max_frame = int(frames[-1])
        num_timesteps = int(round(self.sample_length / self.step_size))
        frames_per_sample = int(self.sample_length * self.framerate)
        
        all_stride_arrays = []
        for _start_frame in range(min_frame, max_frame, frames_per_sample):
            start_frame = _start_frame
            if (start_frame + frames_per_sample) > max_frame:
                start_frame = max(min_frame, max_frame - frames_per_sample)
            end_frame = start_frame + frames_per_sample

            times = torch.linspace(start_frame, end_frame, num_timesteps)
            times = times.round().int()

            stride_arrays = [None] * num_timesteps
            iter_idx = 0
            found_count = 0
            for i, time in enumerate(times):
                if iter_idx >= len(int_frames):
                    break
                while int_frames[iter_idx] < time:
                    iter_idx += 1
                    if iter_idx >= len(int_frames):
                        break
                if iter_idx >= len(int_frames):
                        break
                if int_frames[iter_idx] == time:
                    found_count += 1
                    stride_arrays[i] = frames[iter_idx]

            if found_count > 1:
                all_stride_arrays.append(stride_arrays)

        return all_stride_arrays


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, stride_array = self.samples[idx]
        arrays = []
        seq_idxs = [self.special_tokens['SEQ']]
        array_counter = 0

        for i, file_name in enumerate(stride_array):
            if file_name is None:
                seq_idxs.append(self.special_tokens['MASK'])
            else:
                arrays.append(self._load_array(path, file_name))
                seq_idxs.append(array_counter)
                array_counter += 1

        seq_label_idxs = [False] * len(seq_idxs)
        label_idxs = torch.tensor(seq_idxs.copy())


        seq_matches = torch.where(torch.tensor(seq_idxs) >= 0)[0]
        rand_labels = torch.rand(len(seq_matches)) < self.label_prob
        if rand_labels.sum() == 0:
            rand_labels[torch.randint(len(seq_matches), (1,))] = True
        for rand_idx in seq_matches[rand_labels]:
            seq_label_idxs[rand_idx] = True

            if torch.rand(1).item() >= self.label_keep_prob:
                seq_idxs[rand_idx] = self.special_tokens['MASK']

        arrays = torch.stack(arrays)
        seq_label_idxs = torch.tensor(seq_label_idxs)
        label_arrays = arrays[label_idxs[seq_label_idxs]]

        return arrays, torch.tensor(seq_idxs), seq_label_idxs, label_arrays

    def _load_array(self, path, frame_stem):
        path = Path(path, frame_stem).with_suffix(self.array_ext)
        keypoints = np.load(path)
        keypoints_norm = zscore(keypoints)
        keypoints_dist = pdist(keypoints_norm)
        return torch.tensor(keypoints_dist, dtype=torch.float)


def DISFA_get_videos_tvt(root_dir, val=0.1, test=0.1):
    videos = []
    root_dir = Path(root_dir)
    for group in root_dir.iterdir():
        for video in group.iterdir():
            videos.append(video.relative_to(root_dir))
    train, val = train_test_split(videos, test_size=val)
    train, test = train_test_split(train, test_size=test)
    return train, val, test
    


class DISFADataset(Dataset):
    """Face Keypoints dataset."""

    def __init__(
        self, 
        root_dir,
        labels_dir,
        array_ext='.npy',
        framerate=30.0,
        sample_length=4.0, #seconds
        step_size=0.1, #seconds
        num_keypoints=68,
        special_tokens=SPECIAL_TOKENS
    ):
        self.root_dir = Path(root_dir)
        self.labels_dir = Path(labels_dir)
        self.array_ext = array_ext
        self.framerate = framerate
        self.sample_length = sample_length
        self.step_size = step_size
        self.num_keypoints = num_keypoints
        self.num_distances = count = torch.arange(self.num_keypoints).sum().item()
        self.special_tokens = special_tokens

        self._build_directory_index()

    def _build_directory_index(self):
        #list all files (frames) in each folder (video)
        self.directory_index = {}
        file_pattern = F"**/*{self.array_ext}"
        for file_path in self.root_dir.glob(file_pattern):
            folder_path = file_path.parent
            frame = file_path.stem
            if folder_path in self.directory_index:
                self.directory_index[folder_path].append(frame)
            else:
                self.directory_index[folder_path] = [frame]

        self.samples = []
        for path in self.directory_index:
            frames = self.directory_index[path]
            for stride_array in self._stride_video(frames):
                self.samples.append((path, stride_array))

    def _stride_video(self, frames):
        frames = sorted(frames, key=lambda frame: int(frame))
        int_frames = list(map(int, frames))
        min_frame = int(frames[0])
        max_frame = int(frames[-1])
        num_timesteps = int(round(self.sample_length / self.step_size))
        frames_per_sample = int(self.sample_length * self.framerate)
        
        all_stride_arrays = []
        for _start_frame in range(min_frame, max_frame, frames_per_sample):
            start_frame = _start_frame
            if (start_frame + frames_per_sample) > max_frame:
                start_frame = max(min_frame, max_frame - frames_per_sample)
            end_frame = start_frame + frames_per_sample

            times = torch.linspace(start_frame, end_frame, num_timesteps)
            times = times.round().int()

            stride_arrays = [None] * num_timesteps
            iter_idx = 0
            found_count = 0
            for i, time in enumerate(times):
                if iter_idx >= len(int_frames):
                    break
                while int_frames[iter_idx] < time:
                    iter_idx += 1
                    if iter_idx >= len(int_frames):
                        break
                if iter_idx >= len(int_frames):
                        break
                if int_frames[iter_idx] == time:
                    found_count += 1
                    stride_arrays[i] = frames[iter_idx]

            if found_count > 1:
                all_stride_arrays.append(stride_arrays)

        return all_stride_arrays


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, stride_array = self.samples[idx]
        arrays = []
        seq_idxs = [self.special_tokens['SEQ']]
        array_counter = 0

        for i, file_name in enumerate(stride_array):
            if file_name is None:
                seq_idxs.append(self.special_tokens['MASK'])
            else:
                arrays.append(self._load_array(path, file_name))
                seq_idxs.append(array_counter)
                array_counter += 1

        seq_label_idxs = [False] * len(seq_idxs)
        label_idxs = torch.tensor(seq_idxs.copy())


        seq_matches = torch.where(torch.tensor(seq_idxs) >= 0)[0]
        rand_labels = torch.rand(len(seq_matches)) < self.label_prob
        if rand_labels.sum() == 0:
            rand_labels[torch.randint(len(seq_matches), (1,))] = True
        for rand_idx in seq_matches[rand_labels]:
            seq_label_idxs[rand_idx] = True

            if torch.rand(1).item() >= self.label_keep_prob:
                seq_idxs[rand_idx] = self.special_tokens['MASK']

        arrays = torch.stack(arrays)
        seq_label_idxs = torch.tensor(seq_label_idxs)
        label_arrays = arrays[label_idxs[seq_label_idxs]]

        return arrays, torch.tensor(seq_idxs), seq_label_idxs, label_arrays

    def _load_array(self, path, frame_stem):
        path = Path(path, frame_stem).with_suffix(self.array_ext)
        keypoints = np.load(path)
        keypoints_norm = zscore(keypoints)
        keypoints_dist = pdist(keypoints_norm)
        return torch.tensor(keypoints_dist, dtype=torch.float)


def cycle(loader):
    while True:
        for data in loader:
            yield data

