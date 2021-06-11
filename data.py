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
    "SEQ": -2,
    "SEP": -3,
    "PAD": -4,
    "MISSING": -5
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
        root_path,
        clips_list,
        sample_length=30 * 2, #frames
        num_keypoints=68,
        label_prob=0.15,
        array_ext='.npy',
    ):
        self.root_path = Path(root_path)
        self.clips_list = Path(clips_list)
        self.sample_length = sample_length
        self.num_keypoints = num_keypoints
        self.label_prob = label_prob
        self.array_ext = array_ext

        self.clips = self._get_clips()
        
    def _sort_filenames(self, names):
        return sorted(names, key=lambda name: int(name.stem))

    def _get_clips(self):
        clips = []
        for line in self.clips_list.open("r"):
            clip_path = line.strip("\n")
            clip_path = Path(self.root_path, clip_path)
            
            clips.append(clip_path)
        return clips

    def __len__(self):
        return len(self.clips)
    
    def _load_array(self, path):
        keypoints = np.load(path)
        keypoints_norm = zscore(keypoints)
        return torch.tensor(keypoints_norm, dtype=torch.float)
    
    def _empty_array(self):
        return torch.zeros((self.num_keypoints, 2), dtype=torch.float)
    
    def _random_sample(self, path, frames):
        first_idx = int(frames[0].stem)
        last_idx = int(frames[-1].stem)
        cutoff = max(first_idx, last_idx - self.sample_length * 2)
        start_idx = random.randint(first_idx, cutoff)
        end_idx = start_idx + self.sample_length * 2
        
        frame_idx = 0
        while int(frames[frame_idx].stem) < start_idx:
            frame_idx += 1
        samples = []
        for idx in range(start_idx, min(last_idx, end_idx)):
            frame = frames[frame_idx]
            frame_id = int(frame.stem)
            if frame_id == idx:
                samples.append(Path(path, frame))
                frame_idx += 1
            else:
                samples.append(SPECIAL_TOKENS['MISSING'])
                
        return samples

    def _split_samples(self, samples):
        if random.random() > 0.5:
            samples1 = samples[:self.sample_length]
            samples2 = samples[self.sample_length:]
        else:
            samples1 = samples[-self.sample_length:]
            samples2 = samples[:-self.sample_length]
            
        return samples1, samples2
    
    def _concat_samples(self, samples1, samples2):
        samples1.append(SPECIAL_TOKENS['SEP'])
        samples2.append(SPECIAL_TOKENS['SEP'])
        if len(samples2) < self.sample_length + 1:
            samples2.extend([SPECIAL_TOKENS['PAD']] * (self.sample_length + 1 - len(samples2)))
        return [SPECIAL_TOKENS['SEQ']] + samples1 + samples2
    
    def _read_and_mask(self, samples):
        arrays, special_mask = [], []
        for sample in samples:
            if type(sample) == int:
                arrays.append(self._empty_array())
                special_mask.append(sample)
            else:
                arrays.append(self._load_array(sample))
                special_mask.append(0)
                
        arrays = torch.stack(arrays)
        special_mask = torch.tensor(special_mask, dtype=torch.long)
        
        found_idxs = torch.where(special_mask == 0)[0]
        mask = torch.rand(found_idxs.shape) < self.label_prob
        if not torch.any(mask):
            mask[random.randint(0, found_idxs.shape[0]-1)] = True
        special_mask[found_idxs[mask]] = SPECIAL_TOKENS['MASK']
        
        return arrays, special_mask

    def _get_frames(self, path):
        frames = list(path.glob(F"*{self.array_ext}"))
        frames = self._sort_filenames(frames)
        return frames

    def __getitem__(self, clip_index):
        path = self.clips[clip_index]
        frames = self._get_frames(path)

        samples = self._random_sample(path, frames)
        samples1, samples2 = self._split_samples(samples)
        
        nsp_label = 1
        if random.random() > 0.5:
            nsp_label = 0
            rand_path = random.choice(self.clips)
            rand_frames = self._get_frames(rand_path)
            rand_samples = self._random_sample(rand_path, rand_frames)
            rand_samples1, rand_samples2 = self._split_samples(rand_samples)
            rand_sample = rand_samples1 if random.random() > 0.5 else rand_samples2
            samples2 = rand_sample
            
        samples = self._concat_samples(samples1, samples2)
        arrays, special_mask = self._read_and_mask(samples)
        
        return arrays, special_mask, nsp_label

    
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

