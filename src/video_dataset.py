import os
import os.path as path
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


MB140_DATASETS = {
    "MB140_Bern_Phase_1fps",
    "MB140_Bern_Step_1fps",
    "MB140_Stras_Phase_1fps",
    "MB140_Stras_Step_1fps",
    "Cholec80_Phase_1fps",
    "AutoLaparo_Phase_1fps",
}


def parse_action_name(fname: str, dataset: str) -> str:
    if dataset == "Breakfast":
        return fname.split("_")[-1]
    if dataset == "YTI":  # ignores _idt files in groundTruth automatically
        return "_".join(fname.split("_")[:-1])
    if dataset in ["FS", "desktop_assembly"]:
        return ""  # single activity class
    if dataset in MB140_DATASETS:
        return ""  # single activity class
    raise ValueError(f"{dataset} is not a valid dataset!")


def _standardize(feats: np.ndarray) -> np.ndarray:
    zmask = np.ones(feats.shape[0], dtype=bool)
    for rdx, row in enumerate(feats):
        if np.sum(row) == 0:
            zmask[rdx] = False

    out = np.zeros(feats.shape, dtype=np.float32)
    if zmask.any():
        z = feats[zmask] - np.mean(feats[zmask], axis=0)
        z = z / np.std(feats[zmask], axis=0)
        out[zmask] = z

    out = np.nan_to_num(out)
    out /= np.sqrt(out.shape[1])
    return out


def _load_text_aligned(npz_path: Path, inds: np.ndarray) -> np.ndarray:
    """
    Align each sampled time index (in seconds) to the caption window from NPZ:
      embeddings: (W, Dt)
      start_sec:  (W,)
      end_sec:    (W,)
    We pick window w where start[w] <= t < end[w].
    """
    z = np.load(str(npz_path))
    emb = z["embeddings"]
    start = z["start_sec"]
    end = z["end_sec"]

    t = inds.astype(np.int32)
    w_idx = np.searchsorted(end, t, side="right")
    w_idx = np.clip(w_idx, 0, emb.shape[0] - 1)

    bad = t < start[w_idx]
    w_idx[bad] = np.clip(w_idx[bad] - 1, 0, emb.shape[0] - 1)

    return emb[w_idx]


class VideoDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        dataset: str,
        n_frames,
        standardise: bool = True,
        split: str | None = None,
        random: bool = True,
        n_videos=None,
        action_class=["all"],
        visual_dir: str | None = None,   
        text_dir: str | None = None,     
    ):
        self.root_dir = root_dir
        self.dataset = dataset
        self.data_dir = path.join(root_dir, self.dataset)

        self.visual_dir = visual_dir
        self.text_dir = text_dir

        if self.dataset == "FSeval":
            self.dataset = "FS"
            granularity = "eval"
        else:
            granularity = None

        # -------------------------
        # list GT files
        # -------------------------
        gt_dir = path.join(self.data_dir, "groundTruth")
        all_files = os.listdir(gt_dir)

        if self.dataset in MB140_DATASETS:
            self.video_fnames = sorted([f for f in all_files if f.endswith(".txt")])
        else:
            self.video_fnames = sorted(
                [f for f in all_files if len(f.split("_")) > 1 or len(f.split("-")) > 1]
            )

        # activity filtering (original behavior)
        if self.dataset in ["FS", "desktop_assembly"] or self.dataset in MB140_DATASETS:
            action_class = ""

        if action_class != ["all"]:
            if isinstance(action_class, list):
                self.video_fnames = [
                    fname for fname in self.video_fnames
                    if parse_action_name(fname, self.dataset) in action_class
                ]
            else:
                self.video_fnames = [
                    fname for fname in self.video_fnames
                    if parse_action_name(fname, self.dataset) == action_class
                ]

        # optional subsampling
        if n_videos is not None:
            self.video_fnames = self.video_fnames[:: int(len(self.video_fnames) / n_videos)]

        # fold split support (preserve split order)
        if split is not None:
            split_path = path.join(self.data_dir, split)
            if not path.exists(split_path):
                raise FileNotFoundError(f"Split file not found: {split_path}")

            with open(split_path, "r") as f:
                vids = [ln.strip() for ln in f.readlines() if ln.strip()]

            existing = set(self.video_fnames)  # include ".txt"
            ordered = [v + ".txt" for v in vids if (v + ".txt") in existing]
            self.video_fnames = ordered

            if len(self.video_fnames) == 0:
                raise ValueError(f"Split produced 0 videos. Check split file: {split_path}")

        # mapping
        def prep(x: str):
            i, nm = x.rstrip().split(" ")
            if self.dataset in MB140_DATASETS:
                return str(i), int(i)   # "0" -> 0
            else:
                return nm, int(i)

        if granularity is None:
            action_mapping = list(map(prep, open(path.join(self.data_dir, "mapping/mapping.txt"))))
        else:
            action_mapping = list(map(prep, open(path.join(self.data_dir, f"mapping/mapping{granularity}.txt"))))

        self.action_mapping = dict(action_mapping)
        self.n_subactions = len(set(self.action_mapping.values()))
        self.n_frames = n_frames
        self.standardise = standardise
        self.random = random

    def __len__(self):
        return len(self.video_fnames)

    def __getitem__(self, idx):
        video_fname = self.video_fnames[idx]           # e.g. "BBP01.txt"
        vid = os.path.splitext(video_fname)[0]         # "BBP01"

        # load GT labels
        gt_lines = [line.rstrip() for line in open(path.join(self.data_dir, "groundTruth", video_fname))]
        n_gt = len(gt_lines)

        # compute safe available length across GT/vis/txt
        n_avail = n_gt

        if self.visual_dir is not None:
            vpath = Path(self.visual_dir) / f"{vid}.npy"
            n_vis = int(np.load(str(vpath), mmap_mode="r").shape[0])
            n_avail = min(n_avail, n_vis)

        if self.text_dir is not None:
            tpath = Path(self.text_dir) / f"{vid}.npz"
            z = np.load(str(tpath))
            n_txt = int(z["end_sec"].max()) if "end_sec" in z.files else n_gt
            n_avail = min(n_avail, n_txt)

        # sample indices
        inds, mask = self._partition_and_sample(self.n_frames, n_avail)

        # map GT -> ids
        gt = torch.tensor([self.action_mapping[gt_lines[i]] for i in inds]).long()

        # MULTIMODAL return
        if self.visual_dir is not None and self.text_dir is not None:
            v = np.load(str(Path(self.visual_dir) / f"{vid}.npy"))[inds, :]

            txt = _load_text_aligned(Path(self.text_dir) / f"{vid}.npz", inds)

            if self.standardise:
                v = _standardize(v)
                txt = _standardize(txt)

            v = torch.from_numpy(v).float()
            txt = torch.from_numpy(txt).float()
            return (v, txt), mask, gt, video_fname, gt.unique().shape[0]

        # VISUAL-ONLY (original behavior)
        action = parse_action_name(video_fname, self.dataset)
        if action == "":
            feat_base = path.join(self.data_dir, "features", vid)
        else:
            feat_base = path.join(self.data_dir, "features", action, vid)

        try:
            features = np.loadtxt(feat_base + ".txt")[inds, :]
        except Exception:
            features = np.load(feat_base + ".npy")[inds, :]

        if self.standardise:
            features = _standardize(features)

        features = torch.from_numpy(features).float()
        return features, mask, gt, video_fname, gt.unique().shape[0]

    def _partition_and_sample(self, n_samples, n_frames):
        if n_samples is None:
            indices = np.arange(n_frames)
            mask = np.full(n_frames, 1, dtype=bool)
        elif n_samples < n_frames:
            if self.random:
                boundaries = np.linspace(0, n_frames - 1, n_samples + 1).astype(int)
                indices = np.random.randint(low=boundaries[:-1], high=boundaries[1:])
            else:
                indices = np.linspace(0, n_frames - 1, n_samples).astype(int)
            mask = np.full(n_samples, 1, dtype=bool)
        else:
            indices = np.concatenate((np.arange(n_frames), np.full(n_samples - n_frames, n_frames - 1)))
            mask = np.concatenate((np.full(n_frames, 1, dtype=bool), np.zeros(n_samples - n_frames, dtype=bool)))
        return indices, mask
