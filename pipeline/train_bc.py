import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from pathlib import Path

# CONSTANTS
ACC = 9.81
TRAIN_START = 5000
TRAIN_END = 19999
HISTORY_LEN = 5
LOOKAHEAD_LEN = 10
SEQ_LEN = 20
INPUT_DIM = HISTORY_LEN + 3 + LOOKAHEAD_LEN

os.makedirs("pipeline", exist_ok=True)


# LOADING DATA
def load_segments(data_dir: str) -> list[dict]:
	assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist"
	all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
	print(f"Found {len(all_files)} files")
	train_files = [f for f in all_files
	                if TRAIN_START <= int(Path(f).stem) <= TRAIN_END]

	print(f"Using {len(train_files)} training files")

	segments = []
	for f in tqdm(train_files, desc="Loading segments"):
		try:
			df = pd.read_csv(f)
			df_labeled = df[df['steerCommand'].notna()].reset_index(drop=True)

			if len(df_labeled) < (HISTORY_LEN + LOOKAHEAD_LEN + SEQ_LEN):
				continue

			segments.append({
				'roll_lataccel': np.sin(df_labeled['roll'].values) * ACC,
				'v_ego': df_labeled['vEgo'].values,
				'a_ego': df_labeled['aEgo'].values,
				'target': df_labeled['targetLateralAcceleration'].values,
				'steer': -df_labeled['steerCommand'].values,
				'full_target': df['targetLateralAcceleration'].values,
				'full_t': df['t'].values,
				'labeled_t': df_labeled['t'].values,
			})
		except Exception as e:
			print(f"Error loading {f}: {e}")

	print(f"Loaded {len(segments)} segments")
	print(f"Total labeled rows: {sum([len(s['steer']) for s in segments]):,}")
	return segments

def compute_stats(segments: list[dict]) -> dict:
	"""Compute normalization stats over each column for the training data"""
	stats = {}
	for key in segments[0].keys():
		vals = np.concatenate([s[key] for s in segments])
		stats[key] = {
			'mean': vals.mean(),
			'std': vals.std(),
		}

	print(f"\nNormalization stats:\n")
	print(f"    {'key':<15} {'mean':>8} {'std':>8}")
	print(f"    {'-'*35}")
	for key, val in stats.items():
		print(f"    {key:<15} {val['mean']:>8.4f} {val['std']:>8.4f}")
	return stats


# DATASET
class SteeringDataset(Dataset):
	def __init__(self, segments: list[dict], stats: dict):
		self.stats = stats
		self.segments = segments
		self.windows = []

	def build_windows(self, segments: list[dict], stats: dict):
		for seg in tqdm(segments, desc="Building windows"):
			n = len(seg['steer'])
			lookahead = self._build_lookahead(seg)
			for start in range(HISTORY_LEN,
			                   n-LOOKAHEAD_LEN-SEQ_LEN+1,
			                   SEQ_LEN):
				feat_seq = []
				label_seq = []

				for i in range(start, start+SEQ_LEN):
					past_steers = self._norm(seg['steer'][i-HISTORY_LEN:i], 'steer')
					state = np.array([
						self._norm(seg['v_ego'][i], 'v_ego'),
						self._norm(seg['a_ego'][i], 'a_ego'),
						seg['roll_lataccel'][i],
					])

					future = self._norm(lookahead[i], 'target')

					features = np.concatenate([past_steers, state, future]).astype(np.float32)
					label = np.array(seg['steer'][i], dtype=np.float32)
					feat_seq.append(features)
					label_seq.append(label)
				self.windows.append([np.stack(feat_seq),
				                     np.stack(label_seq)])

			print(f"Built {len(self.windows)} windows")

	def _build_lookahead(self, seg:dict) -> np.ndarray:
		n = len(seg['labeled_t'])
		lookahead = np.zeros((n, LOOKAHEAD_LEN), dtype=np.float32)

		full_target = seg['full_target']
		full_t = seg['full_t']
		labeled_t = seg['labeled_t']

		for i, t in enumerate(labeled_t):
			full_idx = np.searchsorted(full_t, t)
			end_idx = full_idx + LOOKAHEAD_LEN + 1
			if end_idx <= len(full_target):
				lookahead[i] = full_target[full_idx+1:end_idx]
			else:
				available = full_target[full_idx+1:]
				pad = np.full(LOOKAHEAD_LEN, full_target[-1])
				pad[:len(available)] = available
				lookahead[i] = pad
		return lookahead

	def _norm(self, x, key):
		return (x - self.stats[key]['mean']) / self.stats[key]['std']

	def __len__(self):
		return len(self.windows)

	def __getitem__(self, idx):
		feat, label = self.windows[idx]
		return torch.from_numpy(feat), torch.from_numpy(label)

class SteeringMLP(nn.Module):
	pass

if __name__ == "__main__":
	segments = load_segments("../data/SYNTHETIC")
	stats = compute_stats(segments)

