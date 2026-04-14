import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends.mkl import verbose
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from pathlib import Path

# CONSTANTS
ACC = 9.81
TRAIN_START = 5000
TRAIN_END = 19999
HISTORY_LEN = 5
LATACCEL_HIST_LEN = 1
LOOKAHEAD_LEN = 10
SEQ_LEN = 20
INPUT_DIM = HISTORY_LEN + 4 + LATACCEL_HIST_LEN + LOOKAHEAD_LEN

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

			target_vals = df_labeled['targetLateralAcceleration'].values
			lataccel_hist = np.stack([
				np.concatenate([np.full(k, target_vals[0]), target_vals[:-k]])
				for k in range(1, LATACCEL_HIST_LEN + 1)
			], axis=1)
			segments.append({
				'roll_lataccel': np.sin(df_labeled['roll'].values) * ACC,
				'v_ego': df_labeled['vEgo'].values,
				'a_ego': df_labeled['aEgo'].values,
				'target': target_vals,
				'steer': -df_labeled['steerCommand'].values,
				'current_target': target_vals,
				'lataccel_hist': lataccel_hist,

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
		self.build_windows(self.segments, self.stats)

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
						self._norm(seg['current_target'][i], 'current_target'),
					])
					lataccel_hist = self._norm(seg['lataccel_hist'][i], 'lataccel_hist')

					future = self._norm(lookahead[i], 'target')

					features = np.concatenate([past_steers, state, lataccel_hist, future]).astype(np.float32)
					label = np.array([seg['steer'][i]], dtype=np.float32)
					feat_seq.append(features)
					label_seq.append(label)
				self.windows.append((np.stack(feat_seq),
				                     np.stack(label_seq)))

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
		std = self.stats[key]['std']
		if std < 1e-8:
			return x - self.stats[key]['mean']
		return (x - self.stats[key]['mean']) / std

	def __len__(self):
		return len(self.windows)

	def __getitem__(self, idx):
		feat, label = self.windows[idx]
		return torch.from_numpy(feat), torch.from_numpy(label)

class SteeringMLP(nn.Module):
	def __init__(self, input_dim: int = INPUT_DIM):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dim, 64),
			nn.Mish(),
			nn.Linear(64, 32),
			nn.Mish(),
			nn.Linear(32, 1),
		)

	def forward(self, x):
		if x.dim() == 3:
			batch, seq_len, input_dim = x.shape
			out = self.net(x.view(batch*seq_len, input_dim))
			return out.view(batch, seq_len, 1)
		return self.net(x)

def steering_loss(predictions, labels, alpha=0.1):
	if labels.dim() == 2:
		labels = labels.unsqueeze(-1)
	if predictions.shape != labels.shape:
		raise ValueError(
			f"Shape mismatch in steering_loss: predictions={tuple(predictions.shape)} "
			f"labels={tuple(labels.shape)}"
		)
	mse = F.mse_loss(predictions, labels)
	jerk = predictions[:, 1:, :] - predictions[:, :-1, :]
	jerk_loss = (jerk ** 2).mean()
	total = mse + alpha * jerk_loss
	return total, mse.item(), jerk_loss.item()

def train(data_dir: str, epochs: int = 40,
          batch_size: int = 128, lr: float = 1e-3,
          alpha: float = 0.8, max_norm: float = 1.0):

	val_percent = 0.15
	weight_decay = 1e-4

	device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
	print(f"Using device: {device}")

	segments = load_segments(data_dir)
	if not segments:
		raise ValueError(
			f"No usable segments were loaded from {data_dir}. "
			f"Check the path and filtering constants (TRAIN_START={TRAIN_START}, TRAIN_END={TRAIN_END})."
		)
	stats = compute_stats(segments)

	with open("pipeline/stats.pkl", "wb") as f:
		pickle.dump(stats, f)

	full_ds = SteeringDataset(segments, stats)
	if len(full_ds) == 0:
		raise ValueError(
			"No training windows were built. "
			"Verify HISTORY_LEN/LOOKAHEAD_LEN/SEQ_LEN and your labeled data."
		)
	val_size = int(val_percent * len(full_ds))
	if len(full_ds) > 1:
		val_size = max(1, val_size)
	val_size = min(val_size, len(full_ds) - 1)
	train_ds, val_ds = random_split(full_ds,
	                                [len(full_ds)-val_size, val_size],
	                                generator=torch.Generator().manual_seed(42))
	if len(train_ds) == 0:
		raise ValueError("Training split is empty after dataset split; adjust val_percent or dataset size.")

	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
	                          num_workers=0, pin_memory=True)
	val_loader = None
	if len(val_ds) > 0:
		val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
		                        num_workers=0, pin_memory=True)

	model = SteeringMLP(INPUT_DIM).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr,
	                             weight_decay=weight_decay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, 'min', patience=3, factor=0.5)

	total_params = sum(p.numel() for p in model.parameters())
	print(f"Total parameters: {total_params:,}")

	best_val = float('inf')
	best_epoch = 0

	for epoch in range(epochs):
		model.train()
		t_loss, t_mse, t_jerk = 0.0, 0.0, 0.0
		for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1:>2}/{epochs}", leave=False):
			features, labels = features.to(device), labels.to(device)
			optimizer.zero_grad()
			preds = model(features)
			loss, mse, jerk = steering_loss(preds, labels, alpha)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
			optimizer.step()
			t_loss += loss.item()
			t_mse += mse
			t_jerk += jerk

		# EVALUATE MODEL
		model.eval()
		v_loss, v_mse, v_jerk = 0.0, 0.0, 0.0
		if val_loader is not None:
			with torch.no_grad():
					for features, labels in tqdm(val_loader, desc="Evaluating model", leave=False):
						features, labels = features.to(device), labels.to(device)
						preds = model(features)
						loss, mse, jerk = steering_loss(preds, labels, alpha)
						v_loss += loss.item()
						v_mse += mse
						v_jerk += jerk
		n_tr = len(train_loader)
		train_loss = t_loss / n_tr
		train_mse = t_mse / n_tr
		train_jerk = t_jerk / n_tr
		if val_loader is not None:
			n_val = len(val_loader)
			val_loss = v_loss / n_val
			val_mse = v_mse / n_val
			val_jerk = v_jerk / n_val
			print(f"Epoch {epoch+1:>2}/{epochs}: "
			      f"train loss={train_loss:.4f}, "
			      f"(mse={train_mse:.4f}, jerk={train_jerk:.4f}), "
			      f"val loss={val_loss:.4f}, "
			      f"(mse={val_mse:.4f}, jerk={val_jerk:.4f}), "
			      f"lr={optimizer.param_groups[0]['lr']:.1e}")
			scheduler.step(val_loss)
			model_score = val_loss
		else:
			print(f"Epoch {epoch+1:>2}/{epochs}: "
			      f"train loss={train_loss:.4f}, "
			      f"(mse={train_mse:.4f}, jerk={train_jerk:.4f}), "
			      f"lr={optimizer.param_groups[0]['lr']:.1e}")
			model_score = train_loss

		if model_score < best_val:
			best_val = model_score
			best_epoch = epoch + 1
			torch.save({
				'model_state': model.state_dict(),
				'stats': stats,
				'input_dim': INPUT_DIM,
				'history_len': HISTORY_LEN,
				'lookahead_len': LOOKAHEAD_LEN,
			}, 'pipeline/steering_mlp.pth')
			print(f"New best model saved (val = {best_val:.4f}, epoch = {best_epoch})")

	print(f"Training complete."
	      f"Best val loss: {best_val:.4f} at epoch {best_epoch}")




if __name__ == "__main__":
	train("../data/SYNTHETIC/")
