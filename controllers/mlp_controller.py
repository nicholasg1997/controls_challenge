import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import warnings
from . import BaseController

HISTORY_LEN      = 5
LATACCEL_HIST_LEN = 1
LOOKAHEAD_LEN    = 10
INPUT_DIM        = HISTORY_LEN + 4 + LATACCEL_HIST_LEN + LOOKAHEAD_LEN


class SteeringMLP(nn.Module):
    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Mish(),
            nn.Linear(64, 32),
            nn.Mish(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


class Controller(BaseController):
    def __init__(self):
        checkpoint_path = (
            Path(__file__).resolve().parents[1]
            / "pipeline"
            / "pipeline"
            / "steering_mlp.pth"
        )
        checkpoint = self._load_checkpoint(checkpoint_path)
        self.stats  = checkpoint['stats']
        self.model  = SteeringMLP(INPUT_DIM)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

        # Rolling history buffers
        self.steer_history    = [0.0] * HISTORY_LEN
        self.lataccel_history = [0.0] * LATACCEL_HIST_LEN

    @staticmethod
    def _load_checkpoint(checkpoint_path: Path):
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. "
                "Ensure the model artifact exists before running rollouts."
            )

        try:
            return torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except pickle.UnpicklingError:
            # PyTorch 2.6+ defaults to weights_only=True and blocks some numpy globals.
            # This checkpoint stores numpy scalar + dtype metadata under `stats`.
            safe_globals = []
            scalar = getattr(np.core.multiarray, "scalar", None)
            if scalar is not None:
                safe_globals.append(scalar)
            dtype_cls = getattr(np, "dtype", None)
            if dtype_cls is not None:
                safe_globals.append(dtype_cls)

            # NumPy 2.x dtype classes (e.g., numpy.dtypes.Float64DType) are
            # serialized in some checkpoints and must be allowlisted.
            np_dtypes = getattr(np, "dtypes", None)
            if np_dtypes is not None:
                for name in dir(np_dtypes):
                    if not name.endswith("DType"):
                        continue
                    dtype_type = getattr(np_dtypes, name, None)
                    if isinstance(dtype_type, type):
                        safe_globals.append(dtype_type)

            if safe_globals and hasattr(torch.serialization, "safe_globals"):
                with torch.serialization.safe_globals(safe_globals):
                    return torch.load(
                        checkpoint_path, map_location='cpu', weights_only=True
                    )

            warnings.warn(
                "Falling back to torch.load(..., weights_only=False) for a trusted "
                "local checkpoint that includes numpy metadata.",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    def _norm(self, x, key):
        s = self.stats[key]
        return (x - s['mean']) / s['std']

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Build feature vector — must match training exactly
        past_steers = self._norm(
            np.array(self.steer_history), 'steer'
        )
        current_state = np.array([
            self._norm(state.v_ego,       'v_ego'),
            self._norm(state.a_ego,       'a_ego'),
            state.roll_lataccel,
            self._norm(target_lataccel,   'current_target'),
        ])
        lataccel_hist = self._norm(np.array(self.lataccel_history), 'lataccel_hist')

        future = list(future_plan.lataccel[:LOOKAHEAD_LEN])
        if len(future) == 0:
            # Tail timesteps can have no lookahead; use current target as a stable fallback.
            future = [float(target_lataccel)] * LOOKAHEAD_LEN
        elif len(future) < LOOKAHEAD_LEN:
            future.extend([future[-1]] * (LOOKAHEAD_LEN - len(future)))
        future_norm = self._norm(np.array(future), 'target')

        features = np.concatenate(
            [past_steers, current_state, lataccel_hist, future_norm]
        ).astype(np.float32)

        with torch.no_grad():
            action = self.model(
                torch.from_numpy(features).unsqueeze(0)
            ).item()

        # Update histories
        self.steer_history.pop(0)
        self.steer_history.append(action)
        self.lataccel_history.pop(0)
        self.lataccel_history.append(current_lataccel)

        return action
