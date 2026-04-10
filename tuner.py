import itertools
import importlib
import json
import numpy as np
import matplotlib
from pathlib import Path
from functools import partial
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

matplotlib.use('Agg')

MODEL_PATH = "./models/tinyphysics.onnx"
DATA_PATH = "./data/SYNTHETIC"
NUM_SEGS = 100

param_grid = {
    'ff_gain':         [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'lookahead_steps': [5, 10, 15, 20],
    'p':               [0.15, 0.2, 0.25, 0.3],
    'i':               [0.05, 0.1, 0.15],
    'd':               [-0.01, -0.03, -0.05, -0.08],
}

def run_single(data_path, params):
	model = TinyPhysicsModel(MODEL_PATH, debug=False)
	mod = importlib.import_module('controllers.feedforward_pid')
	controller = mod.Controller()
	for k, v in params.items():
		setattr(controller, k, v)
	sim = TinyPhysicsSimulator(model, str(data_path), controller=controller, debug=False)
	return sim.rollout()['total_cost']

def evaluate_params(params, files):
	runner = partial(run_single, params=params)
	costs = process_map(runner, files, max_workers=16, chunksize=10, disable=True)
	return np.mean(costs)

def evaluate_model(params, model, files):
	costs = []
	for f in files:
		controller_mod = importlib.import_module('controllers.feedforward_pid')
		controller = controller_mod.Controller()
		for k, v in params.items():
			setattr(controller, k, v)
		sim = TinyPhysicsSimulator(model, str(f), controller=controller, debug=False)
		costs.append(sim.rollout()['total_cost'])
	return np.mean(costs)

if __name__ == "__main__":
	model = TinyPhysicsModel(MODEL_PATH, debug=False)
	files = sorted(Path(DATA_PATH).rglob('*.csv'))[:NUM_SEGS]
	keys = list(param_grid.keys())
	combos = list(itertools.product(*param_grid.values()))

	best_score = float('inf')
	best_params = None

	for value in tqdm(combos, total=len(combos), desc='Evaluating models'):
		params = dict(zip(keys, value))
		score = evaluate_params(params, files)
		if score < best_score:
			best_score = score
			best_params = params
			tqdm.write(f'New best score: {best_score} with params: {best_params}')

	print(f"\nBest score: {best_score}")
	print(f"Best params: {json.dumps(best_params, indent=2)}")

