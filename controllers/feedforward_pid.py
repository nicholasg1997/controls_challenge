import numpy as np
from . import BaseController

class Controller(BaseController):
	def __init__(self):
		self.p = 0.2
		self.i = 0.1
		self.d = -0.01

		self.ff_gain = 0.5
		self.lookahead_steps = 10

		self.error_integral = 0
		self.prev_error = 0
		self.integral_limit = 50.0

	def update(self, target_lataccel, current_lataccel, state, future_plan):
		v = max(state.v_ego, 1.0)

		if future_plan.lataccel and len(future_plan.lataccel) >= self.lookahead_steps:
			lookahead_target = np.mean(future_plan.lataccel[:self.lookahead_steps])
		else:
			lookahead_target = target_lataccel

		desired_correction = lookahead_target - state.roll_lataccel
		feedforward = self.ff_gain * desired_correction

		error = target_lataccel - current_lataccel
		self.error_integral = np.clip(self.error_integral + error,
		                              -self.integral_limit,
		                              self.integral_limit)
		error_diff = error - self.prev_error
		self.prev_error = error

		feedback = (self.p*error +
		            self.i*self.error_integral +
		            self.d*error_diff)

		return feedback + feedforward

