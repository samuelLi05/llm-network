from pathlib import Path
import sys
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR.parent
if str(MODELS_DIR) not in sys.path:
	sys.path.insert(0, str(MODELS_DIR))

from data_prep import build_neighbors_index, build_run_trajectory


def test_build_neighbors_index_is_reverse_lookup_with_self_fallback():
	data = {
		'graph': {
			'agent_0': ['agent_1', 'agent_2'],
			'agent_2': ['agent_1'],
		}
	}
	global_agent_ids = ['agent_0', 'agent_1', 'agent_2', 'agent_3']

	neighbors = build_neighbors_index(data, global_agent_ids)

	assert neighbors[0] == [0]
	assert neighbors[1] == [0, 2]
	assert neighbors[2] == [0]
	assert neighbors[3] == [3]


def test_build_run_trajectory_treats_slice0_as_given_initial_condition():
	global_agent_ids = ['agent_0', 'agent_1']
	data = {
		'message_events': [
			(0.0, 'agent_0', 0.25),
			(20000.0, 'agent_1', -0.60),
		],
		'profile_seed_by_slice': {
			'agent_0': {0: 0.10},
			'agent_1': {0: -0.20},
		},
	}

	traj, post_mask = build_run_trajectory(
		data,
		global_agent_ids,
		target_agent_fraction=0.4,
		constrain_messages=2,
		return_post_mask=True,
	)

	first_agent1_post = int(np.where(post_mask[:, 1])[0][0])

	assert np.isclose(traj[0, 0], 0.25)
	assert np.isclose(traj[0, 1], -0.20)
	assert first_agent1_post > 0
	assert np.allclose(traj[:first_agent1_post, 1], -0.20)
	assert np.isclose(traj[first_agent1_post, 1], -0.60)
