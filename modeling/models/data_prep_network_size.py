"""
    Extra data preparation functions for analysis with varied network size, 
    and more generally, runs that have been rscored
"""

import json
from pathlib import Path
import numpy as np
from collections import defaultdict
from typing import Callable, Dict, List, Tuple
from modeling.models.data_prep import (
    _load_json, 
    _parse_agent_id, 
    _load_jsonl,
    _numeric_agent_key,
    compute_required_time_slice_ms,
    _bucket_events_to_slices
)

def load_cleaned_run_data(run_dir):

    # Load the data for a single run using the cleaned 
    #  data generated with log correcter

    run_dir = Path(run_dir)
    graph = defaultdict(list)

    g = _load_json(run_dir / 'connection_graph.json')
    for k, vals in g.items():
        ps = _parse_agent_id(k)
        graph[ps] = [_parse_agent_id(y) for y in vals if _parse_agent_id(y)]

    
    message_events = []
    for row in _load_jsonl(run_dir / 'messages_with_alignment.jsonl'):
        aid = _parse_agent_id(row.get('sender_id'))
        ss = row.get('published').get('stance_score')

        t_ms = float((row.get('time') or {}).get('t_ms', np.nan))
        if np.isfinite(t_ms):
            message_events.append((t_ms, aid, float(ss)))
        else:
            raise ValueError(f"Invalid t_ms value: {t_ms} in row: {row}")
        
    all_agents = set(graph.keys())
    for src, dsts in graph.items():
        all_agents.add(src)
        all_agents.update(dsts)
    sorted_agents = sorted(all_agents, key=_numeric_agent_key)

    # load in initial opinions directly from initial_stance_map.json
    init_stance_log = _load_json(run_dir / 'initial_stance_map.json')

    init_stance_map = {}

    # iterate over key-vals in init_stance_log
    for k, v in init_stance_log.items():
        ps = _parse_agent_id(k)
        if ps is not None:
            init_stance_map[ps] = v["recomputed"]
        else:
            raise ValueError(f"Invalid agent ID: {k} in initial_stance_map.json")

    return {
        'run_name': run_dir.name,
        'graph': graph,
        'agent_ids': sorted_agents,
        'message_events': message_events,
        'initial_stance_map': init_stance_map
    }



def build_run_trajectory_from_clean(
    data,
    global_agent_ids, 
    target_agent_fraction,
    constrain_messages=150,
    return_post_mask=False,
    reference_agent_number = None
):
    agent_index = {a: i for i, a in enumerate(global_agent_ids)}
    slice_ms = compute_required_time_slice_ms(reference_agent_number, target_agent_fraction=target_agent_fraction)

    if reference_agent_number is None:
        raise ValueError("reference_agent_number must be provided for build_run_trajectory_from_clean")

    events = data.get('message_events', [])
    if constrain_messages is not None:
        if not isinstance(constrain_messages, int):
            raise TypeError("constrain_messages must be an integer")
        if constrain_messages < 1:
            raise ValueError("constrain_messages must be >= 1")
        
        # sort by t_ms, and take the first constrain_messages events
        events = sorted(events, key=lambda x: x[0])[:constrain_messages]    
        
    rebucketed_slice_obs, last_slice = _bucket_events_to_slices(events, slice_ms)
    T = int(last_slice)

    traj = np.full((T + 1, len(global_agent_ids)), np.nan, dtype=float)
    post_mask = np.zeros((T + 1, len(global_agent_ids)), dtype=bool)
    x0 = np.full(len(global_agent_ids), np.nan, dtype=float)

    slice0_obs = rebucketed_slice_obs.get(0, [])    # dict: agent_id -> stance_score (if any)

    for a in global_agent_ids:
        i = agent_index[a]
        if a in slice0_obs:
            x0[i] = float(slice0_obs[a])
            post_mask[0, i] = True
            continue
            
        # pull from initial_stance_map to get the initial stance if not in slice0_obs
        init_stance_map = data.get('initial_stance_map', {})
        if a in init_stance_map:
            x0[i] = float(init_stance_map[a])
        else:
            raise ValueError(f"Agent {a} not found in initial_stance_map or slice0_obs for run {data.get('run_name')}")
        
    traj[0] = x0
    for slice_idx in range(1, T + 1):
        traj[slice_idx] = traj[slice_idx - 1]
        obs = rebucketed_slice_obs.get(slice_idx, {})
        for a, val in obs.items():
            if a in agent_index:
                j = agent_index[a]
                traj[slice_idx, j] = float(val)
                post_mask[slice_idx, j] = True

    if return_post_mask:
        return traj, post_mask
    else:
        return traj