import json
from pathlib import Path
import numpy as np
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

Array = np.ndarray

FIXED_MEAN_MSGS_PER_SLICE = 15.0
FIXED_BASE_WINDOW_MS = 8000.0
FIXED_MSG_RATE_PER_MS = FIXED_MEAN_MSGS_PER_SLICE / FIXED_BASE_WINDOW_MS
FIXED_MAX_SLICE_MS = 120000
REQUIRED_SLICE_MS_BY_N = {}
REQUIRED_SLICE_MS_BY_N_AND_FRACTION = {}


def _load_json(path):
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)

def _load_jsonl(path):
    path = Path(path)
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows

def _parse_agent_id(v):
    if isinstance(v, str) and v.startswith('agent_'):
        return v
    return None

def _numeric_agent_key(agent_id):
    return agent_id.split('_')[-1]

def _expected_distinct_agents_no_repeat(n_agents, k):
    n = int(n_agents)
    kk = float(k)
    if n <= 0 or kk <= 0:
        return 0.0
    if n == 1:
        return 1.0
    if n == 2:
        return 1.0 if kk <= 1.0 else 2.0
    unseen_prob = ((n - 1) / n) * (((n - 2) / (n - 1)) ** (kk - 1.0))
    return float(n * (1.0 - unseen_prob))


def compute_required_time_slice_ms(n_agents, target_agent_fraction):
    n = int(n_agents)
    target_fraction = float(target_agent_fraction)
    cache_key = (n, target_fraction)
    if cache_key in REQUIRED_SLICE_MS_BY_N_AND_FRACTION:
        return REQUIRED_SLICE_MS_BY_N_AND_FRACTION[cache_key]

    if target_fraction <= 0.0:
        # Keep minimum one-message behavior for degenerate targets.
        required_msgs = 1
        required_ms = int(np.ceil(required_msgs / FIXED_MSG_RATE_PER_MS))
        required_ms = min(required_ms, FIXED_MAX_SLICE_MS)
        REQUIRED_SLICE_MS_BY_N_AND_FRACTION[cache_key] = required_ms
        return required_ms

    if target_fraction >= 1.0:
        REQUIRED_SLICE_MS_BY_N_AND_FRACTION[cache_key] = FIXED_MAX_SLICE_MS
        return FIXED_MAX_SLICE_MS

    if n <= 1:
        required_msgs = 1
    else:
        # Poisson arrivals
        # E[distinct(t)] / n = 1 - exp(-(lambda * t) / n)
        # Solve for t at target_fraction.
        required_ms_cont = -(n / FIXED_MSG_RATE_PER_MS) * np.log(1.0 - target_fraction)
        required_ms = int(np.ceil(required_ms_cont))
        required_ms = max(1, required_ms)
        required_ms = min(required_ms, FIXED_MAX_SLICE_MS)
        REQUIRED_SLICE_MS_BY_N_AND_FRACTION[cache_key] = required_ms
        return required_ms

    required_ms = int(np.ceil(required_msgs / FIXED_MSG_RATE_PER_MS))
    required_ms = min(required_ms, FIXED_MAX_SLICE_MS)
    REQUIRED_SLICE_MS_BY_N_AND_FRACTION[cache_key] = required_ms
    # print (required_ms)
    return required_ms


def _bucket_events_to_slices(message_events, slice_ms):
    ordered_events = sorted(message_events, key=lambda x: x[0])
    start_ms = ordered_events[0][0]
    slice_stance = defaultdict(dict)

    for t_ms, aid, ss in ordered_events:
        slice_idx = int((t_ms - start_ms) // slice_ms)
        slice_stance[slice_idx][aid] = float(ss)

    last_slice = max(slice_stance.keys()) if slice_stance else 0
    return dict(slice_stance), int(last_slice)

def load_run_data(run_dir):
    run_dir = Path(run_dir)
    graph = defaultdict(list)
    g = _load_json(run_dir / 'connection_graph.json')
    for s, dsts in g.items():
        ps = _parse_agent_id(s)
        graph[ps] = [_parse_agent_id(y) for y in dsts if _parse_agent_id(y)]

    stance_by_agent = defaultdict(lambda: defaultdict(list))
    msg_count = defaultdict(int)
    message_events = []
    for row in _load_jsonl(run_dir / 'messages_with_alignment.jsonl'):
        aid = _parse_agent_id(row.get('sender_id'))
        ts = row.get('time_slice')
        ss = row.get("published").get('stance_score')

        t = int(ts)
        stance_by_agent[aid][t].append(ss)
        msg_count[t] += 1
        t_ms = float((row.get('time') or {}).get('t_ms', np.nan))
        if np.isfinite(t_ms):
            message_events.append((t_ms, aid, float(ss)))

    profile_seed = defaultdict(lambda: defaultdict(list))
    for fp in sorted((run_dir / 'per_agent').glob('agent_*.jsonl') if (run_dir / 'per_agent').exists() else []):
        default_agent = fp.stem
        for row in _load_jsonl(fp):
            aid = _parse_agent_id(row.get('agent_id')) or default_agent
            st = row.get('matched_topology_snapshot_time_slice', row.get('message_time_slice'))
            ss = row.get('topology_profile_for_agent').get('ss')
            if aid and st is not None and ss is not None:
                profile_seed[aid][int(st)].append(ss)

    first_self = {}
    for fp in sorted((run_dir / 'per_agent').glob('agent_*.jsonl') if (run_dir / 'per_agent').exists() else []):
        default_agent = fp.stem
        for row in _load_jsonl(fp):
            if not row.get('is_self_influence'):
                continue
            aid = _parse_agent_id(row.get('agent_id')) or default_agent
            ps = row.get('published')
            ts = row.get('message_time_slice')
            mi = row.get('message_index')
            if aid and ps is not None and ts is not None and mi is not None:
                key = (int(ts), int(mi))
                if aid not in first_self or key < first_self[aid][0]:
                    first_self[aid] = (key, {'time_slice': int(ts), 'message_index': int(mi), 'stance_score': ps})
    first_self = {k: v[1] for k, v in first_self.items()}

    min_t, max_t = 0, 0
    if msg_count:
        min_t, max_t = min(msg_count), max(msg_count)

    all_agents = set(graph.keys())
    for src, dsts in graph.items():
        all_agents.add(src)
        all_agents.update(dsts)
    all_agents.update(stance_by_agent.keys(), profile_seed.keys(), first_self.keys())
    sorted_agents = sorted(all_agents, key=_numeric_agent_key)

    stance_summary = {a: {t: float(np.mean(vals)) for t, vals in ts.items()} for a, ts in stance_by_agent.items()}
    profile_summary = {a: {t: float(np.mean(vals)) for t, vals in ts.items()} for a, ts in profile_seed.items()}

    return {
        'run_name': run_dir.name,
        'graph': graph,
        'agent_ids': sorted_agents,
        'stance_by_agent_slice': stance_summary,
        'messages_per_slice': dict(msg_count),
        'profile_seed_by_slice': profile_summary,
        'first_self_posts': first_self,
        'message_events': message_events,
        'min_time_slice': int(min_t),
        'max_time_slice': int(max_t),
    }


def build_global_init_map(run_data, global_agent_ids):
    out = {}
    for agent in global_agent_ids:
        vals = []
        for d in run_data.values():
            mapv = d['profile_seed_by_slice'].get(agent, {})
            if mapv:
                t0 = min(mapv)
                v = mapv.get(t0)
                if v is not None and np.isfinite(v):
                    vals.append(float(v))
        if vals:
            out[agent] = float(np.mean(vals))
    return out


def build_run_trajectory(
    data,
    global_agent_ids,
    target_agent_fraction,
    constrain_messages=150,
    return_post_mask=False,
):
    agent_index = {a: i for i, a in enumerate(global_agent_ids)}
    slice_ms = compute_required_time_slice_ms(len(global_agent_ids), target_agent_fraction=target_agent_fraction)
    # print (slice_ms)
    events = data.get('message_events', [])
    if constrain_messages is not None:
        if not isinstance(constrain_messages, int):
            raise TypeError("constrain_messages must be an integer")
        if constrain_messages < 1:
            raise ValueError("constrain_messages must be >= 1")
        events = sorted(events, key=lambda x: x[0])[:constrain_messages]

    rebucketed_slice_obs, last_slice = _bucket_events_to_slices(events, slice_ms)
    T = int(last_slice)
    traj = np.full((T + 1, len(global_agent_ids)), np.nan, dtype=float)
    post_mask = np.zeros((T + 1, len(global_agent_ids)), dtype=bool)
    x0 = np.full((len(global_agent_ids),), np.nan, dtype=float)

    slice0_obs = rebucketed_slice_obs.get(0, {})

    for a in global_agent_ids:
        i = agent_index[a]
        if a in slice0_obs:
            x0[i] = float(slice0_obs[a])
            post_mask[0, i] = True
            continue

        seed = None
        profile = data['profile_seed_by_slice'].get(a, {})
        t0 = min(profile)
        seed = profile.get(t0)
        x0[i] = float(seed)

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
    return traj


def build_neighbors_index(data, global_agent_ids):
    index = {a: i for i, a in enumerate(global_agent_ids)}
    pred = {a: [] for a in global_agent_ids}
    for s, dsts in data['graph'].items():
        for d in dsts:
            if s in index and d in pred:
                pred[d].append(s)

    out = {}
    for a in global_agent_ids:
        ns = sorted({index[n] for n in pred[a] if n in index})
        out[index[a]] = ns or [index[a]]
    return out


def sanitize_array(values):
    return np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def build_dataset_from_run(run):
    x_rows = []
    y_rows = []
    for t in range(len(run) - 1):
        x_rows.append(run[t])
        y_rows.append(run[t + 1])
    x = np.asarray(x_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)
    return x, y

def build_x0_from_agent_inits(agent_inits, n):
    x0 = np.full((n,), np.nan, dtype=float)
    for aid, val in agent_inits.items():
        idx = int(aid.split("_", 1)[1]) - 1
        x0[idx] = float(val)
    if np.isnan(x0).any():
        missing = np.where(np.isnan(x0))[0].tolist()
        raise ValueError(f"missing init values for indices: {missing}")
    return x0


def build_row_normalized_adjacency(neighbors, n):
    a = np.zeros((n, n), dtype=float)
    for i in range(n):
        row_neighbors = list(neighbors[i])
        if len(row_neighbors) == 0:
            a[i, i] = 1.0
            continue
        row_neighbors = [j for j in row_neighbors if 0 <= j < n]
        if len(row_neighbors) == 0:
            a[i, i] = 1.0
            continue
        a[i, row_neighbors] = 1.0 / len(row_neighbors)
    return a


def build_expected_message_matrix(neighbors, n, poisson_mean: float = FIXED_MEAN_MSGS_PER_SLICE):
    # expected posts per source (uniform)
    expected_per_source = float(poisson_mean) / float(max(1, int(n)))

    # build out-degree counts for each source j
    out_deg = np.zeros((n,), dtype=int)
    for recv, srcs in neighbors.items():
        for j in srcs:
            if 0 <= j < n:
                out_deg[j] += 1

    # ensure self-loop counted if agent has no out-neighbors (neighbors mapping may omit)
    for j in range(n):
        if out_deg[j] == 0:
            out_deg[j] = 1

    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        srcs = [j for j in neighbors.get(i, []) if 0 <= j < n]
        if not srcs:
            continue
        for j in srcs:
            A[i, j] = expected_per_source / float(out_deg[j])

    return A


def _gamma_to_theta(gamma: float) -> float:
    return float(np.log(max(float(gamma), 1e-12)))


def _theta_to_gamma(theta: float) -> float:
    return float(np.exp(float(theta)))


def _make_homophily_step(abar: Array) -> Callable[[Array, float], Array]:
    def _homophily_step(x_t: Array, gamma: float) -> Array:
        x_t = sanitize_array(x_t).ravel()
        diff = np.abs(x_t[:, None] - x_t[None, :])
        raw = abar * np.exp(-gamma * diff)
        row_sums = raw.sum(axis=1, keepdims=True)
        w_t = np.zeros_like(raw, dtype=float)
        valid = row_sums[:, 0] > 0
        w_t[valid] = raw[valid] / row_sums[valid]
        return w_t @ x_t

    return _homophily_step


def _pooled_blocks(run_traj_map: Dict[str, Array]) -> Tuple[Array, Array]:
    run_names = sorted(run_traj_map.keys())
    x_blocks, y_blocks = [], []

    for run_name in run_names:
        traj = np.asarray(run_traj_map[run_name], dtype=float)
        x, y = build_dataset_from_run(traj)
        x_blocks.append(x)
        y_blocks.append(y)

    return np.vstack(x_blocks), np.vstack(y_blocks)


def build_gamma_line_search_grid(
    gamma0: float,
    local_decades: float = 1.0,
    num_local_points: int = 160,
) -> Array:
    base = max(abs(float(gamma0)), 1e-6)
    local_count = max(int(num_local_points), 5)
    span = 10.0 ** max(float(local_decades), 0.5)

    lo = max(base / span, 1e-8)
    hi = max(base * span, lo * 1.0001)
    local = np.geomspace(lo, hi, num=local_count)

    anchors = np.asarray(
        [0.0, base * 0.5, base * 0.8, base, base * 1.25, base * 1.5],
        dtype=float,
    )

    gamma_grid = np.unique(np.concatenate([anchors, local]))
    gamma_grid = gamma_grid[gamma_grid >= 0.0]
    return np.sort(gamma_grid)


def expand_search_region(best_gamma: float, expansion_factor: float = 1.5, points_per_side: int = 80) -> Array:
    best_gamma = max(float(best_gamma), 1e-8)
    expansion_factor = max(float(expansion_factor), 1.1)
    point_count = max(int(points_per_side), 2) * 2
    lower = best_gamma / expansion_factor
    upper = best_gamma * expansion_factor
    return np.geomspace(lower, upper, num=point_count)


def golden_section_search(
    objective: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> float:
    left = max(float(min(a, b)), 1e-12)
    right = max(float(max(a, b)), left * 1.0001)

    phi = (1.0 + np.sqrt(5.0)) / 2.0
    invphi = 1.0 / phi

    c = right - (right - left) * invphi
    d = left + (right - left) * invphi
    fc = float(objective(c))
    fd = float(objective(d))

    for _ in range(int(max_iter)):
        if abs(right - left) < tol:
            break

        if fc < fd:
            right = d
            d = c
            fd = fc
            c = right - (right - left) * invphi
            fc = float(objective(c))
        else:
            left = c
            c = d
            fc = fd
            d = left + (right - left) * invphi
            fd = float(objective(d))

    return float((left + right) / 2.0)


def _refine_gamma_search(objective: Callable[[float], float], gamma0: float) -> Tuple[float, Array, Array]:
    coarse_grid = build_gamma_line_search_grid(gamma0)
    coarse_thetas = np.asarray([_gamma_to_theta(gamma) for gamma in coarse_grid], dtype=float)
    coarse_losses = np.asarray([float(objective(float(gamma))) for gamma in coarse_grid], dtype=float)
    coarse_best_idx = int(np.argmin(coarse_losses))
    coarse_best_theta = float(coarse_thetas[coarse_best_idx])

    refined_grid = expand_search_region(_theta_to_gamma(coarse_best_theta))
    refined_thetas = np.asarray([_gamma_to_theta(gamma) for gamma in refined_grid], dtype=float)
    refined_losses = np.asarray([float(objective(float(gamma))) for gamma in refined_grid], dtype=float)
    refined_best_idx = int(np.argmin(refined_losses))

    if len(refined_grid) >= 2:
        left_idx = max(refined_best_idx - 1, 0)
        right_idx = min(refined_best_idx + 1, len(refined_grid) - 1)
        left_theta = float(refined_thetas[left_idx])
        right_theta = float(refined_thetas[right_idx])
        if right_theta > left_theta:
            best_theta = golden_section_search(
                lambda theta: objective(_theta_to_gamma(theta)),
                left_theta,
                right_theta,
            )
            best_gamma = _theta_to_gamma(best_theta)
        else:
            best_gamma = float(refined_grid[refined_best_idx])
    else:
        best_gamma = float(refined_grid[refined_best_idx])

    return best_gamma, coarse_grid, refined_grid
