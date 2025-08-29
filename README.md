# MLP‑PPO‑2PLY‑MULTI

An implementation of a sigmoid‑based neural network for Backgammon that uses Tesauro‑style board encoding. The model is a simple MLP that consumes a 198‑dimensional feature vector per board state (Tesauro’s representation) and outputs a scalar evaluation.

## What this is
- Sigmoid MLP: First hidden layer uses `sigmoid` activation (see `src/agents/policy_network.py`).
- Tesauro encoding: Board states are encoded into 198 features (see `src/backgammon/board/generate_board_tensor.py`).

## System Overview
- Entry point: `src/main.py` orchestrates multiprocessing rollouts and periodic training.
- Rollout workers: N CPU processes (`src/multi/worker.py`) play full games and push finished episodes to a shared queue.
- Trainer: A single trainer (`src/agents/trainer.py`) runs on GPU/CPU, consumes a fixed batch of episodes and performs supervised TD(0)-style updates on the value network.
- Parameter sync: `src/multi/parameter_manager.py` holds a shared, versioned state_dict (NumPy arrays) and a temperature schedule. Workers pull updates by version.
- Storage: A ring replay buffer (`src/utils/ring_replay_buffer.py`) buffers recent episodes. Models/metrics optionally save to S3.

## Training Loop (src/main.py)
- Spawns a `multiprocessing.Manager()` with a shared lock, version, and `parameters` dict.
- Starts 7 worker processes (configurable) via `worker_function(i, parameter_manager, experience_queue)`.
- Main process pulls completed `Episode` objects from `ExperienceQueue`, appends to `RingReplayBuffer`.
- When `MIN_EPISODES_TO_TRAIN` (default 200) are buffered, it drains them, moves tensors to the trainer’s device, and calls `Trainer.update(episodes)`.
- After each update the trainer pushes the new `state_dict` back to `ParameterManager` (version++). Workers notice version bumps and refresh their local weights/temperature.
- Periodically saves checkpoints to S3 (every `MODEL_SAVE_FREQUENCY`).

## Trainer (src/agents/trainer.py)
The trainer performs a batched TD(0) regression update over full episodes:
- Batch size: expects exactly `MIN_EPISODES_TO_TRAIN` episodes per update.
- Targets: For each time step t in an episode, computes `target_t = r_t + γ * V(s_{t+1})` (bootstraps using current network), with last step unbootstrapped.
- Loss: Mean squared error between predicted `V(s_t)` and `target_t` over the sequence; gradients are clipped (`GRAD_CLIP_THRESHOLD`).
- Optimizer: Adam with `LEARNING_RATE`; optional decay via `LR_DECAY`/`LR_DECAY_STEPS` hooks.
- Metrics: Aggregates and logs average loss, TD error, gradient norm, predicted value, reward, episode length; also logs win types (regular/gammon/backgammon) and shaped reward counts (close‑out, prime) per player.
- GPU profiling: Uses NVML to print before/after GPU utilization and memory; logs histograms of weights/biases to TensorBoard (local or S3 via `S3Logger`).
- Parameter publish: After stepping the optimizer, sets updated parameters in `ParameterManager` so workers can refresh.

## Workers & Multiprocessing (src/multi/worker.py)
Each worker is an independent CPU process that:
- Holds a local `BackgammonPolicyNetwork` initialized from `ParameterManager` and a decaying exploration temperature.
- Runs `play_episode(env)` in a loop:
  - At each decision, obtains up to `max_legal_moves` resulting board features from the env, concatenates current state + resulting states, evaluates the network once, and uses softmax( V(next_state) / T ) to sample an action.
  - Steps the env, collects `(observation, state_value, reward, next_observation, next_state_value)` into an `Episode`.
  - Handles “no legal moves” by passing the turn (reward 0) and continuing.
- Pushes each finished `Episode` to the `ExperienceQueue` and every 2 episodes checks for a newer parameter version. On version bump it pulls fresh weights and updates its temperature using `ParameterManager.get_temperature()` (linear decay from `INITIAL_TEMPERATURE` → `FINAL_TEMPERATURE` over `MAX_UPDATES`).

Notes on parallelism:
- Within a single backgammon game, steps are sequential. This repo parallelizes across games: many workers each generate whole games independently.
- Workers avoid GPU contention by evaluating on CPU and forwarding small batches (current + legal next states per move) efficiently with a single network call per decision.

## Environment (src/environments/backgammon_env.py)
Gym‑style environment with performance‑oriented internals:
- Observation: 198‑D features for the current player (`ImmutableBoard.get_board_features`).
- Actions: Discrete index into legal move list, truncated/padded to `max_legal_moves` (default 500). Env maintains:
  - `action_mask` (preallocated tensor) marking valid action slots.
  - `legal_board_features` (preallocated [max_legal_moves, 198]) holding resulting state features.
- Move generation: `get_all_possible_moves(player, board, roll_result)` produces unique full moves obeying rules (including doubles, ordering, and “use the maximum number of dice” filtering).
- Feature generation: `generate_all_board_features(board, current_player, legal_moves)` applies each full move immutably and computes features on‑device without Python object churn.
- Rewards:
  - Win: regular = 1.0, gammon = 2.0, backgammon = 2.5.
  - Shaping: one‑time bonuses per player for first close‑out (`is_closed_out`) and first 5‑prime (`made_at_least_five_prime`).
  - Invalid actions: −1.0; no legal move: pass with 0.0.
- Efficiency details:
  - Caches `current_board_features` and updates it only when board/player changes.
  - Incrementally updates `action_mask` ranges instead of resetting the whole tensor.
  - Copies only the populated head of `legal_board_features` and tracks `num_moves`/`previous_num_moves` to avoid unnecessary zeroing.

## Experience Queue (src/multi/experience_queue.py)
Thin wrapper around `multiprocessing.Queue` optimized by design choices:
- Queue whole episodes: Workers call `episode.to_numpy()` before enqueueing so payloads are NumPy arrays (IPC‑friendly), minimizing pickling overhead versus nested tensors.
- Coarse granularity: Pushing complete episodes reduces queue contention vs. per‑step experiences and lets the trainer consume fixed‑size episode batches efficiently.
- CPU scalability: Although game generation is sequential within an episode, spawning multiple worker processes yields near‑linear speedup in episodes/sec up to core/IPC limits.

## Parameter Manager (src/multi/parameter_manager.py)
- Shared state: Stores a versioned `state_dict` in a `Manager().dict()` as NumPy arrays; workers reconstruct PyTorch tensors on read. Updates are guarded by a lock, reads are lock‑free.
- Temperature schedule: Computes exploration temperature from the current version for annealed action sampling in workers.
- IO utilities: Save/load checkpoints locally or from S3.

## Two‑Ply Scoring (src/multi/two_ply.py)
Optional lookahead utilities:
- `compute_weighted_opponent_response` evaluates opponent replies over all dice with roll probabilities, scoring the top few replies per roll via the network.
- `compute_scores_for_boards` combines immediate state values with expected opponent replies: score = α·S(move) − β·E[V(opponent reply)].
- The worker includes commented code showing how to replace simple softmax over V(next) with two‑ply scores when desired.

## Files to Explore
- `src/main.py`: Orchestrates workers, queue, replay buffer, and trainer.
- `src/agents/trainer.py`: Batch update, metrics, parameter publish.
- `src/multi/worker.py`: Episode rollout loop and sampling.
- `src/environments/backgammon_env.py`: Env internals, rewards, and performance tricks.
- `src/multi/experience_queue.py`: Episode‑level queue for multi‑CPU.
- `src/multi/parameter_manager.py`: Versioned weights, temperature, checkpointing.
- `src/backgammon/moves/*`: Complete legal move generation pipeline.

## Running
- Local: `python -m src.main`
  - Adjust hyperparameters in `src/config/configuration.py`.
  - Set `S3_BUCKET_NAME`/prefixes if you want logging/checkpointing to S3.
  - Ensure NVML is available if running on GPU for utilization metrics.

## Performance Tips
- Increase worker count to match available CPU cores.
- If `max_legal_moves` rarely exceeds a few hundred for your setting, keep it tight to reduce tensor copies.
- Consider enabling the two‑ply scorer for stronger play at a compute cost.
