# MLP‑PPO‑2PLY‑MULTI

An implementation of a sigmoid‑based neural network for Backgammon that uses Tesauro‑style board encoding. The model is a simple MLP that consumes a 198‑dimensional feature vector per board state (Tesauro’s representation) and outputs a scalar evaluation.

## What this is
- Sigmoid MLP: First hidden layer uses `sigmoid` activation (see `src/agents/policy_network.py`).
- Tesauro encoding: Board states are encoded into 198 features (see `src/backgammon/board/generate_board_tensor.py`).

## Tesauro‑Style 198‑Feature Encoding
This project follows the classic TD‑Gammon/Neurogammon style features:
- 24 points × 4 features × 2 players = 192
  - For each point, features represent checker counts: 1, 2, ≥3 (capped) and an extra linear feature for stacks beyond 3: `(n−3)/2`.
- Bar checkers per player (normalized) = 2
- Borne‑off checkers per player (normalized) = 2
- Current player one‑hot indicator = 2
- Total = 198 features

See implementation in `src/backgammon/board/generate_board_tensor.py` (function `compute_features`).

## Model
- File: `src/agents/policy_network.py`
- Architecture: `Linear(198 → H)` → `sigmoid` → `Linear(H → 1)` producing a scalar state value.

## Quick start
- Install deps: `pip install -r requirements.txt`
