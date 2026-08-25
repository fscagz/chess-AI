# chessAI

A self-learning chess engine built around an AlphaZero-style architecture: a residual convolutional neural network with dual policy/value heads, guided by Monte Carlo Tree Search (MCTS), trained through self-play. 

This README summarizes the approach, architecture, and results from the project, and how to run the code.

## How it works

### Board representation

The game state is stored as a [FEN string](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) rather than a `chess.Board` object, so that thousands of self-play states can be kept in memory cheaply. A `chess.Board` is reconstructed from the FEN string only when needed (e.g. to compute legal moves).

Before being fed to the network, a board is encoded as thirteen 8x8 binary matrices: one per piece type per color (12 total), plus one marking all empty squares.

### Move representation

Rather than the naive 4,032-move encoding (every from-square to every to-square), the policy head predicts over a reduced list of ~1,836 moves — only moves a piece could geometrically make from a given square (queen-like + knight moves, plus promotions). This roughly halved model inference time (from ~6-7s to ~3-4s per step).

### Network architecture

`ResNet` (built with PyTorch) takes the 13-channel board encoding through a stack of residual blocks, then splits into two heads:
- **Value head** — outputs a scalar in `[-1, 1]` estimating whether the current player is winning, losing, or drawing.
- **Policy head** — outputs a probability distribution over the move list, used to select the next move.

The two heads are trained jointly with MSE loss (value) and cross-entropy loss (policy), following the same setup used by AlphaZero.

### Monte Carlo Tree Search

Plain minimax/alpha-beta search propagates the neural network's approximation errors up the tree. Instead, this project uses **Alpha MCTS**: each node represents a board state and tracks its visit count, value sum, and prior probability (from the policy head). Search proceeds in three phases — select (via UCB, balancing exploration/exploitation), expand, and backpropagate — repeated for a configurable number of iterations per move. `MCTSParallel` runs many self-play games' searches simultaneously to make better use of compute.

### Training

Two approaches were tried:

1. **Self-play reinforcement learning** (the AlphaZero approach): the model plays against itself, MCTS visit counts become the policy training target, and the game's final outcome becomes the value training target for every state in that game. This is by far the most computationally expensive part of the project — a single game took 25-36 minutes depending on model size, which limited training to a few hundred games.
2. **Supervised ("labeled") training**: the model is instead trained to imitate move evaluations from Stockfish, giving it a faster way to pick up short-term tactics (e.g. castling) without needing millions of self-play games.

### Results

Chess is enormously complex (Shannon estimated ~10^120 possible games), and self-play from scratch as done by AlphaZero requires millions of games and huge compute budgets. This project was limited to a handful of GPUs and roughly a week of training time (~550 self-play games), which was nowhere near enough data for the model to play well — it topped out at an Elo of ~100 on Chess.com (the platform's minimum rating), though the labeled training phase visibly improved short-term tactical understanding.

To validate that the underlying self-play/MCTS pipeline actually works given enough games, the same architecture was retrained from scratch on **Connect Four**, a far less complex game. There, the approach worked convincingly — a model trained on 1,500 self-play games beat every human tester it played against.

## Repository contents

This project is packaged as a single Google Colab notebook, [`chess-AI-gh.ipynb`](chess-AI-gh.ipynb), containing:

- Move-list generation (`generate_moves`, `get_move_matrix`)
- The chess environment/game wrapper (`ChessInterface`) — state transitions, legal move generation, encoding, and terminal/win checks
- The network (`ResNet`, `ResBlock`)
- The search algorithm (`Node`, `MCTSParallel`)
- The training loop (`LearningAlg`, with `SelfPlay` and `train`)
- Demo cells for playing against the trained model, including an SVG board display

Also included is [`model_6.pt`](model_6.pt), a trained checkpoint (`ResNet` weights) saved after training iteration 6, so the demo cells can be run without retraining from scratch.

## Running it

This notebook was written to run in Google Colab (it installs its own dependencies at the top via `pip install torch torchvision chess tqdm svgwrite`). To run it:

1. Open `chess-AI-gh.ipynb` in [Google Colab](https://colab.research.google.com/) (or Jupyter locally with a GPU).
2. Run the setup cells at the top to install dependencies and define the model, MCTS, and game classes.
3. A precomputed move list (`moves.npy`) is expected in the working directory — generate it by running `get_move_matrix()` and saving its output. This defines the action space and must match what a checkpoint was trained against, so regenerate it fresh rather than reusing an old copy from another run.
4. To play against the included checkpoint, point the loader at `model_6.pt` (the demo cells default to `model_4.pt` — update the filename in `initialize_chess_AI()`/the loading cell to `model_6.pt`), making sure the `ResNet(...)` constructor args match what it was trained with (4 residual blocks, 16 hidden channels).
5. Run the demo cells at the bottom to play against the model, either in plain text (UCI move input) or with an SVG board.

Training saves a matching `optimizer_{iteration}.pt` file (Adam optimizer state) alongside each `model_{iteration}.pt`. That file is only needed if you want to resume training from that checkpoint — it isn't required to load the model and play against it, so it doesn't need to be added to the repo unless you plan to continue training from iteration 6.
