# /src/data/tetris_generator.py

import numpy as np
import random
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Dict, Tuple

# --- Game Constants ---
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BLOCK_SIZE = 32
IMAGE_SIZE = (BOARD_WIDTH * BLOCK_SIZE, BOARD_HEIGHT * BLOCK_SIZE)

SHAPES = [
    [[1, 1], [1, 1]],        # Square (0)
    [[1, 1, 1, 1]],          # Line (1)
    [[0, 1, 0], [1, 1, 1]],  # T-shape (2)
    [[1, 1, 0], [0, 1, 1]],  # S-shape (3)
    [[0, 1, 1], [1, 1, 0]]   # Z-shape (4)
]
NUM_SHAPES = len(SHAPES)

# --- Game Logic Functions ---

def create_board():
    """Initializes an empty Tetris board."""
    return np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)

def draw_board(board: np.ndarray) -> Image.Image:
    """
    Draws a visual representation of the board as a single-channel (grayscale) image.
    This is more efficient and consistent with the binary nature of the board state.
    """
    img = Image.new('L', IMAGE_SIZE, color='black')
    draw = ImageDraw.Draw(img)
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            if board[y, x] == 1:
                draw.rectangle([x * BLOCK_SIZE, y * BLOCK_SIZE,
                                (x + 1) * BLOCK_SIZE, (y + 1) * BLOCK_SIZE],
                                fill='white')
    return img

def place_piece(board: np.ndarray, piece: list, x_pos: int) -> np.ndarray | None:
    """Places a piece on the board at the lowest possible y-position."""
    new_board = board.copy()
    piece_height, piece_width = len(piece), len(piece[0])
    x_pos = min(x_pos, BOARD_WIDTH - piece_width)
    x_pos = max(x_pos, 0)

    y_pos = 0
    while y_pos + piece_height <= BOARD_HEIGHT:
        collides = False
        for y in range(piece_height):
            for x in range(piece_width):
                if piece[y][x] == 1 and new_board[y_pos + y, x_pos + x] == 1:
                    collides = True
                    break
            if collides:
                break
        if collides:
            break
        y_pos += 1

    y_pos -= 1

    if y_pos + piece_height > BOARD_HEIGHT or y_pos < 0:
        return None

    for y in range(piece_height):
        for x in range(piece_width):
            if piece[y][x] == 1:
                new_board[y_pos + y, x_pos + x] = 1

    return new_board

# --- Conceptual Feature Extraction ---

def evaluate_conceptual_features(board: np.ndarray) -> np.ndarray:
    """Calculates conceptual features of a board state."""
    lines_cleared = np.sum(np.all(board != 0, axis=1))

    gaps = 0
    for col in range(board.shape[1]):
        filled_found = False
        for row in range(board.shape[0]):
            if board[row, col] != 0:
                filled_found = True
            elif filled_found:
                gaps += 1

    max_height = 0
    for col in range(board.shape[1]):
        for row in range(board.shape[0]):
            if board[row, col] != 0:
                col_height = BOARD_HEIGHT - row
                max_height = max(max_height, col_height)
                break

    board_fullness = np.sum(board) / (BOARD_WIDTH * BOARD_HEIGHT)

    return np.array([lines_cleared, gaps, max_height, board_fullness], dtype=np.float32)

# --- Data Generation and Dataset Class ---

def generate_pair_with_next_piece(max_retries=2500) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray] | None:
    """Generates a before/after board pair and a next piece for training."""
    for _ in range(max_retries):
        board_before = create_board()
        # Create a more complex initial board state to make the problem more interesting
        for _ in range(5):
            initial_piece_idx = random.randint(0, NUM_SHAPES - 1)
            initial_x_pos = random.randint(0, BOARD_WIDTH - len(SHAPES[initial_piece_idx][0]))
            board_before = place_piece(board_before, SHAPES[initial_piece_idx], initial_x_pos)
            if board_before is None:
                board_before = create_board() # Restart if placing fails

        piece_idx = random.randint(0, NUM_SHAPES - 1)
        next_piece_idx = random.randint(0, NUM_SHAPES - 1)
        x_pos = random.randint(0, BOARD_WIDTH - len(SHAPES[piece_idx][0]))
        board_after = place_piece(board_before, SHAPES[piece_idx], x_pos)

        if board_after is not None:
            conceptual_features_before = evaluate_conceptual_features(board_before)
            return board_before, board_after, next_piece_idx, conceptual_features_before

    return None

class MemoryEfficientDataset(Dataset):
    """A dataset class that generates Tetris game data on the fly."""
    def __init__(self, size=250000):
        self.size = size
        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.size

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data_pair = None
        while data_pair is None:
            data_pair = generate_pair_with_next_piece()

        board_before, board_after, next_piece_idx, conceptual_features = data_pair
        
        # Convert numpy boards to PIL Images, then to PyTorch Tensors
        img_before = draw_board(board_before)
        img_after = draw_board(board_after)
        
        tensor_before = self.transform(img_before)
        board_target = self.transform(img_after)
        
        # Ensure tensors are binary (0 or 1) for BCE loss
        tensor_before_binary = (tensor_before > 0).float()
        board_target_binary = (board_target > 0).float()

        next_piece_tensor = torch.tensor(next_piece_idx, dtype=torch.long)
        conceptual_tensor = torch.tensor(conceptual_features, dtype=torch.float32)

        return tensor_before_binary, board_target_binary, next_piece_tensor, conceptual_tensor
