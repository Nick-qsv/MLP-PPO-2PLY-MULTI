# Hyperparameters
HIDDEN_SIZE = 128
VALUE_LOSS_COEF = 0.5
MAX_TIMESTEPS = 300
NUM_EPISODES = 1_000_000
MODEL_SAVE_FREQUENCY = 100_000
MIN_EPISODES_TO_TRAIN = 1_000
# S3 Configuration
S3_BUCKET_NAME = "bgppomodels"
S3_MODEL_PREFIX = "models/"
S3_LOG_PREFIX = "logs/"


# Hyperparameters
GAMMA = 0.99
LAMBDA = 0.90
LEARNING_RATE = 1e-3
ENTROPY_COEF_START = 0.15
ENTROPY_COEF_END = 0.03
ENTROPY_ANNEAL_EPISODES = (
    500_000  # Number of episodes over which to anneal entropy coefficient
)
EPSILON = 0.2  # PPO clipping parameter
K_EPOCHS = 4  # Number of epochs to update policy
BATCH_SIZE = 64  # Batch size for mini-batch updates

USE_SIGMOID = False  # Use sigmoid for False, reLU for True
