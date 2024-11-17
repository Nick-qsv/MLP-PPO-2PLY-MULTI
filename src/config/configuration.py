# Hyperparameters
HIDDEN_SIZE = 128
VALUE_LOSS_COEF = 0.5
MAX_TIMESTEPS = 300
NUM_EPISODES = 1_000_000
MODEL_SAVE_FREQUENCY = 20_000
MIN_EPISODES_TO_TRAIN = 200
# S3 Configuration
S3_BUCKET_NAME = "bgppomodels"
S3_MODEL_PREFIX = "models/"
S3_LOG_PREFIX = "logs/"


# Hyperparameters
GAMMA = 0.99
LAMBDA = 0.99
LEARNING_RATE = 1e-3
GRAD_CLIP_THRESHOLD = 1.0  # Gradient clipping threshold
LR_DECAY = 0.99  # Learning rate decay
LR_DECAY_STEPS = 100_000  # Number of steps over which to decay learning rate

# Temperature decay parameters
INITIAL_TEMPERATURE = 1.5
FINAL_TEMPERATURE = 0.5
MAX_UPDATES = 3000
