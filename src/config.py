MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_LENGTH = 256
NUM_LABELS = 3
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 1e-3

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
