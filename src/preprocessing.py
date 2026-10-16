
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = RAW_DATA_DIR / "KDDTrain+.txt"
TEST_FILE = RAW_DATA_DIR / "KDDTest+.txt"

# NSL-KDD has 41 input features, followed by the attack label and difficulty.
FEATURE_COUNT = 41
ATTACK_LABEL = "label"
DIFFICULTY_LEVEL = "difficulty"

# Feature names follow the NSL-KDD dataset specification.
FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    ATTACK_LABEL, DIFFICULTY_LEVEL,
]


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load an NSL-KDD TXT file and assign column names."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_csv(file_path, header=None)

    if df.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} columns, got {df.shape[1]}."
        )

    df.columns = FEATURE_NAMES
    df[ATTACK_LABEL] = df[ATTACK_LABEL].astype(str).str.strip()

    return df


def prepare_datasets():
    """Load train/test data and save clean, labelled CSV files."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(TRAIN_FILE)
    test_df = load_dataset(TEST_FILE)

    # Binary classification: normal versus attack.
    for df in (train_df, test_df):
        df["target"] = (
            df[ATTACK_LABEL].str.lower() != "normal"
        ).astype(int)

    train_output = PROCESSED_DATA_DIR / "train_processed.csv"
    test_output = PROCESSED_DATA_DIR / "test_processed.csv"

    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows: {len(test_df):,}")
    print(f"Training labels:\n{train_df['target'].value_counts().sort_index()}")
    print(f"Testing labels:\n{test_df['target'].value_counts().sort_index()}")
    print(f"\nSaved: {train_output}")
    print(f"Saved: {test_output}")


if __name__ == "__main__":
    prepare_datasets()
