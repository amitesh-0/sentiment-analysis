
# 1. Import required libraries
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
from scipy.sparse import save_npz

# 2. Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 3. Define the input and output paths
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "IMDB_cleaned.csv"
MODEL_DIR = PROJECT_ROOT / "models"
FEATURE_DIR = PROJECT_ROOT / "data" / "processed"

# 4. Create directories if they do not exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# 5. Load the cleaned dataset
df = pd.read_csv(DATA_PATH)

# 6. Remove rows with missing reviews or sentiments
df = df.dropna(subset=["cleaned_review", "sentiment"])

# 7. Convert sentiment labels into numbers
df["sentiment"] = df["sentiment"].str.lower().map({
    "positive": 1,
    "negative": 0
})

# 8. Remove rows with unexpected sentiment labels
df = df.dropna(subset=["sentiment"])
df["sentiment"] = df["sentiment"].astype(int)

# 9. Separate input text (X) and target labels (y)
X = df["cleaned_review"].astype(str)
y = df["sentiment"]

# 10. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 11. Create the TF-IDF vectorizer
tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

# 12. Learn vocabulary and transform training reviews
X_train_tfidf = tfidf.fit_transform(X_train)

# 13. Transform testing reviews using the same vocabulary
X_test_tfidf = tfidf.transform(X_test)

# 14. Save the trained vectorizer
joblib.dump(
    tfidf,
    MODEL_DIR / "tfidf_vectorizer.joblib"
)

# 15. Save sparse numerical feature matrices
save_npz(
    FEATURE_DIR / "X_train_tfidf.npz",
    X_train_tfidf
)

save_npz(
    FEATURE_DIR / "X_test_tfidf.npz",
    X_test_tfidf
)

# 16. Save training and testing labels
y_train.to_csv(FEATURE_DIR / "y_train.csv", index=False)
y_test.to_csv(FEATURE_DIR / "y_test.csv", index=False)

# 17. Display the results
print("Text feature preparation completed!")
print("Training reviews:", X_train_tfidf.shape[0])
print("Testing reviews:", X_test_tfidf.shape[0])
print("Number of features:", X_train_tfidf.shape[1])
print("Training matrix shape:", X_train_tfidf.shape)
print("Testing matrix shape:", X_test_tfidf.shape)
print("Label values:", sorted(y.unique()))
