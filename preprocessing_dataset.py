import os
import pandas as pd

# -----------------------------
# STEP 1: PATH SETUP
# -----------------------------
DATA_DIR = "datasets"
ARTICLES_DIR = os.path.join(DATA_DIR, "train-articles")
LABELS_DIR = os.path.join(DATA_DIR, "train-labels-task2-technique-classification")

print("Current working directory:", os.getcwd())
print(f"✅ Found folder: {ARTICLES_DIR}, contains {len(os.listdir(ARTICLES_DIR))} files")
print(f"✅ Found folder: {LABELS_DIR}, contains {len(os.listdir(LABELS_DIR))} files")

# -----------------------------
# STEP 2: LOAD LABEL FILES
# -----------------------------
def load_labels_from_directory(directory):
    label_data = []
    skipped_lines = 0
    total_lines = 0

    for filename in os.listdir(directory):
        if not filename.endswith(".labels"):
            continue

        file_path = os.path.join(directory, filename)
        article_id = filename.split(".")[0]

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                # Expected 4 parts: article_id, technique, start, end
                if len(parts) != 4:
                    skipped_lines += 1
                    continue

                _, technique, start, end = parts
                try:
                    start = int(start)
                    end = int(end)
                    label_data.append((article_id, technique, start, end))
                except ValueError:
                    skipped_lines += 1
                    continue

    print(f"\n✅ Finished parsing {directory}")
    print(f"   ➤ Total lines processed: {total_lines}")
    print(f"   ➤ Valid entries: {len(label_data)}")
    print(f"   ➤ Skipped malformed lines: {skipped_lines}\n")
    return label_data


# -----------------------------
# STEP 3: LOAD ARTICLES + MATCH LABELS
# -----------------------------
label_entries = load_labels_from_directory(LABELS_DIR)
data = []

for article_file in os.listdir(ARTICLES_DIR):
    if not article_file.endswith(".txt"):
        continue

    article_id = article_file.replace(".txt", "")
    article_path = os.path.join(ARTICLES_DIR, article_file)

    with open(article_path, "r", encoding="utf-8") as f:
        article_text = f.read()

    # Find all label fragments for this article
    labels = [lbl for lbl in label_entries if lbl[0] == article_id]
    for (_, technique, start, end) in labels:
        if 0 <= start < len(article_text) and 0 < end <= len(article_text):
            fragment = article_text[start:end].strip()
            if fragment:
                data.append({
                    "article_id": article_id,
                    "text_fragment": fragment,
                    "technique": technique
                })

# -----------------------------
# STEP 4: CREATE DATAFRAME
# -----------------------------
df = pd.DataFrame(data)
print("\n✅ Dataset successfully assembled!")
print(f"Total valid labeled fragments: {len(df)}\n")
print(df.head(10))

# -----------------------------
# STEP 5: SAVE FOR FUTURE USE
# -----------------------------
os.makedirs("processed", exist_ok=True)
csv_path = os.path.join("processed", "train_processed.csv")
df.to_csv(csv_path, index=False, encoding="utf-8")
print(f"📁 Saved cleaned dataset at: {csv_path}")
