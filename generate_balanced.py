import os
import pandas as pd
import random

# Load the processed manipulative dataset
df = pd.read_csv("processed/train_processed.csv")

# Group manipulative spans by article
manipulative_texts = df.groupby("article_id")["text_fragment"].apply(list).to_dict()

# Read all articles to extract neutral text
ARTICLES_DIR = "datasets/train-articles"
neutral_data = []

for article_file in os.listdir(ARTICLES_DIR):
    if not article_file.endswith(".txt"):
        continue

    article_id = article_file.replace(".txt", "")
    with open(os.path.join(ARTICLES_DIR, article_file), "r", encoding="utf-8") as f:
        text = f.read()

    # Get manipulative spans for this article
    manip_spans = manipulative_texts.get(article_id, [])
    for span in manip_spans:
        text = text.replace(span, "")

    # Randomly sample neutral fragments
    neutral_fragments = [sent.strip() for sent in text.split(".") if len(sent.split()) > 5]
    sampled = random.sample(neutral_fragments, min(len(neutral_fragments), 5))
    for frag in sampled:
        neutral_data.append({"article_id": article_id, "text_fragment": frag, "technique": "Neutral"})

# Combine manipulative + neutral
neutral_df = pd.DataFrame(neutral_data)
combined = pd.concat([df, neutral_df], ignore_index=True)

# Add binary label
combined["label"] = combined["technique"].apply(lambda x: 0 if x == "Neutral" else 1)

print(f"✅ Final combined dataset: {len(combined)} samples")
print(combined["label"].value_counts())

os.makedirs("processed", exist_ok=True)
combined.to_csv("processed/final_dataset.csv", index=False, encoding="utf-8")
print("📁 Saved combined dataset at: processed/final_dataset.csv")
