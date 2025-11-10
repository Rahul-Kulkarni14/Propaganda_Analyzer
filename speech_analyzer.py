# =============================
# speech_analyzer.py - Speech Propaganda/Manipulation Analyzer
# =============================
import os
# =============================
# Suppress TensorFlow / Backend Warnings
# =============================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN info messages

import warnings
warnings.filterwarnings("ignore")  # Suppress general Python warnings 

# Silence NLTK info logging
import logging
logging.getLogger('nltk').setLevel(logging.ERROR)
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification



# =============================
# NLTK Setup
# =============================
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download standard 'punkt' if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

punkt_params = PunktParameters()
sentence_tokenizer = PunktSentenceTokenizer(punkt_params)

# =============================
# Section 0: Load Models & Tokenizer
# =============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

binary_model_dir = "transformer_binary_model"
multi_model_dir = "transformer_multiclass_model"

binary_model = DistilBertForSequenceClassification.from_pretrained(binary_model_dir, num_labels=2).to(device)
multi_model = DistilBertForSequenceClassification.from_pretrained(multi_model_dir, num_labels=15).to(device)
tokenizer = DistilBertTokenizerFast.from_pretrained(binary_model_dir)

# =============================
# Section 1: Technique Mapping (Multiclass)
# =============================
technique_map = {
    0: "Appeal_to_Authority",
    1: "Repetition",
    2: "Doubt",
    3: "Name_Calling",
    4: "Appeal_to_Fear",
    5: "Exaggeration",
    6: "Loaded_Language",
    7: "Bandwagon",
    8: "Stereotyping",
    9: "Flag_Waving",
    10: "Causal_Oversimplification",
    11: "Appeal_to_Pity",
    12: "Red_Herring",
    13: "Card_Stacking",
    14: "Testimonial"
}

# =============================
# Section 2: Split Speech into Sentences & Merge Short Ones
# =============================
def split_and_merge_speech(speech_text, min_words=5, merge_threshold=10):
    sentences = sentence_tokenizer.tokenize(speech_text)  # use explicit tokenizer
    sentences = [s for s in sentences if len(s.split()) >= min_words]

    merged_sentences = []
    i = 0
    while i < len(sentences):
        frag = sentences[i]
        while len(frag.split()) < merge_threshold and i+1 < len(sentences):
            frag += " " + sentences[i+1]
            i += 1
        merged_sentences.append(frag)
        i += 1
    return merged_sentences

# =============================
# Section 3: Analyze Speech Function
# =============================
def analyze_speech(speech_text):
    fragments = split_and_merge_speech(speech_text)
    detected_fragments = []

    for frag in fragments:
        inputs = tokenizer(frag, truncation=True, padding=True, max_length=128, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Binary prediction
        binary_model.eval()
        with torch.no_grad():
            outputs = binary_model(input_ids, attention_mask=attention_mask)
            binary_pred = torch.argmax(outputs.logits, dim=1).item()

        if binary_pred == 0:
            continue  # no manipulation detected

        # Multiclass prediction
        multi_model.eval()
        with torch.no_grad():
            outputs_mc = multi_model(input_ids, attention_mask=attention_mask)
            logits = outputs_mc.logits[0]  # Get logits for this fragment

            # Sort predictions by confidence (highest first)
            sorted_preds = torch.argsort(logits, descending=True)

            # Get the most confident label
            primary_label = sorted_preds[0].item()
            primary_name = technique_map.get(primary_label, f"Technique_{primary_label}")

            # If top label is "Name_Calling", pick the next most confident one
            if primary_name == "Name_Calling" and len(sorted_preds) > 1:
                secondary_label = sorted_preds[1].item()
                technique_name = technique_map.get(secondary_label, f"Technique_{secondary_label}")
            else:
                technique_name = primary_name

        detected_fragments.append((frag, technique_name))

    # =============================
    # Section 4: Print Summary
    # =============================
    if not detected_fragments:
        print("✅ No propaganda/manipulation detected in this speech.")
    else:
        print("⚠️ Propaganda/manipulation detected in the speech!\n")
        print("Detected fragments and techniques:\n")
        for i, (frag, tech) in enumerate(detected_fragments, 1):
            print(f"{i}. [{tech}] {frag}\n")

# =============================
# Section 5: Example Usage
# =============================
if __name__ == "__main__":
    print("\n=== Speech Analyzer ===\n")
    sample_speech = input("Enter the speech text to analyze:\n")
    print("\n")
    analyze_speech(sample_speech)
