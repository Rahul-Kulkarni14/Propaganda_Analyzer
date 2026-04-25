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
EXPLANATION_KEYWORDS = {
    "Appeal_to_Authority": [
        "expert", "experts", "authority", "official", "scientist", "doctor",
        "research", "study", "proven", "according to"
    ],
    "Repetition": [
        "again", "again and again", "repeat", "repeatedly", "always", "never"
    ],
    "Doubt": [
        "maybe", "perhaps", "uncertain", "question", "doubt", "allegedly",
        "supposedly", "unverified", "rumor"
    ],
    "Name_Calling": [
        "traitor", "corrupt", "criminal", "liar", "enemy", "fool",
        "coward", "radical", "extremist"
    ],
    "Appeal_to_Fear": [
        "danger", "threat", "fear", "disaster", "crisis", "attack",
        "destroy", "collapse", "unsafe", "risk"
    ],
    "Exaggeration": [
        "always", "never", "everyone", "nobody", "completely", "totally",
        "best", "worst", "massive", "huge", "unbelievable"
    ],
    "Loaded_Language": [
        "evil", "brave", "hero", "traitor", "shameful", "dangerous",
        "disgusting", "glorious", "terrible", "innocent"
    ],
    "Bandwagon": [
        "everyone", "everybody", "majority", "millions", "join", "supporters",
        "popular", "all of us", "people are saying"
    ],
    "Stereotyping": [
        "all", "always", "never", "these people", "those people", "they are",
        "their kind"
    ],
    "Flag_Waving": [
        "nation", "country", "patriot", "patriotic", "flag", "freedom",
        "homeland", "motherland", "our people"
    ],
    "Causal_Oversimplification": [
        "because of", "only reason", "single cause", "caused by", "blame",
        "responsible for", "this is why"
    ],
    "Appeal_to_Pity": [
        "suffer", "suffering", "poor", "helpless", "victim", "pain",
        "struggle", "hardship", "sympathy"
    ],
    "Red_Herring": [
        "instead", "what about", "look at", "ignore", "distract", "other issue",
        "why talk about"
    ],
    "Card_Stacking": [
        "only", "just", "clearly", "undeniable", "facts show", "no evidence against",
        "without mentioning"
    ],
    "Testimonial": [
        "I believe", "I saw", "my experience", "people say", "witness",
        "testimonial", "endorsed", "recommended"
    ]
}


TECHNIQUE_EXPLANATIONS = {
    "Appeal_to_Authority": "This fragment may rely on authority figures or expert claims to persuade the audience.",
    "Repetition": "This fragment may use repeated wording or repeated ideas to reinforce a message.",
    "Doubt": "This fragment may create uncertainty or suspicion without strong evidence.",
    "Name_Calling": "This fragment may use negative labels or insults to attack a person or group.",
    "Appeal_to_Fear": "This fragment may use fear, danger, or threat-based language to influence the audience.",
    "Exaggeration": "This fragment may overstate or amplify claims beyond a balanced description.",
    "Loaded_Language": "This fragment may use emotionally charged words to influence the reader.",
    "Bandwagon": "This fragment may suggest that many people support something, encouraging others to follow.",
    "Stereotyping": "This fragment may generalize about a group of people.",
    "Flag_Waving": "This fragment may appeal to patriotism, national identity, or loyalty.",
    "Causal_Oversimplification": "This fragment may present a complex issue as having one simple cause.",
    "Appeal_to_Pity": "This fragment may use sympathy or suffering to influence the audience.",
    "Red_Herring": "This fragment may shift attention away from the main issue.",
    "Card_Stacking": "This fragment may present selective information while leaving out important context.",
    "Testimonial": "This fragment may rely on personal endorsement or individual experience as persuasion."
}


def get_lightweight_explanation(fragment, technique):
    fragment_lower = fragment.lower()
    keywords = EXPLANATION_KEYWORDS.get(technique, [])

    matched_cues = []
    for keyword in keywords:
        if keyword.lower() in fragment_lower:
            matched_cues.append(keyword)

    return {
        "matched_cues": matched_cues[:8],
        "explanation": TECHNIQUE_EXPLANATIONS.get(
            technique,
            "This fragment contains linguistic cues that may be associated with the predicted technique."
        )
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

def analyze_speech_with_confidence(speech_text):
    fragments = split_and_merge_speech(speech_text)
    detected_fragments = []

    for frag in fragments:
        inputs = tokenizer(frag, truncation=True, padding=True, max_length=128, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        binary_model.eval()
        with torch.no_grad():
            binary_outputs = binary_model(input_ids, attention_mask=attention_mask)
            binary_probs = torch.softmax(binary_outputs.logits, dim=1)
            binary_pred = torch.argmax(binary_probs, dim=1).item()
            manipulation_confidence = binary_probs[0][binary_pred].item()

        if binary_pred == 0:
            continue

        multi_model.eval()
        with torch.no_grad():
            multi_outputs = multi_model(input_ids, attention_mask=attention_mask)
            multi_probs = torch.softmax(multi_outputs.logits, dim=1)[0]
            sorted_preds = torch.argsort(multi_probs, descending=True)

            primary_label = sorted_preds[0].item()
            primary_name = technique_map.get(primary_label, f"Technique_{primary_label}")

            if primary_name == "Name_Calling" and len(sorted_preds) > 1:
                selected_label = sorted_preds[1].item()
            else:
                selected_label = primary_label

            technique_name = technique_map.get(selected_label, f"Technique_{selected_label}")
            technique_confidence = multi_probs[selected_label].item()
        
        xai_result = get_lightweight_explanation(frag, technique_name)

        detected_fragments.append({
            "fragment": frag,
            "technique": technique_name,
            "manipulation_confidence": round(manipulation_confidence * 100, 2),
            "technique_confidence": round(technique_confidence * 100, 2),
            "matched_cues": xai_result["matched_cues"],
            "explanation": xai_result["explanation"]
        })

    return detected_fragments

# =============================
# Section 5: Example Usage
# =============================
if __name__ == "__main__":
    print("\n=== Speech Analyzer ===\n")
    sample_speech = input("Enter the speech text to analyze:\n")
    print("\n")
    analyze_speech(sample_speech)
