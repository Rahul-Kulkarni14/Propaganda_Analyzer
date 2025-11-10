
# Propaganda Analyzer – Speech Manipulation Detection System

An intelligent, modular NLP system that detects propaganda and manipulative techniques in speech. The system uses DistilBERT-based models to classify text fragments as either manipulated or genuine, and further identifies the specific manipulative techniques applied in each fragment.

---

## Overview

Propaganda Analyzer is a hybrid text analysis framework that detects manipulative大发语言 in speeches and textual content using:

* **Binary classification model**: Identifies whether a fragment contains manipulation.
* **Multiclass classification model**: Identifies the type of manipulative technique, such as Bandwagon, Appeal to Fear, Name Calling, and more.
* **Sentence-based fragmentation**: Splits speeches intelligently into manageable fragments to improve detection accuracy.

The system is designed to analyze speeches of arbitrary length and highlight specific propaganda techniques for research, media analysis, or social studies.

---

## Features

* **Speech fragmentation**: Automatically splits speeches into sentences or word-based fragments.

* **Binary classification**: Detects presence of manipulation.

* **Multiclass classification**: Detects 15 different manipulative techniques:

  * Appeal_to_Authority, Repetition, Doubt, Name_Calling, Appeal_to_Fear, Exaggeration, Loaded_Language, Bandwagon, Stereotyping, Flag_Waving, Causal_Oversimplification, Appeal_to_Pity, Red_Herring, Card_Stacking, Testimonial

* **GPU support**: Uses PyTorch with CUDA for faster inference if available.

* **Detailed output**: Highlights which fragment contains which manipulation technique.

* **Modular codebase**: Separate scripts for model training (`main.py`) and inference (`speech_analyzer.py`).

---

## Technologies Used

| Component             | Framework / Library      | Purpose                                   |
| --------------------- | ------------------------ | ----------------------------------------- |
| Binary Classifier     | PyTorch / Transformers   | Detects presence of manipulation          |
| Multiclass Classifier | PyTorch / Transformers   | Identifies specific propaganda techniques |
| Tokenization          | HuggingFace Transformers | Preprocessing & embedding                 |
| Sentence Splitting    | NLTK                     | Fragmenting long speeches for analysis    |

---

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/Rahul-Kulkarni14/Propaganda_Analyzer.git
cd Propaganda_Analyzer
```

2. **Install Git LFS** (for large model files):

```bash
git lfs install
git lfs pull
```

> **Note:** This ensures that model files (`.safetensors`) over 100 MB are properly downloaded. All `.safetensors` files in this repo are tracked with Git LFS.

3. **Install required Python packages**:

```bash
pip install -r requirements.txt
```

---

## Running the System

### 1. Analyze a speech (CLI)

Run the main analysis script:

```bash
python speech_analyzer.py
```

* Enter a speech or textual input when prompted.
* The system will output detected propaganda fragments with corresponding techniques.

### 2. Train / Evaluate Models

Run your training and evaluation pipeline:

```bash
python main.py
```

* Outputs include performance metrics such as Accuracy, Precision, Recall, F1-Score for both binary and multiclass models.

### 3. View Model Performance

```bash
python performance.py
```

---

**Launch the full web app** with a beautiful, interactive UI:

```bash
python app.py
```

Then open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Web App Features

* **Real-time analysis**: Paste any speech and see results instantly
* **Clean, modern UI**: Built with HTML/CSS/JS (no extra frontend needed)
* **Two powerful tools**:
  - **Analyze Speech** → Highlights propaganda fragments with technique names
  - **Model Performance** → Full classification report (cached after first load)
* **CORS enabled** → Ready for deployment or frontend separation
* **First load**: 15–50 seconds (model loading)  
  **Subsequent use**: Lightning fast

> Perfect for demos, presentations, or sharing with professors!

---

## Model Insights

* **Binary Model**: Detects whether a speech fragment contains manipulation or not.
* **Multiclass Model**: Detects 15 types of manipulative techniques.
* **Fragment-based approach**: Shorter sentence-based fragments improve multiclass detection accuracy compared to large blocks of text.

---

## Evaluation Metrics

The system provides:

* **Accuracy, Precision, Recall, F1-Score**
* **Confusion Matrix** per model
* **Fragment-level analysis** for detailed technique detection

---

## Applications

Propaganda Analyzer is useful for:

* Media analysis and journalism
* Research in social sciences and political studies
* Fact-checking speeches, interviews, or debates
* Identifying manipulative language in public communications

---

## References

* HuggingFace Transformers – [https://huggingface.co/](https://huggingface.co/)
* PyTorch – [https://pytorch.org/](https://pytorch.org/)
* NLTK – [https://www.nltk.org/](https://www.nltk.org/)
* Flask – [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)

---