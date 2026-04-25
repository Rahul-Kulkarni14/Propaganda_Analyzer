# Propaganda Analyzer – Speech Manipulation Detection System

An intelligent, modular NLP system that detects propaganda and manipulative techniques in speeches, articles, uploaded documents, and live speech input. The system uses DistilBERT-based models to classify text fragments as either manipulated or genuine, identifies the specific propaganda technique used, provides confidence scores, and gives lightweight explanations for detected fragments.

---

## Overview

Propaganda Analyzer is a hybrid text analysis framework that detects manipulative language in speeches and textual content using:

* **Binary classification model**: Identifies whether a text fragment contains manipulation.
* **Multiclass classification model**: Identifies the type of manipulative technique, such as Bandwagon, Appeal to Fear, Name Calling, and more.
* **Machine translation support**: Detects non-English input and translates it into English before analysis.
* **Speech-to-text input**: Allows users to speak directly through the browser microphone instead of typing.
* **Document upload support**: Extracts text from `.txt` and text-based `.pdf` files for analysis.
* **Confidence scoring**: Shows how confident the model is for each detected propaganda technique.
* **Lightweight explainability**: Highlights keyword-based linguistic cues that may support the detected technique.
* **Sentence-based fragmentation**: Splits long speeches into manageable fragments to improve detection accuracy.

The system is designed to analyze speeches of arbitrary length and highlight specific propaganda techniques for research, media analysis, social studies, and academic demonstrations.

---

## Features

* **Speech fragmentation**: Automatically splits speeches into sentences or word-based fragments.

* **Binary classification**: Detects whether a fragment contains manipulation.

* **Multiclass classification**: Detects 15 different manipulative techniques:

  * Appeal_to_Authority, Repetition, Doubt, Name_Calling, Appeal_to_Fear, Exaggeration, Loaded_Language, Bandwagon, Stereotyping, Flag_Waving, Causal_Oversimplification, Appeal_to_Pity, Red_Herring, Card_Stacking, Testimonial

* **Machine translation**: Detects the input language and translates non-English text into English before analysis.

* **Speech-to-text**: Uses browser-based speech recognition so users can speak directly into the app.

* **Multilingual speech input**: Supports selectable speech recognition languages such as English, Hindi, Marathi, Kannada, Tamil, Telugu, Bengali, Gujarati, Punjabi, Urdu, French, Spanish, German, Italian, Portuguese, Japanese, Korean, Chinese, and Arabic.

* **Document upload**: Supports `.txt` and text-based `.pdf` files.

* **Confidence scores**: Displays the confidence percentage for each detected propaganda fragment.

* **Lightweight XAI explanations**: Shows matched linguistic cues and short explanations for the predicted propaganda technique.

* **GPU support**: Uses PyTorch with CUDA for faster inference if available.

* **Improved Flask UI**: Provides a cleaner, organized web interface for text input, speech input, document upload, model performance, and analysis results.

* **Modular codebase**: Separate scripts for model training, inference, translation, file extraction, and performance evaluation.

---

## Technologies Used

| Component             | Framework / Library      | Purpose                                      |
| --------------------- | ------------------------ | -------------------------------------------- |
| Binary Classifier     | PyTorch / Transformers   | Detects presence of manipulation             |
| Multiclass Classifier | PyTorch / Transformers   | Identifies specific propaganda techniques    |
| Tokenization          | HuggingFace Transformers | Preprocessing and embeddings                 |
| Sentence Splitting    | NLTK                     | Fragmenting long speeches for analysis       |
| Web Framework         | Flask                    | Runs the web application                     |
| Translation           | deep-translator          | Translates non-English input into English    |
| Language Detection    | langdetect               | Detects input language                       |
| PDF Extraction        | pypdf                    | Extracts readable text from PDF files        |
| Speech Recognition    | Web Speech API           | Converts microphone speech into text         |
| Explainability        | Keyword-based XAI layer  | Provides lightweight explanation cues        |

---

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/Rahul-Kulkarni14/Propaganda_Analyzer.git
cd Propaganda_Analyzer
```

2. **Switch to the future-scope branch**:

```bash
git switch future-scope-upgrades
```

3. **Install Git LFS** for large model files:

```bash
git lfs install
git lfs pull
```

> **Note:** This ensures that model files (`.safetensors`) over 100 MB are properly downloaded. All `.safetensors` files in this repo are tracked with Git LFS.

4. **Install required Python packages**:

```bash
pip install -r requirements.txt
```

---

## Running the System

### 1. Launch the Web App

Run:

```bash
python app.py
```

Then open:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

### Web App Capabilities

* Paste or type speech/article text manually.
* Speak through the microphone and convert speech to text.
* Select the speech input language before using the microphone.
* Upload `.txt` or text-based `.pdf` documents.
* Automatically translate non-English text into English.
* Analyze the translated or original English text.
* View detected propaganda fragments.
* View detected propaganda technique names.
* View model confidence scores.
* View lightweight explanation cues for each detected fragment.
* View model performance from the web interface.

---

### 2. Analyze a Speech Using CLI

Run:

```bash
python speech_analyzer.py
```

* Enter a speech or textual input when prompted.
* The system will output detected propaganda fragments with corresponding techniques.

---

### 3. Train / Evaluate Models

Run the training and evaluation pipeline:

```bash
python main.py
```

* Outputs include performance metrics such as Accuracy, Precision, Recall, and F1-Score for both binary and multiclass models.
* Trained models are saved into local model folders.

---

### 4. View Model Performance

Run:

```bash
python performance.py
```

Or use the **Model Performance** button in the Flask web app.

---

## Web App Workflow

```text
User enters text, speaks through microphone, or uploads a document
        ↓
Text is extracted or transcribed
        ↓
Language is detected
        ↓
Non-English text is translated into English
        ↓
Speech is split into smaller fragments
        ↓
Binary model checks if each fragment is manipulative
        ↓
Multiclass model predicts the propaganda technique
        ↓
Confidence score is calculated
        ↓
Lightweight explanation cues are generated
        ↓
Results are shown in the Flask UI
```

---

## Model Insights

* **Binary Model**: Detects whether a speech fragment contains manipulation or not.
* **Multiclass Model**: Detects 15 types of manipulative techniques.
* **Fragment-based approach**: Shorter sentence-based fragments improve multiclass detection accuracy compared to large blocks of text.
* **Confidence score**: Uses model probability scores to display how confident the classifier is about the predicted technique.
* **Lightweight explainability**: Uses technique-specific keyword dictionaries to show linguistic cues that may support the prediction.

---

## Evaluation Metrics

The system provides:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Classification Report**
* **Fragment-level analysis**
* **Confidence scores for predictions**

---

## Supported Inputs

| Input Type      | Supported | Notes |
| --------------- | --------- | ----- |
| Typed text      | Yes       | User can paste or type text directly |
| Microphone input | Yes      | Works best in Chrome or Edge |
| `.txt` files    | Yes       | Text is extracted directly |
| Text-based `.pdf` files | Yes | Selectable/readable PDF text is extracted |
| Scanned PDFs    | Limited   | Requires OCR, not included in current version |
| Image files     | No        | Can be added later with OCR support |

---

## Limitations

* Translation depends on external translation services and may require an internet connection.
* Browser speech-to-text works best in Chrome or Edge.
* Speech recognition accuracy depends on microphone quality, pronunciation, and browser support.
* PDF extraction works for text-based PDFs, but scanned or image-based PDFs may not return readable text.
* Lightweight XAI is keyword-based and does not represent full SHAP or deep model interpretability.
* Confidence scores indicate model certainty, but a confident prediction can still be incorrect.

---

## Future Enhancements

* Add OCR support for scanned PDFs and image-based documents.
* Add downloadable analysis reports.
* Add full SHAP-based model interpretability.
* Add user history or saved analysis sessions.
* Add deployment configuration for cloud hosting.
* Improve model training with larger multilingual datasets.
* Add visual charts for model performance and confusion matrices.

---

## Applications

Propaganda Analyzer is useful for:

* Media analysis and journalism
* Research in social sciences and political studies
* Fact-checking speeches, interviews, or debates
* Identifying manipulative language in public communications
* Academic demonstrations of NLP, classification, translation, and explainability

---

## References

* HuggingFace Transformers – [https://huggingface.co/](https://huggingface.co/)
* PyTorch – [https://pytorch.org/](https://pytorch.org/)
* NLTK – [https://www.nltk.org/](https://www.nltk.org/)
* Flask – [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)
* deep-translator – [https://pypi.org/project/deep-translator/](https://pypi.org/project/deep-translator/)
* langdetect – [https://pypi.org/project/langdetect/](https://pypi.org/project/langdetect/)
* pypdf – [https://pypi.org/project/pypdf/](https://pypi.org/project/pypdf/)

---