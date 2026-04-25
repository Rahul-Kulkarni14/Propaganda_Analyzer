# app.py 
import os
import sys
from io import StringIO
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Cache for instant performance after first click
performance_cache = None

# Helper: Capture any print() output
def capture_print_output(func, *args, **kwargs):
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    try:
        func(*args, **kwargs)
        return captured_output.getvalue().strip()
    finally:
        sys.stdout = old_stdout

# Lazy import (so app starts instantly)
def lazy_import(module_name):
    import importlib
    return importlib.import_module(module_name)

# Clean & Beautiful HTML
HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Propaganda Analyzer</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; margin: 40px; background: #f0f2f5; }
        .container { max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 16px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #7f8c8d; margin-bottom: 20px; font-size: 18px; }
        textarea { width: 100%; height: 180px; padding: 14px; border: 2px solid #ddd; border-radius: 10px; font-size: 16px; resize: vertical; }
        button { padding: 14px 28px; margin: 12px 8px; font-size: 16px; border: none; border-radius: 10px; cursor: pointer; transition: 0.3s; }
        select { padding: 12px; margin: 12px 8px; font-size: 16px; border: 2px solid #ddd; border-radius: 10px; }
        input[type="file"] { margin: 12px 8px; font-size: 15px; }
.btn-upload { background: #2980b9; color: white; }
.btn-upload:hover { background: #1f6391; transform: translateY(-2px); }

        .btn-mic { background: #8e44ad; color: white; }
        .btn-mic:hover { background: #71368a; transform: translateY(-2px); }
        .btn-analyze { background: #e74c3c; color: white; }
        .btn-analyze:hover { background: #c0392b; transform: translateY(-2px); }
        .btn-perf { background: #27ae60; color: white; }
        .btn-perf:hover { background: #1e8449; transform: translateY(-2px); }
        #result { margin-top: 30px; padding: 20px; border-radius: 10px; background: #2c3e50; color: #f1f1f1; min-height: 120px; white-space: pre-wrap; font-family: 'Courier New', monospace; line-height: 1.6; }
        .loading { color: #f39c12; font-style: italic; }
        .no-prop { color: #2ecc71; font-weight: bold; }
        .yes-prop { color: #e74c3c; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Propaganda & Manipulation Analyzer</h1>
        <p class="subtitle">Detect 15 propaganda techniques in real-time</p>
       <textarea id="speech" placeholder="Paste any speech, article, or text here..."></textarea><br>

<select id="speechLang">
    <option value="en-US">English</option>
    <option value="hi-IN">Hindi</option>
    <option value="mr-IN">Marathi</option>
    <option value="kn-IN">Kannada</option>
    <option value="ta-IN">Tamil</option>
    <option value="te-IN">Telugu</option>
    <option value="bn-IN">Bengali</option>
    <option value="gu-IN">Gujarati</option>
    <option value="pa-IN">Punjabi</option>
    <option value="ur-IN">Urdu</option>
    <option value="fr-FR">French</option>
    <option value="es-ES">Spanish</option>
    <option value="de-DE">German</option>
    <option value="it-IT">Italian</option>
    <option value="pt-PT">Portuguese</option>
    <option value="ja-JP">Japanese</option>
    <option value="ko-KR">Korean</option>
    <option value="zh-CN">Chinese</option>
    <option value="ar-SA">Arabic</option>
</select>

<button class="btn-mic" onclick="startListening()">Start Speaking</button>
<input type="file" id="documentFile" accept=".txt,.pdf">
<button class="btn-upload" onclick="uploadDocument()">Upload Document</button>


        <button class="btn-analyze" onclick="analyze()">Analyze Speech</button>
        <button class="btn-perf" onclick="showPerformance()">Model Performance</button>
        <div id="result">Results will appear here...</div>
    </div>

    <script>
        function startListening() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        document.getElementById('result').innerHTML =
            '<i style="color:#e67e22">Speech recognition is not supported in this browser. Please use Chrome or Edge.</i>';
        return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = document.getElementById('speechLang').value;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    document.getElementById('result').innerHTML =
        '<span class="loading">Listening... speak now.</span>';

    recognition.start();

    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        const textarea = document.getElementById('speech');

        if (textarea.value.trim()) {
            textarea.value += ' ' + transcript;
        } else {
            textarea.value = transcript;
        }

        document.getElementById('result').innerHTML =
            '<span class="loading">Speech converted to text. Click Analyze Speech.</span>';
    };

    recognition.onerror = function(event) {
        document.getElementById('result').innerHTML =
            '<i style="color:#e74c3c">Speech recognition error: ' + event.error + '</i>';
    };
}
async function uploadDocument() {
    const fileInput = document.getElementById('documentFile');
    const file = fileInput.files[0];

    if (!file) {
        document.getElementById('result').innerHTML =
            '<i style="color:#e67e22">Please choose a .txt or .pdf file first.</i>';
        return;
    }

    const formData = new FormData();
    formData.append('document', file);

    document.getElementById('result').innerHTML =
        '<span class="loading">Extracting text from document...</span>';

    const res = await fetch('/upload-document', {
        method: 'POST',
        body: formData
    });

    const data = await res.json();

    if (!res.ok || data.error) {
        document.getElementById('result').innerHTML =
            '<i style="color:#e74c3c">' + data.error + '</i>';
        return;
    }

    document.getElementById('speech').value = data.text;
    document.getElementById('result').innerHTML =
        '<span class="loading">Document text extracted. Click Analyze Speech.</span>';
}


        async function analyze() {
            const text = document.getElementById('speech').value.trim();
            if (!text) {
                document.getElementById('result').innerHTML = '<i style="color:#e67e22">Please enter some text.</i>';
                return;
            }
            document.getElementById('result').innerHTML = '<span class="loading">Analyzing speech...</span>';
            const res = await fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });
            const data = await res.json();
            const html = data.output
                .replace(/No propaganda/g, '<span class="no-prop">No propaganda</span>')
                .replace(/Propaganda\/manipulation detected/g, '<span class="yes-prop">Propaganda/manipulation detected</span>');
            document.getElementById('result').innerHTML = html || '<i>No output generated.</i>';
        }

        async function showPerformance() {
            document.getElementById('result').innerHTML = 
                '<span class="loading">Loading model performance...</span>';
            const res = await fetch('/performance');
            const data = await res.json();
            document.getElementById('result').innerHTML = '<pre>' + data.output + '</pre>';
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze_api():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'output': 'Please enter some text.'})

    from translation_utils import prepare_text_for_analysis

    translation_result = prepare_text_for_analysis(text)
    analysis_text = translation_result["translated_text"]

    speech_analyzer = lazy_import('speech_analyzer')
    
    import builtins
    real_input = builtins.input
    builtins.input = lambda _: analysis_text

    try:
        output = capture_print_output(speech_analyzer.analyze_speech, analysis_text)
    finally:
        builtins.input = real_input

    translation_summary = (
        f"Detected Language: {translation_result['detected_language_name']} "
        f"({translation_result['detected_language']})\n"
        f"Translated to English: {'Yes' if translation_result['was_translated'] else 'No'}\n"
    )

    if translation_result["error"]:
        translation_summary += f"Translation Note: {translation_result['error']}\n"

    if translation_result["was_translated"]:
        translation_summary += f"\nTranslated Text:\n{analysis_text}\n"

    final_output = (
        translation_summary
        + "\nAnalysis Result:\n"
        + (output or "No propaganda detected.")
    )

    return jsonify({'output': final_output})

@app.route('/upload-document', methods=['POST'])
def upload_document_api():
    if 'document' not in request.files:
        return jsonify({
            'text': '',
            'error': 'No document uploaded.'
        }), 400

    document = request.files['document']

    if not document.filename:
        return jsonify({
            'text': '',
            'error': 'No selected file.'
        }), 400

    from file_utils import extract_text_from_document

    extraction_result = extract_text_from_document(document)

    if extraction_result["error"]:
        return jsonify(extraction_result), 400

    return jsonify(extraction_result)

@app.route('/performance')
def performance_api():
    global performance_cache
    if performance_cache:
        return jsonify({'output': performance_cache})

    # Silent loading — no print messages
    performance = lazy_import('performance')
    from performance import evaluate_model, model_bin, model_mc, loader_bin, loader_mc

    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    try:
        evaluate_model(model_bin, loader_bin, task_name="Binary Model")
        evaluate_model(model_mc, loader_mc, task_name="Multi-class Model")
    finally:
        sys.stdout = old_stdout

    performance_cache = captured.getvalue().strip()
    return jsonify({'output': performance_cache})

if __name__ == '__main__':
    print("="*60)
    print("Propaganda Analyzer Web App Started!")
    print("Open: http://127.0.0.1:5000")
    print("First use may take 15-50 seconds. Then instant!")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=False)