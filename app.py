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
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            font-family: "Segoe UI", Arial, sans-serif;
            background: #e8eef3;
            color: #1f2937;
        }

        .page {
            width: 100%;
            max-width: 1180px;
            margin: 0 auto;
            padding: 32px 24px;
        }

        .app-shell {
            background: #ffffff;
            border: 1px solid #d4dee8;
            border-radius: 14px;
            box-shadow: 0 18px 45px rgba(31, 41, 55, 0.10);
            overflow: hidden;
        }

        .header {
            padding: 30px 34px 24px;
            background: linear-gradient(135deg, #315b67, #466f7a);
            border-bottom: 1px solid #d4dee8;
        }

        .header h1 {
            margin: 0;
            color: #ffffff;
            font-size: 34px;
            line-height: 1.2;
            letter-spacing: 0;
        }

        .subtitle {
            margin: 8px 0 0;
            color: #e6f0f2;
            font-size: 16px;
        }

        .content {
            padding: 28px 34px 34px;
            background: #fbfcfd;
        }

        .section-title {
            margin: 0 0 12px;
            font-size: 15px;
            font-weight: 700;
            color: #2f4a55;
            text-transform: uppercase;
            letter-spacing: 0;
        }

        textarea {
            width: 100%;
            min-height: 230px;
            padding: 16px;
            border: 1px solid #c8d6e2;
            border-radius: 10px;
            font-size: 16px;
            line-height: 1.55;
            resize: vertical;
            outline: none;
            color: #111827;
            background: #ffffff;
            font-family: "Segoe UI", Arial, sans-serif;
        }

        textarea:focus {
            border-color: #3b7c88;
            box-shadow: 0 0 0 3px rgba(59, 124, 136, 0.14);
        }

        .control-panel {
            margin-top: 18px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 18px;
        }

        .control-group {
            border: 1px solid #d6e1ea;
            background: #f3f7f9;
            border-radius: 10px;
            padding: 16px;
        }

        .control-row {
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
        }

        select,
        input[type="file"] {
            min-height: 44px;
            border: 1px solid #c8d6e2;
            border-radius: 8px;
            background: #ffffff;
            color: #111827;
            font-size: 15px;
        }

        select {
            min-width: 170px;
            padding: 0 12px;
        }

        input[type="file"] {
            flex: 1;
            min-width: 220px;
            padding: 10px;
        }

        .actions {
            margin-top: 22px;
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            align-items: center;
        }

        button {
            min-height: 46px;
            padding: 0 20px;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
            color: #ffffff;
        }

        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(31, 41, 55, 0.14);
        }

        .btn-mic {
            background: #6f5aa7;
        }

        .btn-mic:hover {
            background: #5f4b95;
        }

        .btn-upload {
            background: #3b7c88;
        }

        .btn-upload:hover {
            background: #316a74;
        }

        .btn-analyze {
            background: #b84a4a;
            min-width: 170px;
        }

        .btn-analyze:hover {
            background: #9f3d3d;
        }

        .btn-perf {
            background: #3d8b63;
            min-width: 170px;
        }

        .btn-perf:hover {
            background: #337755;
        }

        .btn-clear {
            background: #64748b;
            min-width: 110px;
        }

        .btn-clear:hover {
            background: #526174;
        }

        .result-shell {
            margin-top: 28px;
            border: 1px solid #d4dee8;
            border-radius: 12px;
            overflow: hidden;
            background: #ffffff;
        }

        .result-header {
            padding: 16px 18px;
            background: #edf4f6;
            border-bottom: 1px solid #d4dee8;
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: center;
        }

        .result-header h2 {
            margin: 0;
            font-size: 18px;
            color: #2f4a55;
        }

        .result-hint {
            font-size: 13px;
            color: #687789;
        }

        #result {
            min-height: 190px;
            padding: 20px;
            background: #f7fafb;
            color: #1f2937;
            white-space: normal;
            font-family: "Segoe UI", Arial, sans-serif;
            line-height: 1.6;
            overflow-x: auto;
        }

        .placeholder {
            color: #738196;
        }

        .loading {
            color: #9a6515;
            font-weight: 700;
        }

        .meta-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }

        .meta-card {
            background: #ffffff;
            border: 1px solid #d6e1ea;
            border-radius: 10px;
            padding: 12px 14px;
        }

        .meta-label {
            margin: 0 0 4px;
            color: #687789;
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
        }

        .meta-value {
            margin: 0;
            color: #1f2937;
            font-size: 15px;
            font-weight: 700;
        }

        .translated-box {
            background: #ffffff;
            border: 1px solid #d6e1ea;
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 16px;
            color: #1f2937;
            white-space: pre-wrap;
        }

        .analysis-card {
            background: #ffffff;
            border: 1px solid #d6e1ea;
            border-left: 5px solid #3b7c88;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 14px;
        }

        .card-top {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: center;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }

        .technique-badge {
            display: inline-block;
            background: #edf4f6;
            color: #2f4a55;
            border: 1px solid #b9c9d6;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 13px;
            font-weight: 800;
        }

        .confidence {
            color: #247c54;
            font-size: 14px;
            font-weight: 800;
        }

        .fragment-text {
            margin: 0;
            color: #1f2937;
            line-height: 1.6;
            white-space: pre-wrap;
        }

        .xai-box {
            margin-top: 12px;
            padding: 12px 14px;
            background: #f3f7f9;
            border: 1px solid #d6e1ea;
            border-radius: 10px;
        }

        .xai-title {
            margin: 0 0 8px;
            font-size: 13px;
            font-weight: 800;
            color: #2f4a55;
            text-transform: uppercase;
        }

        .xai-text {
            margin: 0;
            color: #374151;
            font-size: 14px;
            line-height: 1.5;
        }

        .cue-list {
            margin-top: 8px;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .cue-pill {
            display: inline-block;
            padding: 5px 9px;
            border-radius: 999px;
            background: #e2eef1;
            color: #2f4a55;
            font-size: 12px;
            font-weight: 800;
        }

        .empty-state {
            background: #ffffff;
            border: 1px solid #d6e1ea;
            border-radius: 10px;
            padding: 16px;
            color: #247c54;
            font-weight: 800;
        }

        pre {
            margin: 0;
            white-space: pre-wrap;
            font-family: "Consolas", "Courier New", monospace;
        }

        @media (max-width: 760px) {
            .page {
                padding: 18px;
            }

            .header,
            .content {
                padding-left: 20px;
                padding-right: 20px;
            }

            .header h1 {
                font-size: 28px;
            }

            .control-panel,
            .meta-grid {
                grid-template-columns: 1fr;
            }

            button,
            select,
            input[type="file"] {
                width: 100%;
            }

            .actions {
                align-items: stretch;
            }
        }
    </style>
</head>
<body>
    <main class="page">
        <section class="app-shell">
            <header class="header">
                <h1>Propaganda & Manipulation Analyzer</h1>
                <p class="subtitle">Analyze typed text, live speech, or uploaded documents for propaganda techniques.</p>
            </header>

            <div class="content">
                <p class="section-title">Input Text</p>
                <textarea id="speech" placeholder="Paste any speech, article, or text here..."></textarea>

                <div class="control-panel">
                    <div class="control-group">
                        <p class="section-title">Speech Input</p>
                        <div class="control-row">
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
                        </div>
                    </div>

                    <div class="control-group">
                        <p class="section-title">Document Upload</p>
                        <div class="control-row">
                            <input type="file" id="documentFile" accept=".txt,.pdf">
                            <button class="btn-upload" onclick="uploadDocument()">Upload Document</button>
                        </div>
                    </div>
                </div>

                <div class="actions">
                    <button class="btn-analyze" onclick="analyze()">Analyze Speech</button>
                    <button class="btn-perf" onclick="showPerformance()">Model Performance</button>
                    <button class="btn-clear" onclick="clearInput()">Clear</button>
                </div>

                <section class="result-shell">
                    <div class="result-header">
                        <h2>Results</h2>
                        <span class="result-hint">Fragment-level predictions with confidence and lightweight explanations</span>
                    </div>
                    <div id="result"><span class="placeholder">Results will appear here after analysis.</span></div>
                </section>
            </div>
        </section>
    </main>

    <script>
        function startListening() {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

            if (!SpeechRecognition) {
                document.getElementById('result').innerHTML =
                    '<i style="color:#9a6515">Speech recognition is not supported in this browser. Please use Chrome or Edge.</i>';
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
                    '<i style="color:#b84a4a">Speech recognition error: ' + event.error + '</i>';
            };
        }

        async function uploadDocument() {
            const fileInput = document.getElementById('documentFile');
            const file = fileInput.files[0];

            if (!file) {
                document.getElementById('result').innerHTML =
                    '<i style="color:#9a6515">Please choose a .txt or .pdf file first.</i>';
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
                    '<i style="color:#b84a4a">' + data.error + '</i>';
                return;
            }

            document.getElementById('speech').value = data.text;
            document.getElementById('result').innerHTML =
                '<span class="loading">Document text extracted. Click Analyze Speech.</span>';
        }

        async function analyze() {
            const text = document.getElementById('speech').value.trim();
            if (!text) {
                document.getElementById('result').innerHTML =
                    '<i style="color:#9a6515">Please enter some text.</i>';
                return;
            }

            document.getElementById('result').innerHTML =
                '<span class="loading">Analyzing speech...</span>';

            const res = await fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            const data = await res.json();

            if (!res.ok || data.error) {
                document.getElementById('result').innerHTML =
                    '<i style="color:#b84a4a">' + (data.error || 'Analysis failed.') + '</i>';
                return;
            }

            const translation = data.translation;
            const results = data.results || [];

            let html = `
                <div class="meta-grid">
                    <div class="meta-card">
                        <p class="meta-label">Detected Language</p>
                        <p class="meta-value">${translation.detected_language_name} (${translation.detected_language})</p>
                    </div>
                    <div class="meta-card">
                        <p class="meta-label">Translated To English</p>
                        <p class="meta-value">${translation.was_translated ? 'Yes' : 'No'}</p>
                    </div>
                </div>
            `;

            if (translation.error) {
                html += `<div class="translated-box"><strong>Translation Note:</strong><br>${translation.error}</div>`;
            }

            if (translation.was_translated) {
                html += `
                    <div class="translated-box">
                        <strong>Translated Text</strong><br><br>
                        ${translation.translated_text}
                    </div>
                `;
            }

            if (results.length === 0) {
                html += `<div class="empty-state">No propaganda/manipulation detected in this speech.</div>`;
            } else {
                results.forEach((item, index) => {
                    const cues = item.matched_cues || [];
                    const cueHtml = cues.length
                        ? cues.map(cue => `<span class="cue-pill">${cue}</span>`).join('')
                        : '<span class="cue-pill">No direct keyword cue found</span>';

                    html += `
                        <div class="analysis-card">
                            <div class="card-top">
                                <span class="technique-badge">${index + 1}. ${item.technique}</span>
                                <span class="confidence">Confidence: ${item.technique_confidence}%</span>
                            </div>
                            <p class="fragment-text">${item.fragment}</p>
                            <div class="xai-box">
                                <p class="xai-title">Lightweight Explanation</p>
                                <p class="xai-text">${item.explanation}</p>
                                <div class="cue-list">${cueHtml}</div>
                            </div>
                        </div>
                    `;
                });
            }

            document.getElementById('result').innerHTML = html;
        }

        async function showPerformance() {
            document.getElementById('result').innerHTML =
                '<span class="loading">Loading model performance...</span>';

            const res = await fetch('/performance');
            const data = await res.json();

            document.getElementById('result').innerHTML = '<pre>' + data.output + '</pre>';
        }

        function clearInput() {
            document.getElementById('speech').value = '';
            document.getElementById('documentFile').value = '';
            document.getElementById('result').innerHTML =
                '<span class="placeholder">Results will appear here after analysis.</span>';
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
        return jsonify({'error': 'Please enter some text.'}), 400

    from translation_utils import prepare_text_for_analysis

    translation_result = prepare_text_for_analysis(text)
    analysis_text = translation_result["translated_text"]

    speech_analyzer = lazy_import('speech_analyzer')
    results = speech_analyzer.analyze_speech_with_confidence(analysis_text)

    return jsonify({
        'translation': translation_result,
        'results': results
    })


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