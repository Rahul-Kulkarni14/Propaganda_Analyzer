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
        <button class="btn-analyze" onclick="analyze()">Analyze Speech</button>
        <button class="btn-perf" onclick="showPerformance()">Model Performance</button>
        <div id="result">Results will appear here...</div>
    </div>

    <script>
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

    speech_analyzer = lazy_import('speech_analyzer')
    
    import builtins
    real_input = builtins.input
    builtins.input = lambda _: text

    try:
        output = capture_print_output(speech_analyzer.analyze_speech, text)
    finally:
        builtins.input = real_input

    return jsonify({'output': output or 'No propaganda detected.'})

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