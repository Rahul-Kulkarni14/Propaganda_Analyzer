# app2.py
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
            grid-template-columns: repeat(3, minmax(0, 1fr));
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

        .btn-report {
            background: #8a5a22;
            min-width: 170px;
        }

        .btn-report:hover {
            background: #704618;
        }

        .btn-perf {
            background: #3d8b63;
            min-width: 170px;
        }

        .btn-perf:hover {
            background: #337755;
        }

        .btn-guide {
            background: #475569;
            min-width: 170px;
        }

        .btn-guide:hover {
            background: #334155;
        }

        .btn-dataset {
            background: #355f8c;
            min-width: 170px;
        }

        .btn-dataset:hover {
            background: #2c5078;
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

        .meta-grid,
        .dataset-grid {
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

        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }

        .dashboard-card {
            background: #ffffff;
            border: 1px solid #d6e1ea;
            border-radius: 10px;
            padding: 14px;
        }

        .dashboard-label {
            margin: 0 0 6px;
            color: #687789;
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
        }

        .dashboard-value {
            margin: 0;
            color: #1f2937;
            font-size: 22px;
            font-weight: 900;
        }

        .dashboard-note {
            margin: 4px 0 0;
            color: #64748b;
            font-size: 13px;
        }

        .distribution-box {
            background: #ffffff;
            border: 1px solid #d6e1ea;
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 16px;
        }

        .distribution-row {
            display: grid;
            grid-template-columns: 190px 1fr 46px;
            gap: 10px;
            align-items: center;
            margin: 10px 0;
        }

        .distribution-name {
            font-size: 13px;
            font-weight: 800;
            color: #2f4a55;
        }

        .bar-track {
            height: 10px;
            border-radius: 999px;
            background: #e5edf3;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            border-radius: 999px;
            background: #3b7c88;
        }

        .distribution-count {
            font-size: 13px;
            font-weight: 800;
            color: #1f2937;
            text-align: right;
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

        .guide-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
        }

        .guide-card {
            background: #ffffff;
            border: 1px solid #d6e1ea;
            border-radius: 10px;
            padding: 15px;
        }

        .guide-card h3 {
            margin: 0 0 8px;
            color: #2f4a55;
            font-size: 16px;
        }

        .guide-card p {
            margin: 7px 0;
            color: #374151;
            font-size: 14px;
            line-height: 1.5;
        }

        .guide-card strong {
            color: #1f2937;
        }

        .data-table {
            width: 100%;
            border-collapse: collapse;
            background: #ffffff;
            margin-bottom: 16px;
            font-size: 14px;
        }

        .data-table th,
        .data-table td {
            border: 1px solid #d6e1ea;
            padding: 10px;
            text-align: left;
            vertical-align: top;
        }

        .data-table th {
            background: #edf4f6;
            color: #2f4a55;
            font-weight: 800;
        }

        .sample-fragment {
            max-width: 520px;
            white-space: normal;
            line-height: 1.45;
        }

        .eval-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }

        .eval-card {
            background: #ffffff;
            border: 1px solid #d6e1ea;
            border-radius: 10px;
            padding: 14px;
        }

        .eval-title {
            margin: 0 0 10px;
            color: #2f4a55;
            font-size: 16px;
            font-weight: 900;
        }

        .eval-metric {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            padding: 8px 0;
            border-top: 1px solid #edf2f6;
            font-size: 14px;
        }

        .eval-metric:first-of-type {
            border-top: none;
        }

        .eval-metric strong {
            color: #1f2937;
        }

        .raw-eval-box {
            background: #101827;
            color: #e5edf3;
            border-radius: 10px;
            padding: 16px;
            overflow-x: auto;
            font-size: 13px;
            line-height: 1.55;
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
            .meta-grid,
            .dataset-grid,
            .guide-grid,
            .dashboard-grid,
            .eval-grid {
                grid-template-columns: 1fr;
            }

            .distribution-row {
                grid-template-columns: 1fr;
                gap: 6px;
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

                    <div class="control-group">
                        <p class="section-title">Case Studies</p>
                        <div class="control-row">
                            <select id="caseStudySelect">
                                <option value="">Choose sample</option>
                                <option value="patriotic">Patriotic Speech</option>
                                <option value="attack">Attack Speech</option>
                                <option value="support">Mass Support Speech</option>
                                <option value="testimonial">Testimonial Claim</option>
                            </select>
                            <button class="btn-upload" onclick="loadCaseStudy()">Load Sample</button>
                        </div>
                    </div>
                </div>

                <div class="actions">
                    <button class="btn-analyze" onclick="analyze()">Analyze Speech</button>
                    <button class="btn-report" onclick="generateReport()">Generate Report</button>
                    <button class="btn-guide" onclick="showTechniqueGuide()">Technique Guide</button>
                    <button class="btn-dataset" onclick="showDatasetExplorer()">Dataset Explorer</button>
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
    let latestAnalysis = null;

    const caseStudies = {
        patriotic: `Millions of patriotic citizens are standing together to defend the nation, the flag, and our freedom.`,

        attack: `The corrupt and shameful leaders are destroying innocent families with their dangerous and disgusting decisions.`,

        support: `The majority of citizens support this movement because everyone knows it is the only solution to the problem.`,

        testimonial: `I witnessed this plan working in my neighborhood, and several families said it improved their lives.`
    };

    const techniqueGuide = [
        {
            name: "Appeal_to_Authority",
            meaning: "Uses experts, officials, institutions, or authority figures to make a claim seem more convincing.",
            example: "Experts say this policy is the only responsible choice.",
            cues: "expert, official, scientist, doctor, research, study, proven"
        },
        {
            name: "Repetition",
            meaning: "Repeats words or ideas to make a message more memorable or persuasive.",
            example: "We need change, change today, and change for every family.",
            cues: "again, repeatedly, always, never, repeated slogans"
        },
        {
            name: "Doubt",
            meaning: "Creates uncertainty or suspicion, often without strong evidence.",
            example: "Can we really trust what they are telling us?",
            cues: "maybe, perhaps, allegedly, supposedly, unverified, question"
        },
        {
            name: "Name_Calling",
            meaning: "Uses negative labels or insults to attack a person, group, or idea.",
            example: "Those corrupt traitors have ruined everything.",
            cues: "traitor, corrupt, criminal, liar, enemy, extremist"
        },
        {
            name: "Appeal_to_Fear",
            meaning: "Uses fear, danger, threats, or panic to influence the audience.",
            example: "If we do not act now, our country will collapse.",
            cues: "danger, threat, fear, disaster, crisis, attack, collapse"
        },
        {
            name: "Exaggeration",
            meaning: "Overstates claims using extreme or absolute language.",
            example: "This is the worst disaster in history and everyone knows it.",
            cues: "always, never, everyone, nobody, massive, unbelievable"
        },
        {
            name: "Loaded_Language",
            meaning: "Uses emotionally charged words to influence how the reader feels.",
            example: "The shameful decision destroyed the hopes of innocent families.",
            cues: "evil, brave, shameful, dangerous, disgusting, glorious"
        },
        {
            name: "Bandwagon",
            meaning: "Suggests that many people support something, so others should support it too.",
            example: "Millions have already joined us, and everyone is choosing this path.",
            cues: "everyone, majority, millions, join, popular, supporters"
        },
        {
            name: "Stereotyping",
            meaning: "Makes broad generalizations about a group of people.",
            example: "Those people always refuse to follow the rules.",
            cues: "all, always, never, these people, those people, their kind"
        },
        {
            name: "Flag_Waving",
            meaning: "Appeals to patriotism, national identity, loyalty, or symbols of the nation.",
            example: "Every true patriot must defend our flag and homeland.",
            cues: "nation, country, patriot, flag, freedom, homeland"
        },
        {
            name: "Causal_Oversimplification",
            meaning: "Presents a complex issue as if it has one simple cause or solution.",
            example: "All our problems exist because of one bad policy.",
            cues: "because of, only reason, single cause, blame, responsible for"
        },
        {
            name: "Appeal_to_Pity",
            meaning: "Uses suffering, hardship, or sympathy to persuade the audience.",
            example: "Think of the helpless families who are suffering every day.",
            cues: "suffer, poor, helpless, victim, pain, hardship, sympathy"
        },
        {
            name: "Red_Herring",
            meaning: "Shifts attention away from the main issue toward something less relevant.",
            example: "Why discuss corruption when there are other problems too?",
            cues: "instead, what about, look at, ignore, distract"
        },
        {
            name: "Card_Stacking",
            meaning: "Presents selective evidence while leaving out important opposing information.",
            example: "The facts clearly show success, without mentioning the failures.",
            cues: "only, clearly, facts show, undeniable, without mentioning"
        },
        {
            name: "Testimonial",
            meaning: "Uses personal experience, endorsement, or witness claims as persuasive evidence.",
            example: "I used this program myself and recommend it to everyone.",
            cues: "I believe, I saw, my experience, witness, endorsed, recommended"
        }
    ];

    function escapeHtml(value) {
        return String(value || '')
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#039;');
    }

    function buildAnalysisDashboard(results) {
        if (!results || results.length === 0) {
            return `
                <div class="dashboard-grid">
                    <div class="dashboard-card">
                        <p class="dashboard-label">Manipulation Status</p>
                        <p class="dashboard-value">Clear</p>
                        <p class="dashboard-note">No high-confidence propaganda fragments shown</p>
                    </div>
                    <div class="dashboard-card">
                        <p class="dashboard-label">High-Confidence Fragments</p>
                        <p class="dashboard-value">0</p>
                        <p class="dashboard-note">No flagged fragments above threshold</p>
                    </div>
                    <div class="dashboard-card">
                        <p class="dashboard-label">Average Confidence</p>
                        <p class="dashboard-value">0%</p>
                        <p class="dashboard-note">No confidence score available</p>
                    </div>
                    <div class="dashboard-card">
                        <p class="dashboard-label">Top Technique</p>
                        <p class="dashboard-value">None</p>
                        <p class="dashboard-note">No technique assigned</p>
                    </div>
                </div>
            `;
        }

        const techniqueCounts = {};
        let totalConfidence = 0;
        let highestRisk = results[0];

        results.forEach(item => {
            techniqueCounts[item.technique] = (techniqueCounts[item.technique] || 0) + 1;
            totalConfidence += Number(item.technique_confidence || 0);

            if (Number(item.technique_confidence || 0) > Number(highestRisk.technique_confidence || 0)) {
                highestRisk = item;
            }
        });

        const avgConfidence = (totalConfidence / results.length).toFixed(2);
        const sortedTechniques = Object.entries(techniqueCounts).sort((a, b) => b[1] - a[1]);
        const topTechnique = sortedTechniques[0][0];
        const maxCount = sortedTechniques[0][1];

        let distributionHtml = `
            <div class="distribution-box">
                <p class="section-title">Technique Distribution</p>
        `;

        sortedTechniques.forEach(([technique, count]) => {
            const width = Math.round((count / maxCount) * 100);
            distributionHtml += `
                <div class="distribution-row">
                    <div class="distribution-name">${escapeHtml(technique)}</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:${width}%"></div>
                    </div>
                    <div class="distribution-count">${count}</div>
                </div>
            `;
        });

        distributionHtml += `</div>`;

        return `
            <div class="dashboard-grid">
                <div class="dashboard-card">
                    <p class="dashboard-label">Manipulation Status</p>
                    <p class="dashboard-value">Detected</p>
                    <p class="dashboard-note">High-confidence fragments found</p>
                </div>
                <div class="dashboard-card">
                    <p class="dashboard-label">High-Confidence Fragments</p>
                    <p class="dashboard-value">${results.length}</p>
                    <p class="dashboard-note">Above selected confidence threshold</p>
                </div>
                <div class="dashboard-card">
                    <p class="dashboard-label">Average Confidence</p>
                    <p class="dashboard-value">${avgConfidence}%</p>
                    <p class="dashboard-note">Mean technique confidence</p>
                </div>
                <div class="dashboard-card">
                    <p class="dashboard-label">Top Technique</p>
                    <p class="dashboard-value">${escapeHtml(topTechnique)}</p>
                    <p class="dashboard-note">Most frequent detected class</p>
                </div>
            </div>

            ${distributionHtml}

            <div class="translated-box">
                <strong>Highest Confidence Fragment</strong><br>
                ${escapeHtml(highestRisk.technique)} (${escapeHtml(highestRisk.technique_confidence)}%)<br><br>
                ${escapeHtml(highestRisk.fragment)}
            </div>
        `;
    }

    function generateReport() {
        if (!latestAnalysis) {
            document.getElementById('result').innerHTML =
                '<i style="color:#9a6515">Please analyze some text before generating a report.</i>';
            return;
        }

        const { originalText, translation, rawResults, results, confidenceThreshold, generatedAt } = latestAnalysis;

        const techniqueCounts = {};
        let totalConfidence = 0;
        let highestRisk = null;

        results.forEach(item => {
            techniqueCounts[item.technique] = (techniqueCounts[item.technique] || 0) + 1;
            totalConfidence += Number(item.technique_confidence || 0);

            if (!highestRisk || Number(item.technique_confidence || 0) > Number(highestRisk.technique_confidence || 0)) {
                highestRisk = item;
            }
        });

        const avgConfidence = results.length
            ? (totalConfidence / results.length).toFixed(2)
            : "0.00";

        const sortedTechniques = Object.entries(techniqueCounts).sort((a, b) => b[1] - a[1]);
        const topTechnique = sortedTechniques.length ? sortedTechniques[0][0] : "None";

        const distributionRows = sortedTechniques.length
            ? sortedTechniques.map(([technique, count]) => `
                <tr>
                    <td>${escapeHtml(technique)}</td>
                    <td>${count}</td>
                </tr>
            `).join('')
            : `<tr><td colspan="2">No high-confidence techniques detected</td></tr>`;

        const fragmentRows = results.length
            ? results.map((item, index) => {
                const cues = item.matched_cues && item.matched_cues.length
                    ? item.matched_cues.join(', ')
                    : 'No direct keyword cue found';

                return `
                    <div class="fragment-card">
                        <h3>${index + 1}. ${escapeHtml(item.technique)}
                            <span>${escapeHtml(item.technique_confidence)}%</span>
                        </h3>
                        <p>${escapeHtml(item.fragment)}</p>
                        <p><strong>Explanation:</strong> ${escapeHtml(item.explanation)}</p>
                        <p><strong>Matched cues:</strong> ${escapeHtml(cues)}</p>
                    </div>
                `;
            }).join('')
            : `<div class="empty">No high-confidence propaganda fragments were detected.</div>`;

        const translatedSection = translation.was_translated
            ? `
                <section>
                    <h2>Translated Text</h2>
                    <div class="text-box">${escapeHtml(translation.translated_text)}</div>
                </section>
            `
            : '';

        const reportHtml = `
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Propaganda Analysis Report</title>
    <style>
        body {
            margin: 0;
            padding: 32px;
            font-family: "Segoe UI", Arial, sans-serif;
            background: #eef3f7;
            color: #1f2937;
        }

        .report {
            max-width: 980px;
            margin: 0 auto;
            background: #ffffff;
            border: 1px solid #d6e1ea;
            border-radius: 12px;
            overflow: hidden;
        }

        header {
            padding: 28px 32px;
            background: #315b67;
            color: #ffffff;
        }

        header h1 {
            margin: 0;
            font-size: 30px;
        }

        header p {
            margin: 8px 0 0;
            color: #e6f0f2;
        }

        main {
            padding: 28px 32px 34px;
        }

        section {
            margin-bottom: 24px;
        }

        h2 {
            margin: 0 0 12px;
            color: #2f4a55;
            font-size: 20px;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
        }

        .summary-card {
            border: 1px solid #d6e1ea;
            border-radius: 10px;
            padding: 14px;
            background: #f8fbfc;
        }

        .label {
            margin: 0 0 6px;
            color: #64748b;
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
        }

        .value {
            margin: 0;
            font-size: 21px;
            font-weight: 900;
        }

        .text-box {
            border: 1px solid #d6e1ea;
            border-radius: 10px;
            padding: 14px;
            background: #f8fbfc;
            white-space: pre-wrap;
            line-height: 1.6;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            background: #ffffff;
        }

        th, td {
            border: 1px solid #d6e1ea;
            padding: 10px;
            text-align: left;
            vertical-align: top;
        }

        th {
            background: #edf4f6;
            color: #2f4a55;
        }

        .fragment-card {
            border: 1px solid #d6e1ea;
            border-left: 5px solid #3b7c88;
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 14px;
            background: #ffffff;
        }

        .fragment-card h3 {
            margin: 0 0 10px;
            color: #2f4a55;
            font-size: 17px;
        }

        .fragment-card h3 span {
            color: #247c54;
            font-size: 14px;
        }

        .fragment-card p {
            margin: 8px 0;
            line-height: 1.55;
        }

        .empty {
            border: 1px solid #d6e1ea;
            border-radius: 10px;
            padding: 14px;
            background: #f8fbfc;
            color: #247c54;
            font-weight: 800;
        }

        .report-actions {
            margin-bottom: 18px;
            display: flex;
            gap: 10px;
        }

        button {
            min-height: 42px;
            border: none;
            border-radius: 8px;
            padding: 0 16px;
            background: #315b67;
            color: white;
            font-weight: 800;
            cursor: pointer;
        }

        @media print {
            body {
                background: white;
                padding: 0;
            }

            .report {
                border: none;
                border-radius: 0;
            }

            .report-actions {
                display: none;
            }
        }
    </style>
</head>
<body>
    <div class="report">
        <header>
            <h1>Propaganda Analysis Report</h1>
            <p>Generated on ${escapeHtml(generatedAt)}</p>
        </header>

        <main>
            <div class="report-actions">
                <button onclick="window.print()">Print / Save as PDF</button>
            </div>

            <section>
                <h2>Analysis Summary</h2>
                <div class="summary-grid">
                    <div class="summary-card">
                        <p class="label">Status</p>
                        <p class="value">${results.length ? 'Detected' : 'Clear'}</p>
                    </div>
                    <div class="summary-card">
                        <p class="label">Shown Fragments</p>
                        <p class="value">${results.length}</p>
                    </div>
                    <div class="summary-card">
                        <p class="label">Average Confidence</p>
                        <p class="value">${avgConfidence}%</p>
                    </div>
                    <div class="summary-card">
                        <p class="label">Top Technique</p>
                        <p class="value">${escapeHtml(topTechnique)}</p>
                    </div>
                </div>
            </section>

            <section>
                <h2>Processing Details</h2>
                <table>
                    <tr>
                        <th>Detected Language</th>
                        <td>${escapeHtml(translation.detected_language_name)} (${escapeHtml(translation.detected_language)})</td>
                    </tr>
                    <tr>
                        <th>Translated To English</th>
                        <td>${translation.was_translated ? 'Yes' : 'No'}</td>
                    </tr>
                    <tr>
                        <th>Confidence Threshold</th>
                        <td>${confidenceThreshold}%</td>
                    </tr>
                    <tr>
                        <th>Raw Model Detections</th>
                        <td>${rawResults.length}</td>
                    </tr>
                    <tr>
                        <th>High-Confidence Detections Shown</th>
                        <td>${results.length}</td>
                    </tr>
                </table>
            </section>

            <section>
                <h2>Technique Distribution</h2>
                <table>
                    <tr>
                        <th>Technique</th>
                        <th>Count</th>
                    </tr>
                    ${distributionRows}
                </table>
            </section>

            ${highestRisk ? `
            <section>
                <h2>Highest Confidence Fragment</h2>
                <div class="text-box">
                    ${escapeHtml(highestRisk.technique)} (${escapeHtml(highestRisk.technique_confidence)}%)<br><br>
                    ${escapeHtml(highestRisk.fragment)}
                </div>
            </section>
            ` : ''}

            <section>
                <h2>Original Input</h2>
                <div class="text-box">${escapeHtml(originalText)}</div>
            </section>

            ${translatedSection}

            <section>
                <h2>Detected Fragments</h2>
                ${fragmentRows}
            </section>
        </main>
    </div>
</body>
</html>
        `;

        const reportWindow = window.open('', '_blank');

        if (!reportWindow) {
            document.getElementById('result').innerHTML =
                '<i style="color:#b84a4a">Popup blocked. Please allow popups to generate the report.</i>';
            return;
        }

        reportWindow.document.open();
        reportWindow.document.write(reportHtml);
        reportWindow.document.close();
    }
    function loadCaseStudy() {
        const selected = document.getElementById('caseStudySelect').value;

        if (!selected) {
            document.getElementById('result').innerHTML =
                '<i style="color:#9a6515">Please choose a case study first.</i>';
            return;
        }

        document.getElementById('speech').value = caseStudies[selected];
        document.getElementById('result').innerHTML =
            '<span class="loading">Case study loaded. Click Analyze Speech.</span>';
    }

    function showTechniqueGuide() {
        let html = `
            <div class="translated-box">
                <strong>Technique Knowledge Guide</strong><br>
                The model classifies detected propaganda fragments into these 15 technique categories.
            </div>
            <div class="guide-grid">
        `;

        techniqueGuide.forEach(item => {
            html += `
                <div class="guide-card">
                    <h3>${escapeHtml(item.name)}</h3>
                    <p><strong>Meaning:</strong> ${escapeHtml(item.meaning)}</p>
                    <p><strong>Example:</strong> ${escapeHtml(item.example)}</p>
                    <p><strong>Common cues:</strong> ${escapeHtml(item.cues)}</p>
                </div>
            `;
        });

        html += `</div>`;
        document.getElementById('result').innerHTML = html;
    }

    async function showDatasetExplorer() {
        document.getElementById('result').innerHTML =
            '<span class="loading">Loading dataset explorer...</span>';

        const res = await fetch('/dataset-summary');
        const data = await res.json();

        if (!res.ok || data.error) {
            document.getElementById('result').innerHTML =
                '<i style="color:#b84a4a">' + escapeHtml(data.error || 'Could not load dataset summary.') + '</i>';
            return;
        }

        let labelRows = '';
        Object.entries(data.label_distribution).forEach(([label, count]) => {
            labelRows += `
                <tr>
                    <td>${escapeHtml(label)}</td>
                    <td>${count}</td>
                </tr>
            `;
        });

        let techniqueRows = '';
        data.technique_distribution.forEach(item => {
            techniqueRows += `
                <tr>
                    <td>${escapeHtml(item.technique)}</td>
                    <td>${item.count}</td>
                    <td>${item.percentage}%</td>
                </tr>
            `;
        });

        let sampleRows = '';
        data.sample_rows.forEach((row, index) => {
            sampleRows += `
                <tr>
                    <td>${index + 1}</td>
                    <td>${escapeHtml(row.technique)}</td>
                    <td>${escapeHtml(row.label)}</td>
                    <td class="sample-fragment">${escapeHtml(row.text_fragment)}</td>
                </tr>
            `;
        });

        const html = `
            <div class="translated-box">
                <strong>Dataset Explorer</strong><br>
                This view summarizes the processed training dataset used by the propaganda detection pipeline.
            </div>

            <div class="dashboard-grid">
                <div class="dashboard-card">
                    <p class="dashboard-label">Total Rows</p>
                    <p class="dashboard-value">${data.total_rows}</p>
                    <p class="dashboard-note">Rows in final_dataset.csv</p>
                </div>
                <div class="dashboard-card">
                    <p class="dashboard-label">Text Fragments</p>
                    <p class="dashboard-value">${data.text_fragment_count}</p>
                    <p class="dashboard-note">Available fragment records</p>
                </div>
                <div class="dashboard-card">
                    <p class="dashboard-label">Technique Classes</p>
                    <p class="dashboard-value">${data.technique_count}</p>
                    <p class="dashboard-note">Unique propaganda techniques</p>
                </div>
                <div class="dashboard-card">
                    <p class="dashboard-label">Dataset File</p>
                    <p class="dashboard-value">CSV</p>
                    <p class="dashboard-note">processed/final_dataset.csv</p>
                </div>
            </div>

            <div class="dataset-grid">
                <div class="distribution-box">
                    <p class="section-title">Binary Label Distribution</p>
                    <table class="data-table">
                        <tr>
                            <th>Label</th>
                            <th>Count</th>
                        </tr>
                        ${labelRows}
                    </table>
                </div>

                <div class="distribution-box">
                    <p class="section-title">Dataset Notes</p>
                    <p><strong>Most common technique:</strong> ${escapeHtml(data.most_common_technique)}</p>
                    <p><strong>Least common technique:</strong> ${escapeHtml(data.least_common_technique)}</p>
                    <p><strong>Purpose:</strong> Used for binary manipulation detection and multiclass propaganda technique classification.</p>
                </div>
            </div>

            <div class="distribution-box">
                <p class="section-title">Technique Distribution</p>
                <table class="data-table">
                    <tr>
                        <th>Technique</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                    ${techniqueRows}
                </table>
            </div>

            <div class="distribution-box">
                <p class="section-title">Sample Dataset Rows</p>
                <table class="data-table">
                    <tr>
                        <th>#</th>
                        <th>Technique</th>
                        <th>Label</th>
                        <th>Text Fragment</th>
                    </tr>
                    ${sampleRows}
                </table>
            </div>
        `;

        document.getElementById('result').innerHTML = html;
    }

    function extractAccuracy(sectionText) {
        const match = sectionText.match(/Accuracy:\s*([0-9.]+)/i);
        return match ? (Number(match[1]) * 100).toFixed(2) + '%' : 'N/A';
    }

    function extractWeightedAverage(sectionText) {
    const lines = sectionText.split(/\\r?\\n/).map(line => line.trim()).filter(Boolean);
    const weightedLine = lines.find(line => line.toLowerCase().startsWith('weighted avg'));

    if (!weightedLine) {
        return {
            precision: 'N/A',
            recall: 'N/A',
            f1: 'N/A',
            support: 'N/A'
        };
    }

    const parts = weightedLine.split(/\\s+/);

    return {
        precision: parts[2] ? (Number(parts[2]) * 100).toFixed(2) + '%' : 'N/A',
        recall: parts[3] ? (Number(parts[3]) * 100).toFixed(2) + '%' : 'N/A',
        f1: parts[4] ? (Number(parts[4]) * 100).toFixed(2) + '%' : 'N/A',
        support: parts[5] || 'N/A'
    };
}


    function buildEvaluationCard(title, sectionText) {
        const accuracy = extractAccuracy(sectionText);
        const weighted = extractWeightedAverage(sectionText);

        return `
            <div class="eval-card">
                <h3 class="eval-title">${escapeHtml(title)}</h3>
                <div class="eval-metric">
                    <span>Accuracy</span>
                    <strong>${escapeHtml(accuracy)}</strong>
                </div>
                <div class="eval-metric">
                    <span>Weighted Precision</span>
                    <strong>${escapeHtml(weighted.precision)}</strong>
                </div>
                <div class="eval-metric">
                    <span>Weighted Recall</span>
                    <strong>${escapeHtml(weighted.recall)}</strong>
                </div>
                <div class="eval-metric">
                    <span>Weighted F1-Score</span>
                    <strong>${escapeHtml(weighted.f1)}</strong>
                </div>
                <div class="eval-metric">
                    <span>Support</span>
                    <strong>${escapeHtml(weighted.support)}</strong>
                </div>
            </div>
        `;
    }

    function formatPerformanceOutput(rawOutput) {
        const binaryMarker = '===== Binary Model Performance =====';
        const multiclassMarker = '===== Multi-class Model Performance =====';

        const binaryStart = rawOutput.indexOf(binaryMarker);
        const multiclassStart = rawOutput.indexOf(multiclassMarker);

        let binarySection = '';
        let multiclassSection = '';

        if (binaryStart !== -1 && multiclassStart !== -1) {
            binarySection = rawOutput.slice(binaryStart, multiclassStart);
            multiclassSection = rawOutput.slice(multiclassStart);
        } else {
            return `
                <div class="translated-box">
                    <strong>Model Evaluation</strong><br>
                    Raw evaluation output is shown below because the summary parser could not detect both model sections.
                </div>
                <pre class="raw-eval-box">${escapeHtml(rawOutput)}</pre>
            `;
        }

        return `
            <div class="translated-box">
                <strong>Model Evaluation Dashboard</strong><br>
                This section summarizes the trained binary and multiclass transformer model performance using the existing evaluation pipeline.
            </div>

            <div class="eval-grid">
                ${buildEvaluationCard('Binary Classifier', binarySection)}
                ${buildEvaluationCard('Multiclass Technique Classifier', multiclassSection)}
            </div>

            <div class="distribution-box">
                <p class="section-title">Interpretation</p>
                <p><strong>Accuracy:</strong> Overall percentage of correct predictions.</p>
                <p><strong>Precision:</strong> How reliable the model's positive predictions are.</p>
                <p><strong>Recall:</strong> How many true examples the model successfully finds.</p>
                <p><strong>F1-score:</strong> Balance between precision and recall, useful for imbalanced classes.</p>
            </div>

            <div class="distribution-box">
                <p class="section-title">Raw Classification Report</p>
                <pre class="raw-eval-box">${escapeHtml(rawOutput)}</pre>
            </div>
        `;
    }

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
                '<i style="color:#b84a4a">' + escapeHtml(data.error) + '</i>';
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
                '<i style="color:#b84a4a">' + escapeHtml(data.error || 'Analysis failed.') + '</i>';
            return;
        }

        const translation = data.translation;
        const rawResults = data.results || [];
        const confidenceThreshold = 35;
        const results = rawResults.filter(item => Number(item.technique_confidence || 0) >= confidenceThreshold);

        latestAnalysis = {
            originalText: text,
            translation: translation,
            rawResults: rawResults,
            results: results,
            confidenceThreshold: confidenceThreshold,
            generatedAt: new Date().toLocaleString()
        };

        let filteredNote = '';

        if (rawResults.length !== results.length) {
            filteredNote = `
                <div class="translated-box">
                    <strong>Confidence Filter Applied</strong><br>
                    Showing ${results.length} high-confidence fragments out of ${rawResults.length} model detections.
                    Fragments below ${confidenceThreshold}% technique confidence are hidden to reduce noisy predictions.
                </div>
            `;
        }

        let html = filteredNote + buildAnalysisDashboard(results);

        html += `
            <div class="meta-grid">
                <div class="meta-card">
                    <p class="meta-label">Detected Language</p>
                    <p class="meta-value">${escapeHtml(translation.detected_language_name)} (${escapeHtml(translation.detected_language)})</p>
                </div>
                <div class="meta-card">
                    <p class="meta-label">Translated To English</p>
                    <p class="meta-value">${translation.was_translated ? 'Yes' : 'No'}</p>
                </div>
            </div>
        `;

        if (translation.error) {
            html += `<div class="translated-box"><strong>Translation Note:</strong><br>${escapeHtml(translation.error)}</div>`;
        }

        if (translation.was_translated) {
            html += `
                <div class="translated-box">
                    <strong>Translated Text</strong><br><br>
                    ${escapeHtml(translation.translated_text)}
                </div>
            `;
        }

        if (results.length === 0) {
            if (rawResults.length > 0) {
                html += `<div class="empty-state">The model found possible fragments, but none passed the ${confidenceThreshold}% confidence threshold.</div>`;
            } else {
                html += `<div class="empty-state">No propaganda/manipulation detected in this speech.</div>`;
            }
        } else {
            results.forEach((item, index) => {
                const cues = item.matched_cues || [];
                const cueHtml = cues.length
                    ? cues.map(cue => `<span class="cue-pill">${escapeHtml(cue)}</span>`).join('')
                    : '<span class="cue-pill">No direct keyword cue found</span>';

                html += `
                    <div class="analysis-card">
                        <div class="card-top">
                            <span class="technique-badge">${index + 1}. ${escapeHtml(item.technique)}</span>
                            <span class="confidence">Confidence: ${escapeHtml(item.technique_confidence)}%</span>
                        </div>
                        <p class="fragment-text">${escapeHtml(item.fragment)}</p>
                        <div class="xai-box">
                            <p class="xai-title">Lightweight Explanation</p>
                            <p class="xai-text">${escapeHtml(item.explanation)}</p>
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

        if (!res.ok || data.error) {
            document.getElementById('result').innerHTML =
                '<i style="color:#b84a4a">' + escapeHtml(data.error || 'Could not load model performance.') + '</i>';
            return;
        }

        document.getElementById('result').innerHTML = formatPerformanceOutput(data.output || '');
    }

    function clearInput() {
        document.getElementById('speech').value = '';
        document.getElementById('documentFile').value = '';
        document.getElementById('caseStudySelect').value = '';
        latestAnalysis = null;
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

    # Silent loading - no print messages
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

@app.route('/dataset-summary')
def dataset_summary_api():
    dataset_path = os.path.join('processed', 'final_dataset.csv')

    if not os.path.exists(dataset_path):
        return jsonify({
            'error': 'Dataset file not found: processed/final_dataset.csv'
        }), 404

    try:
        pd = lazy_import('pandas')
        df = pd.read_csv(dataset_path)

        required_columns = ['text_fragment', 'label', 'technique']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return jsonify({
                'error': 'Missing expected dataset columns: ' + ', '.join(missing_columns)
            }), 400

        label_distribution = {
            str(label): int(count)
            for label, count in df['label'].value_counts().sort_index().items()
        }

        technique_counts = df['technique'].value_counts()
        total_rows = int(len(df))

        technique_distribution = []
        for technique, count in technique_counts.items():
            technique_distribution.append({
                'technique': str(technique),
                'count': int(count),
                'percentage': round((int(count) / total_rows) * 100, 2) if total_rows else 0
            })

        sample_rows = []
        sample_df = df[['text_fragment', 'label', 'technique']].head(8)

        for _, row in sample_df.iterrows():
            sample_rows.append({
                'text_fragment': str(row['text_fragment']),
                'label': str(row['label']),
                'technique': str(row['technique'])
            })

        most_common_technique = str(technique_counts.idxmax()) if not technique_counts.empty else 'N/A'
        least_common_technique = str(technique_counts.idxmin()) if not technique_counts.empty else 'N/A'

        return jsonify({
            'total_rows': total_rows,
            'text_fragment_count': int(df['text_fragment'].notna().sum()),
            'technique_count': int(df['technique'].nunique()),
            'label_distribution': label_distribution,
            'technique_distribution': technique_distribution,
            'sample_rows': sample_rows,
            'most_common_technique': most_common_technique,
            'least_common_technique': least_common_technique
        })

    except Exception as exc:
        return jsonify({
            'error': f'Could not load dataset summary: {str(exc)}'
        }), 500

if __name__ == '__main__':
    print("="*60)
    print("Propaganda Analyzer Web App Started!")
    print("Open: http://127.0.0.1:5001")
    print("First use may take 15-50 seconds. Then instant!")
    print("="*60)
    app.run(host='0.0.0.0', port=5001, debug=False)
