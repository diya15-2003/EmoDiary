from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the emotion classification model
classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion", return_all_scores=True)

# Original simplified emotion labels you want to use
emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

# Mapping from detailed model labels to your simplified emotions
label_mapping = {
    'anger': 'anger',
    'annoyance': 'anger',
    'disgust': 'disgust',
    'fear': 'fear',
    'sadness': 'sadness',
    'disappointment': 'sadness',
    'remorse': 'sadness',
    'joy': 'joy',
    'amusement': 'joy',
    'excitement': 'joy',
    'love': 'joy',
    'surprise': 'surprise',
    # Map all other labels to neutral (which you treat as 'joy' here)
    'admiration': 'joy',
    'approval': 'joy',
    'caring': 'joy',
    'confusion': 'joy',
    'curiosity': 'joy',
    'desire': 'joy',
    'embarrassment': 'joy',
    'gratitude': 'joy',
    'grief': 'sadness',
    'nervousness': 'fear',
    'optimism': 'joy',
    'pride': 'joy',
    'realization': 'joy',
    'relief': 'joy',
    'surprise': 'surprise',
    'neutral': 'joy'
}

# Pastel color palettes mapped to each emotion
color_palettes = {
    "anger":    ["#ffd6cc", "#fbb1a1", "#ffb3a7", "#e57373", "#ff8a80"],
    "disgust":  ["#e0c5de", "#d4afcd", "#cba6c3", "#bfa5c0", "#b199b3"],
    "fear":     ["#e6e6fa", "#d8d8ff", "#ccccff", "#b2b2ff", "#9999ff"],
    "joy":      ["#ffe4f0", "#ffc1da", "#ff9fcc", "#ff7fbf", "#ff60b3"],
    "sadness": ["#cce0b4", "#b0d189", "#a2c46f", "#90b957", "#8ab64a"],
    "surprise": ["#fffacc", "#fff799", "#fff47d", "#fff066", "#ffeb3b"]
}

# Stores last used palette
latest_emotion_palette = {"palette": color_palettes["joy"]}

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    global latest_emotion_palette
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'emotion': 'joy', 'scores': {}, 'palette': color_palettes["joy"]})

    text = data.get('text', '').strip()
    if not text:
        return jsonify({'emotion': 'joy', 'scores': {}, 'palette': color_palettes["joy"]})

    # Break into meaningful sentences
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if not sentences:
        return jsonify({'emotion': 'joy', 'scores': {}, 'palette': color_palettes["joy"]})

    try:
        predictions = []
        for sentence in sentences:
            raw_results = classifier(sentence)[0]  # List of dicts with 'label' and 'score'

            # Initialize simplified scores for this sentence
            simplified_scores = {label: 0.0 for label in emotion_labels}

            # Map detailed labels to simplified labels and aggregate scores
            for res in raw_results:
                mapped_label = label_mapping.get(res['label'].lower(), None)
                if mapped_label in simplified_scores:
                    simplified_scores[mapped_label] += res['score']

            predictions.append(simplified_scores)

            print(f"\nSentence: {sentence}")
            print("Mapped Prediction:", simplified_scores)

        # Average all sentence scores
        avg_scores = {label: np.mean([p.get(label, 0) for p in predictions]) for label in emotion_labels}
        total = sum(avg_scores.values())
        norm_scores = {k: (v / total if total else 0.0) for k, v in avg_scores.items()}
        
        print("Normalized Scores:", norm_scores)

        # Determine the top emotion
        top_emotion = max(norm_scores, key=norm_scores.get)
        top_score = norm_scores[top_emotion]

        # Threshold check to avoid incorrect high-emotion classifications
        threshold = 0.35
        if top_score < threshold:
            top_emotion = "joy"

        print("Top Emotion:", top_emotion)

        # Store latest palette for download use
        latest_emotion_palette["palette"] = color_palettes[top_emotion]

        return jsonify({
            'emotion': top_emotion,
            'scores': norm_scores,
            'palette': color_palettes[top_emotion]
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'emotion': 'joy', 'scores': {}, 'palette': color_palettes["joy"]})

@app.route('/get_palette', methods=['GET'])
def get_palette():
    return jsonify({'palette': latest_emotion_palette["palette"]})

if __name__ == '__main__':
    app.run(debug=True)
