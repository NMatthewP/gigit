from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)
CORS(app)  # Allow CORS for all origins

# Load your model and tokenizer
model_directory = r"C:\Users\nisha\Desktop\JupyterStuff\College\Smart India Hackathon 24'\model_stuff"
model = AutoModelForSequenceClassification.from_pretrained(model_directory)
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# Define the label map
label_map = {
    'Plumber': 0,
    'Electrician': 1,
    'Web Designer': 2,
    'Graphic Designer': 3,
    'Content Writer': 4,
    'Software Developer': 5,
    'Carpenter': 6,
    'Painter': 7,
    'Mechanic': 8,
    'Gardener': 9,
    'Cleaner': 10,
    'Tutor': 11,
    'Chef': 12,
    'Photographer': 13,
    'Marketing Specialist': 14,
    'Data Analyst': 15,
    'Network Administrator': 16,
    'UX Designer': 17,
    'SEO Specialist': 18,
    'Accountant': 19,
    'Lawyer': 20,
    'Architect': 21,
    'Event Planner': 22,
    'Musician': 23,
    'Translator': 24,
    'Research Scientist': 25,
    'Nurse': 26,
    'Personal Trainer': 27,
    'Social Media Manager': 28,
    'Virtual Assistant': 29,
    'Video Editor': 30,
    'Project Manager': 31,
    'Interior Designer': 32,
    'Public Relations Specialist': 33,
    'Sales Representative': 34,
    'Legal Consultant': 35,
    'Fitness Instructor': 36,
    'Research Analyst': 37,
    'Travel Agent': 38,
    'AutoCAD Designer': 39,
    'Voice Actor': 40,
    'Data Scientist': 41,
    'HR Manager': 42,
    'Real Estate Agent': 43,
    'Investment Advisor': 44,
    'Graphic Illustrator': 45,
    'Database Administrator': 46,
    'Web Developer': 47,
    'Business Consultant': 48
}

# Create inverse label map (number to label)
inverse_label_map = {v: k for k, v in label_map.items()}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data.get('text', '')

    # Tokenize the input and get model outputs
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model(**inputs)

    # Get the probabilities and prediction
    probabilities = torch.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    
    # Map the numerical prediction to the label name
    label = inverse_label_map.get(prediction, 'Unknown')
    
    # Return the prediction
    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)
