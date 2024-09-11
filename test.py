from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define the path to the model directory
model_directory = r"C:\Users\nisha\Desktop\JupyterStuff\College\Smart India Hackathon 24'\model_stuff"  # Change this to the correct path

# Load the model and tokenizer from the local directory
model = AutoModelForSequenceClassification.from_pretrained(model_directory, from_tf=False, force_download=False)
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# Example usage: Tokenizing and predicting
inputs = tokenizer("This is a sample text", return_tensors="pt")
outputs = model(**inputs)
print(outputs)

