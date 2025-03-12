import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("./Llama-3.2-1B")

# Define a sample input
input_text = "Give me the names of some fruit"

# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt")

# Generate output
with torch.no_grad():
    outputs = model.generate(inputs["input_ids"], max_length=100)

# Decode and print the output
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)
