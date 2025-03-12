from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline,BitsAndBytesConfig

app = Flask(__name__)

@app.route('/getResponse', methods=['POST'])
def getResponse():
    torch.random.manual_seed(0)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,)
    model = AutoModelForCausalLM.from_pretrained(
    "./Phi-3.5-mini-instruct", 
    # device_map="cuda", 
    # torch_dtype="auto", 
    # trust_remote_code=True, 
    torch_dtype=torch.float16,
    quantization_config=quantization_config
)
    tokenizer = AutoTokenizer.from_pretrained("./Phi-3.5-mini-instruct", trust_remote_code=True)
    data = request.get_json()  # Get the JSON data from the request body
    userQuery = data.get('userQuery')    
    messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": userQuery},
    # {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    # {"role": "user", "content": "Generate a wrkout routine for me"},
]
    pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
    generation_args = {
    "max_new_tokens": 800,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}
    response = {
        'message': (pipe(messages, **generation_args))[0]
    }

    return jsonify(response)  # Return a JSON response

if __name__ == '__main__':
    app.run(debug=True)
