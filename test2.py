import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline,BitsAndBytesConfig

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

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "Generate a wrkout routine for me"},
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# generation_args = {
#     "max_new_tokens": 500,
#     "return_full_text": False,
#     "temperature": 0.0,
#     "do_sample": False,
# }

generation_args = {
    "max_new_tokens": 800,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])