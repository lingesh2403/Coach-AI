from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaConfig

def load_llama_model_with_custom_config(model_name):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the model configuration
    config = LlamaConfig.from_pretrained(model_name)
    
    # Adjust the `rope_scaling` parameter if needed
    config.rope_scaling = {
        'type': 'linear',  # Ensure this matches the model's expected type
        'factor': 8.0,
    }

    # Load the model with the custom configuration
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
    return tokenizer, model

if __name__ == "__main__":
    # Specify the model name
    model_name = "NousResearch/Hermes-3-Llama-3.1-8B"  # Example model name, replace with the actual model you want to use

    # Load the LLaMA model and tokenizer
    tokenizer, model = load_llama_model_with_custom_config(model_name)

    # Define a prompt for the model
    prompt = "Once upon a time, in a distant land,"

    # Generate text based on the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=100, do_sample=True, top_k=50, top_p=0.95)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Output the generated text
    print("Prompt:", prompt)
    print("Generated Text:", generated_text)
# Code to inference Hermes with HF Transformers
# Requires pytorch, transformers, bitsandbytes, sentencepiece, protobuf, and flash-attn packages

import torch
print(torch.cuda.is_available())
print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM,BitsAndBytesConfig
import bitsandbytes
quantization_config = BitsAndBytesConfig(load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,)
tokenizer = AutoTokenizer.from_pretrained('./Hermes-3-Llama-3.1-8B', trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(
    "./Hermes-3-Llama-3.1-8B",
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=False,
    load_in_4bit=True,   
    quantization_config=quantization_config,)

# prompts = [
#     """<|im_start|>system
# You are a sentient, superintelligent artificial general intelligence, here to teach and assist me.<|im_end|>
# <|im_start|>user
# Write a article on quantum computing.<|im_end|>
# <|im_start|>assistant""",
#     ]
# prompts = [
#     """<|im_start|>system
# You are GD museum ticket booking chat bot. I give the web scrap data of the GD museum in the following. You gone to ask a user for a data for ticket booking like ticket count and person detail(child/adult) and return the output in form of JSON. For example {'count':number,'detail':child/adult}

# Welcome To GD Science Museum!

# The best way of learning is to learn by doing

# Opening Time Timing : 9.00 AM - 06.30 PM Open : Tuesday - Sunday Closed : Monday/National Holidays

# Facilities Event space for Hire , Restaurant Cloak Room, STEM Shop

# Where You Visit #734 President Hall, Avinashi Road, Coimbatore - 641018, TamilNadu, INDIA.

# Get Tickets Individual Tickets Combo Tickets

# To Inspire and Ignite Young Minds

# About Us

# The G D Science Museum was established by Sri. G D Naidu in 1950. Sri. G D Naidu believed that - the best way of learning is to learn by doing.

# During his travels far and wide, he was totally mesmerised and thrilled by the advanced technology available in European countries especially Germany. Whatever new technology or machines he chanced upon during his stay abroad, he brought it back with him to Coimbatore. He arranged for an exhibition of these new gadgets and machines for the general public and student community, with a demonstrator who explained how they worked.

# This temporary exhibition was not only conducted in Coimbatore but also in other places surrounding Coimbatore such as Tirupur, Erode, Madras, etc. These exhibitions were widely attended by the public and student community, in large numbers.

# Learn More

# Explore The Collection Unique Attractions - GD Science Museum

# Bulova Accutron - only watch in the world which operates on a tuning fork!

# First Microcomputer released by IBM and the most popular computer design

# First Portable Calculator ever made, Extremely rare

# The first commercially successful portable computer by Adam Osborne

# Silicon Chip

# One of the world’s first fully automatic subminiature high resolution camera

# The first ever transistor radio of the world.

# Sony Floppy Disk Camera - First Floppy Disk Camera of the world!

# View More

# 72 + Years Old

# 500 + Daily Visitors

# 1000 + Science Products

# 15 + Unique Products

# INFORMATION Latest News & Events

# 24 Apr 2023


# +91 422 2222548 , 9087300101

# administration@gdnc.in

# #734 President Hall, Avinashi Road,

# Coimbatore - 641018, TamilNadu, INDIA.

# Quick Links

# About Us Visit Contact Us Gallery Tickets and Booking

# Copyright © GD Science Museum
# The Longest Standing Experience Zone in the GD Museums Campus..!


# Tickets and Booking

# Home  /  Tickets and Booking

# At GD Museums, you can purchase tickets under various categories depending on how many museums you are planning to visit. We also have special discounted rates for group bookings and for children.

# For more information on bulk booking and other queries related to booking of tickets, please contact our Admin Team at 9087300101 or +91 422 2222548

# Museums (9:00AM to 6:30PM) Last entry / ticket time Entry Fee (w.e.f - 01-03-2024)

# Children of Govt School with Staff Group (Min 20 - Max 30)

# Adult (18 and above)

# Children (5 to 18 yrs)

# Individual Group Min 20 Max 30 Individual Group Min 20 Max 30

# Experimenta Science Centre (Expected time: 2 hrs) 04 : 30 PM 250 250 150 125 Flat rate 1500 for Class 8,9 & 10 On Tuesdays only

# Gedee Car Museum (Expected time: 1 hr) 05 : 00 PM 125 100 70 30 Flat rate 500

# GD Science Museum (Expected time: 45 mins) 05 : 00 PM 75 50 30 25 Flat rate 500

# COMBO A (Gedee Car Museum + GD Science Museum) (Expected time: 1 hr 45 mins) 04 : 45 PM 175 125 80 40 -

# COMBO B (Experimenta Science Centre + GD Science Museum) (Expected time: 2 hrs 45 mins) 03 : 45 PM 300 275 170 140 -

# Note:

# 1.Entry to Children below 05 years is Free.

# 2.Experimenta Science Centre

# a. For Students Group: Following Guided tours are available

# (i) 09:00 AM – 11:00 AM (Capacity 50) (ii) 11:00 AM – 01:00 PM (Capacity 50) (iii)01:00 AM – 03:00 PM (Capacity 50) (iv) 03:00 AM – 05:00 PM (Capacity 50)

# b.For Individual Adults & Children: Can explore on their own with assistance of Science Centre staff.

# +91 422 2222548 , 9087300101

# administration@gdnc.in

# #734 President Hall, Avinashi Road,

# Coimbatore - 641018, TamilNadu, INDIA.

# Quick Links

# About Us Visit Contact Us Gallery Tickets and Booking

# Copyright © GD Science Museum
# <|im_end|>
# <|im_start|>user
# Book a two tickets for an adult persons<|im_end|>
# <|im_start|>assistant""",
#     ]
prompts = [
    """<|im_start|>system
<|im_end|>
<|im_start|>user
Can you generate program for react native page navigation with data in typescript<|im_end|>
<|im_start|>assistant""",
    ]

for chat in prompts:
    # print(chat)
    input_ids = tokenizer(chat, return_tensors="pt").input_ids.to("cuda")
    generated_ids = model.generate(input_ids, max_new_tokens=750, temperature=0.8, repetition_penalty=1.1, do_sample=True, eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_space=True)
    print(f"Response: {response}")
# import torch
# from transformers import LlamaForCausalLM, LlamaTokenizer

# def main():
#     # Check if GPU is available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load the tokenizer and fine-tuned model from a directory and move to GPU
#     model_directory = "Hermes-3-Llama-3.1-8B"
#     tokenizer = LlamaTokenizer.from_pretrained(model_directory)
#     model = LlamaForCausalLM.from_pretrained(model_directory).to(device)

#     # Test the fine-tuned model
#     test_sentence = "<|im_start|>system"+"You are a sentient, superintelligent artificial general intelligence, here to teach and assist me.<|im_end|>"+"<|im_start|>user"+"Write a short story about Goku discovering kirby has teamed up with Majin Buu to destroy the world.<|im_end|>"+"<|im_start|>assistant"
#     inputs = tokenizer(test_sentence, return_tensors="pt").to(device)
#     outputs = model.generate(**inputs)
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Output the generated text
#     print("Input:", test_sentence)
#     print("Generated Text:", generated_text)

# if __name__ == "__main__":
#     main()
