from huggingface_hub import InferenceClient

# Initialize the client
client = InferenceClient(
    "microsoft/Phi-3.5-mini-instruct",
    token="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",
)

for message in client.chat_completion(
	messages=[{"role": "user", "content": "What is the capital of France?"}],
	max_tokens=500,
	stream=True,
):
    print(message.choices[0].delta.content, end="")