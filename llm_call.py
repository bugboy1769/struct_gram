from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_dict = {
    "m1": "gpt2",
    "m2": "HuggingFaceTB/SmolLM-135M",
    "m3": "meta-llama/Llama-3.1-8B",
    "m4": "google-t5/t5-small",
}

inputs = ["the brown fox", "boss", "bruh"]
model = AutoModelForCausalLM.from_pretrained(model_dict["m2"]) #, local_files_only = True)
tokenizer = AutoTokenizer.from_pretrained(model_dict["m2"]) #, local_files_only = True)
device = ("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)


def tokenize(input):
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_input = tokenizer(input, padding = True, return_tensors = "pt")
    tokenized_input.to(device)
    inputs_embeds = model.get_input_embeddings()(tokenized_input['input_ids'])

    return inputs_embeds

def generate_batch_without_decode(input):
    tokenized_input = tokenize(input)
    with torch.no_grad():
        outputs = model(inputs_embeds = tokenized_input)
    logits = outputs.logits
    last_logits = logits[:, -1, :]
    next_token_ids = last_logits.argmax(dim =1)
    next_tokens = tokenizer.batch_decode(next_token_ids)

    return next_token_ids, next_tokens

print(generate_batch_without_decode(inputs))