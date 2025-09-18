from transformers import AutoTokenizer, T5ForConditionalGeneration

print("Loading T5-small model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
print("Model loaded successfully!\n")

print("=" * 60)
print("TRAINING EXAMPLE")
print("=" * 60)

# training
input_text = "The <extra_id_0> walks in <extra_id_1> park"
target_text = "<extra_id_0> cute dog <extra_id_1> the <extra_id_2>"

print(f"Input text: {input_text}")
print(f"Target text: {target_text}")

input_ids = tokenizer(input_text, return_tensors="pt").input_ids
labels = tokenizer(target_text, return_tensors="pt").input_ids

print(f"Input IDs shape: {input_ids.shape}")
print(f"Labels shape: {labels.shape}")

outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss
logits = outputs.logits

print(f"Training loss: {loss.item():.4f}")
print(f"Logits shape: {logits.shape}")
print(f"Logits tensor:\n{logits}")

print("\n" + "=" * 60)
print("INFERENCE EXAMPLE")
print("=" * 60)

# inference
inference_text = "summarize: studies have shown that owning a dog is good for you"
print(f"Input text: {inference_text}")

input_ids = tokenizer(inference_text, return_tensors="pt").input_ids
print(f"Input IDs shape: {input_ids.shape}")

outputs = model.generate(input_ids)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Generated output: {generated_text}")
print(f"Output IDs shape: {outputs.shape}")
print("=" * 60)