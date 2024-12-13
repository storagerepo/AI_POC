!pip install transformers datasets accelerate

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import torch
from google.colab import files
uploaded = files.upload()

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import Dataset, load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Step 1: Load your dataset from JSON
def load_and_process_dataset(file_path):
    # Load the dataset from the provided JSON file
    dataset = load_dataset("json", data_files=file_path, split="train")

    # Define the formatting function for the prompts
    def formatting_prompts_func(examples):
        user_texts = examples["user"]  # Replace 'user' with the actual field name
        assistant_texts = examples["assistant"]  # Replace 'assistant' with the actual field name
        tool_calls_texts = examples["tool_calls"]  # Replace 'tool_calls' with the actual field name
        texts = [
            f"{{\"user\": \"{user}\", \"assistant\": \"{assistant}\", \"tool_calls\": {tool_calls}}}"
            for user, assistant, tool_calls in zip(user_texts, assistant_texts, tool_calls_texts)
        ]
        return {"text": texts}

    # Apply formatting to dataset
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset

# Load your custom dataset (replace 'tool_dataset.jsonl' with your actual JSON file path)
dataset = load_and_process_dataset("tool_dataset.jsonl")

# Step 2: Tokenize the dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 3: Define the model and training settings
model = GPT2LMHeadModel.from_pretrained("gpt2")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)





training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=5e-5,
    warmup_steps=50,
    weight_decay=0.01,
    prediction_loss_only=True,
    fp16=True,
    evaluation_strategy="steps",  # Use 'eval_strategy' if 'evaluation_strategy' shows warnings.
    eval_steps=10,
    save_strategy="steps",
    report_to=None  # Disable WandB integration
)

# Step 4: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Step 5: Train the model
trainer.train()

# Step 6: Save the model
model.save_pretrained("gpt2-finetuned")
tokenizer.save_pretrained("gpt2-finetuned")

from transformers import pipeline

# Load fine-tuned model
model_path = "./gpt2-finetuned"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Input query
query = "Can you search for condos in Maryland with a gym and parking?"
input_text = f"{{\"user\": \"{query}\", \"assistant\": "

# Generate response
response = generator(input_text, max_length=200, num_return_sequences=1)
print(response[0]["generated_text"])
