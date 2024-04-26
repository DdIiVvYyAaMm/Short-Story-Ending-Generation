# pip install transformers datasets rouge-score nltk bert-score
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import csv
import random
from collections import *
from transformers import MambaForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import wandb
import torch
import argparse
from trl import SFTTrainer
from peft import LoraConfig

# %%
df_2016=pd.read_csv('/home/divyams/SI630/Short-Story-Ending-Generation/ROCStories__spring2016 - ROCStories_spring2016.csv')
df_2017=pd.read_csv('/home/divyams/SI630/Short-Story-Ending-Generation/ROCStories_winter2017 - ROCStories_winter2017.csv')
data = pd.concat([df_2016, df_2017])

# %%
print(data.head())

# %%
prompt_text = "Complete this story by generating its last line to give it a logical ending:\n"
data['input'] = data[['sentence1', 'sentence2', 'sentence3', 'sentence4']].agg(lambda x: f"{prompt_text} {' '.join(x)}", axis=1)
data['target'] = data['sentence5']

# Convert to Hugging Face dataset format
dataset = Dataset.from_pandas(data[['input', 'target']])
train_test_split = dataset.train_test_split(test_size=0.3)
datasets = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-370m-hf")
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-370m-hf").to(device)


parser = argparse.ArgumentParser(description="Run training with a checkpoint.")
    
    # Add argument for the checkpoint path with a named flag
parser.add_argument('--resume_from_checkpoint', type=str, required=True,
                        help='Path to the checkpoint from which to resume training.')

    # Parse the arguments
args = parser.parse_args()

if(args.resume_from_checkpoint):
    # Load the model from the specified checkpoint
    print(f"Loading model from: {args.resume_from_checkpoint}")
    model = MambaForCausalLM.from_pretrained(args.resume_from_checkpoint)




def tokenize_function(examples):
    model_inputs = tokenizer(examples['input'], padding="max_length", truncation=True, max_length=64)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target'], padding="max_length", truncation=True, max_length=64)['input_ids']

    model_inputs['labels'] = labels
    return model_inputs

tokenized_datasets = datasets.map(tokenize_function, batched=True)





# wandb.login(key="d0b60f68420737cfaed2d8f28e667b34cbb46094")
wandb.init(project="StoryGeneration", name="SSM_Mamba_model", entity="divyams")

# model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf").to(device)

class CustomTrainer(Trainer):
    def create_optimizer(self):
        optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return optimizer
    def compute_loss(self, model, inputs, return_outputs=False):
        # This method can be overridden to customize the loss computation.
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    

lora_config =  LoraConfig(
        r=8,
        target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
        task_type="CAUSAL_LM",
        bias="none"
)

training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    overwrite_output_dir=True,
    learning_rate=2e-3,# Learning rate
    per_device_train_batch_size=8,   # Batch size for training
    per_device_eval_batch_size=8,    # Batch size for evaluation
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=100,  # Adjust based on your preference and dataset size
    save_strategy="steps",
    save_steps=100,
    num_train_epochs=1,              # Number of epochs
    weight_decay=0.01,                # Weight decay
    logging_dir="./logs_SSM_Story",  # Consider a separate logging directory for clarity
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # Consider changing based on suitable metrics for your multilabel task
    greater_is_better=False,
    report_to="wandb",
    run_name="SSM_Mamba_1"
)


torch.cuda.empty_cache()

trainer = SFTTrainer(
    model=model,
    args=training_args,
    peft_config=lora_config,
    train_dataset=tokenized_datasets['train'].select(range(1, 10000)),
    eval_dataset=tokenized_datasets['test'].select(range(1, 1000)),
    tokenizer=tokenizer,
    dataset_text_field="input"
)

trainer.train()




print("Generating endings \n -----------------------")
# %%
def generate_text(input_text):
    model.eval()
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    output_ids = model.generate(input_ids, max_length=len(input_ids[0])+24)[0]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


test_inputs = tokenized_datasets['test'].select(range(5000, 7000))['input']
test_targets = tokenized_datasets['test'].select(range(5000, 7000))['target']

test_sentences = tokenized_datasets['test'].select(range(5000, 7000))['input']
generated_sentences = [generate_text(text) for text in test_sentences]



# Creating the DataFrame
results_df = pd.DataFrame({
    'Input': test_inputs,
    'Target': test_targets,
    'Generated': generated_sentences
})


results_df.to_csv('test_predictions.csv', index=False)

print("All Done!")