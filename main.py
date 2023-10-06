import os
import torch
from model import Llama
from dataset import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

if __name__ == '__main__':
    output_dir = 'llama_adapter'
    llama = Llama()
    dataset = Dataset('databricks/databricks-dolly-15k', llama.max_length)
    data = dataset.preprocess_dataset(llama.tokenizer)

    trainer = Trainer(
        model=llama.model,
        train_dataset=data,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=20,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir='outputs',
            optim='paged_adamw_8bit'),
        data_collator=DataCollatorForLanguageModeling(
            llama.tokenizer, mlm=False))

    print('Training ...')

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()
    print(metrics)

    print('Saving last checkpoint of the model ...')
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)

    del llama
    del trainer
    torch.cuda.empty_cache()
