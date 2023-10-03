from datasets import load_dataset
from functools import partial


class Dataset:
    def __init__(self, dataset, max_length):
        self.dataset = load_dataset(dataset, split='train')
        self.max_length = max_length

    def format_prompt(self, sample):
        INTRO_BLURB = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.'
        INSTRUCTION_KEY = '### Instruction:'
        INPUT_KEY = 'Input:'
        RESPONSE_KEY = '### Response:'
        END_KEY = '### End'

        blurb = INTRO_BLURB
        instruction = f'{INSTRUCTION_KEY}\n{sample["instruction"]}'
        input_context = f'{INPUT_KEY}\n{sample["context"]}' if sample['context'] else None
        response = f'{RESPONSE_KEY}\n{sample["response"]}'
        end = f'{END_KEY}'

        parts = [part for part in [blurb, instruction,
                                   input_context, response, end] if part]

        formatted_prompt = '\n\n'.join(parts)

        sample['text'] = formatted_prompt

        return sample

    def preprocess_batch(self, batch, tokenizer):
        return tokenizer(
            batch['text'],
            max_length=self.max_length,
            truncation=True
        )

    def preprocess_dataset(self, tokenizer):
        _preprocessing_function = partial(
            self.preprocess_batch, tokenizer=tokenizer)
        dataset = self.dataset.map(self.format_prompt)

        dataset = dataset.map(
            _preprocessing_function,
            batched=True,
            remove_columns=['instruction', 'context',
                            'response', 'text', 'category']
        )

        dataset = dataset.filter(lambda sample: len(
            sample['input_ids']) < self.max_length)

        return dataset


if __name__ == '__main__':
    from model import Llama

    model = Llama()
    dataset = Dataset('databricks/databricks-dolly-15k', model.max_length)

    print('\nDataset Sample:')
    print(dataset.dataset[0])

    data = dataset.preprocess_dataset(model.tokenizer)
    print('\nPre-processed by Llama 2 Tokenizer')
    print(data[0])
