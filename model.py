import torch
import bitsandbytes as bnb
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig


class Llama:
    def __init__(self, trainable=True, load_peft=''):
        self.model_name = 'meta-llama/Llama-2-7b-hf'
        self.model, self.tokenizer = self.load_model()
        self.max_length = self.get_max_length()

        if trainable:
            self.create_peft_model()

    @staticmethod
    def load_pretrained_model(path):
        model = Llama(trainable=False)
        model.load_peft_state(path)
        return model

    def load_peft_state(self, path):
        self.model = PeftModel.from_pretrained(self.model,
                                               path,
                                               torch_dtype=torch.bfloat16,
                                               is_trainable=False)

    def load_model(self):
        n_gpus = torch.cuda.device_count()
        max_memory = '12288MB'

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.get_bnb_config(),
            device_map='auto',
            max_memory={i: max_memory for i in range(n_gpus)}
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_atu_token=True)

        tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def get_bnb_config(self):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    def get_max_length(self):
        conf = self.model.config
        max_length = None
        for length_setting in ['n_positions', 'max_position_embeddings', 'seq_length']:
            max_length = getattr(self.model.config, length_setting, None)
            if max_length:
                print(f'Found max length: {max_length}')
                break
        if not max_length:
            max_length = 1024
            print(f'Using default max length: {max_length}')
        return max_length

    def create_peft_model(self):
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        modules = self.find_all_linear_names(self.model)
        peft_config = self.get_peft_config(modules)
        self.model = get_peft_model(self.model, peft_config)
        self.model.config.use_cache = False

    def find_all_linear_names(self, model):
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[-1])
        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def get_peft_config(self, modules):
        config = LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=modules,
            lora_dropout=0.1,
            bias='none',
            task_type='CAUSAL_LM'
        )
        return config

    def ask_model(self, question):
        tokenized = self.tokenizer(
            [question],
            max_length=self.max_length,
            truncation=True
        )
        response = self.model.generate(
            inputs=torch.as_tensor(tokenized['input_ids']).to('cuda'),
            generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
        return self.retrieve_solution(response)

    def retrieve_solution(self, response):
        response = self.tokenizer.decode(response[0])
        response = response.split('### Response:\n')[1].split('###')[0]
        response = '\n'.join(
            [line for line in response.split('\n') if line and line != '```'])
        return response

    def print_trainable_parameters(self, use_4bit=False):
        trainable_params = 0
        all_params = 0
        for _, param in self.model.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, 'ds_numel'):
                num_params = param.ds_numel

            all_params += num_params
            if param.requires_grad:
                trainable_params += num_params
        if use_4bit:
            trainable_params /= 2
        print(f'All parameters:  \t {all_params:,d}')
        print(f'Trainable params:\t {trainable_params:,d}')
        print(
            f'% Trainable params:\t {100 * trainable_params / all_params:.2f}%')
        print()


if __name__ == '__main__':
    model = Llama()
    model.print_trainable_parameters()
