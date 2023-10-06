# Llama 2 QLoRA Fine Tuning

This project is designed for performing question-answering tasks using the Llama model. Llama is a powerful language model capable of generating responses to a variety of prompts. In this project, we provide code for utilizing Llama to answer questions based on a dataset. Additionally, we demonstrate the adaptation of Llama using LORA (Low-Rank Adaptation) for improved performance.

**Table of Contents**

- Requirements
- Usage
  - Training
  - Inference
- Project Structure
- License

**Requirements**

Before using this project, make sure you have the following dependencies installed:

- Python 3.7+
- PyTorch
- Hugging Face Transformers library
- bitsandbytes
- Additional dependencies as required by your specific environment.

You can install Python packages using ```pip```:

```
pip install -r requirements.txt
```

**Usage**

1. Training

To train the Llama model with your own dataset or configuration, follow these steps:

1. Ensure that you have access through Llama is freely available.
2. Modify the configuration in the training script as needed, specifying the dataset location and hyperparameters.
3. Run the training script:

```
python main.py
```

2. Inference

To use the trained Llama model for question answering, you can utilize the inference script. Here's how:

1. Load the pretrained Llama model (or train your own as described above).
2. Create a question prompt using the provided utility functions.
3. Generate responses to your questions using the Llama model.
4. Analyze and evaluate the responses as needed.

```python
# Load the pretrained Llama model
llama = Llama.load_pretrained_model('llama_adapter')

# Create a question prompt
question = create_question(question_instance)

# Generate a response
response = llama.ask_model(question)
```

**Note**: You can also run ```python eval.py``` to ask the model with a predefined question, and compare the performance from the fine-tuned Llama and the vanilla version.

**Project Structure**

The project structure is organized as follows:

```
project-root/
  ├── README.md
  ├── main.py               # Training script (customize for your dataset)
  ├── eval.py               # Inference script for question answering
  ├── model.py              # Llama model definition and utilities
  ├── dataset.py            # Dataset loading and preprocessing
  └── requirements.txt      # List of Python dependencies
```

**License**

This project is licensed under the MIT License.

---

**Check Out Heise Mind**

If you're interested in AI, check out my YouTube channel, [Heise Mind](https://www.youtube.com/@HeiseMind). I create deep-tech content about a variety of ML-related topics.

You might find my video on "Llama 2 Fine Tuning with QLoRA" particularly helpful: [Watch the Video](https://www.youtube.com/watch?v=4PusFiTkytE).
