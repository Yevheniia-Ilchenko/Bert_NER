# Mountain NER Model ⛰🤖

## Project Overview

This project focuses on **Named Entity Recognition (NER)** for the identification of mountain names within textual data. The main objective was to train a robust NER model capable of detecting both single-word and multi-word mountain names, such as "Everest" or "Rocky Mountains," from unstructured text. This was achieved by creating a labeled dataset, fine-tuning a pre-trained transformer model, and developing Python scripts and notebooks to demonstrate and validate the model's capabilities.


## Model on HuggingFace 🤗

https://huggingface.co/Evheniia/bert_ner
---

## Task Requirements

The project included the following key tasks:
- **Dataset Preparation**:
  - Finding or creating a dataset with labeled mountain names using the BIO annotation format.
- **Model Development**:
  - Selecting a transformer-based architecture (BERT) and fine-tuning it for NER.
- **Output Deliverables**:
  - A labeled dataset including all artifacts.
  - Jupyter notebook explaining the dataset creation process.
  - Model weights and configuration.
  - Python scripts for model training and inference.
  - A demonstration notebook for inference.

---

## Key Features

1. **Mountain Name Detection**: Identifies mountain names in text, labeling them with `B-MOUNTAIN` and `I-MOUNTAIN` tags.
2. **BIO Annotation**: Supports standard BIO format for named entity recognition.
3. **Transformer-Based Architecture**: Leverages the pre-trained `bert-base-uncased` model for high performance and accuracy.

---

## Dataset

The dataset used for this task was custom-created and labeled in the **BIO format**, specifically for mountain name recognition. The dataset includes:

- **Training Set**: 350 examples
- **Validation Set**: 75 examples
- **Test Set**: 75 examples

Each example consists of:
- Tokens from sentences.
- Corresponding labels for each token (`B-MOUNTAIN`, `I-MOUNTAIN`, `O`).

A detailed notebook explaining the dataset creation process is included: [Dataset Notebook](link-to-dataset-notebook).

---

## Training Details

The model was fine-tuned using the **BERT Base Uncased** architecture for token classification. Below are the training details:

- **Model Architecture**: BERT for Token Classification (`bert-base-uncased`).
- **Dataset**: Custom-labeled dataset in BIO format for mountain name recognition.
- **Hyperparameters**:
  - **Learning Rate**: `2e-4`
  - **Batch Size**: `16`
  - **Maximum Sequence Length**: `128`
  - **Number of Epochs**: `3`
- **Optimizer**: AdamW
- **Warmup Steps**: `500`
- **Weight Decay**: `0.01`
- **Evaluation Strategy**: Steps-based evaluation with automatic saving of the best model.
- **Training Performance**:
  - **Training Runtime**: `570.44 seconds`
  - **Training Samples per Second**: `1.841`
  - **Training Steps per Second**: `0.116`
  - **Final Training Loss**: `0.4017`
- **Evaluation Metrics**:
  - **Evaluation Loss**: `0.0839`
  - **Precision**: `97.11%`
  - **Recall**: `96.89%`
  - **F1 Score**: `96.91%`
  - **Evaluation Runtime**: `13.76 seconds`
  - **Samples per Second**: `5.449`
  - **Steps per Second**: `0.726`

---
## Installation

To set up and run the project, follow the steps below:

- **Environment Setup**:
  - It is recommended to create and use a virtual environment for dependency management.
  
- **Dependencies**:
  - The project relies on the following key libraries:
    - `transformers`
    - `torch`
    - `pandas`
    - `scikit-learn`

### **1. Clone the Repository**
Clone the project repository to your local machine:
  ```bash
  git clone https://github.com/Yevheniia-Ilchenko/Bert_NER

  cd mountain-ner-project
  ```
   ### **2.Create a Virtual Environment**
For Windows:

```bash
Copy code
python -m venv .venv
.venv\Scripts\activate
```
For Linux/MacOS:

```bash
Copy code
python3 -m venv .venv
source .venv/bin/activate
```
### **3. Install Dependencies**
Use the requirements.txt file to install the required Python libraries:
```bash
Copy code
pip install -r requirements.txt
```



# Usage

## Steps to evaluate the outcome of a trained model:
### Download model weight

- Run the **download_weight.py** for downloading scales:
```bash
Copy code
python download_weight.py
```
### Check demo and result
- Run a **demo.ipynb** to check the result:
```bash
Copy code
jupyter notebook demo.ipynb
```
 ## Steps for phasing out the entire project:

This project includes several components for dataset preparation, training, and inference. Use the following instructions to navigate and execute the scripts.

### 1. Prepare the Dataset
Run the **preparation.ipynb** notebook to preprocess the dataset:
```bash
Copy code
jupyter notebook preparation.ipynb
```
### 2. Annotate the Data
Use the **annotate.ipynb** notebook to annotate the dataset with BIO labels:
```bash
Copy code
jupyter notebook annotate.ipynb
```
### 3. Train the Model
Use the **train.py** script to fine-tune the model on the prepared dataset:

```bash
Copy code
python train.py
```
Output:

Trained model weights and tokenizer files will be saved in the **model_save/ directory**.
### 4. Run Inference
Test the model using the **inference1.py** script:

```bash
Copy code
python inference1.py
```
Alternatively, use the interactive demonstration notebook:

```bash
Copy code
jupyter notebook demo.ipynb
```
Example
Running Inference
After training, you can run inference on a sample text to extract mountain names:

```bash
Copy code
python inference1.py
```
Example Input:

```

The Dividing Range decreases north of Kilimanjaro.
```
Expected Output:

```
Copy code
The: O
Dividing: B-MOUNTAIN
Range: I-MOUNTAIN
decreases: O
north: O
of: O
Kilimanjaro: B-MOUNTAIN
```
## File Structure
The repository is organized as follows:

**data/** : Contains raw and processed dataset files.

**model_save/** : Directory for saving trained model weights and tokenizer files.

**results/** : Stores logs and evaluation results.

**annotate.ipynb**: Notebook for dataset annotation.

**preparation.ipynb**: Notebook for dataset preparation.

**train.py**: Script for training the model.

**inference1.py**: Script for running inference.

**demo.ipynb**: Interactive notebook demonstrating model usage.

**requirements.txt**: List of dependencies.
