This project focuses on detecting AI-generated text using the BERT (Bidirectional Encoder Representations from Transformers) model. The goal is to classify essays as either human-written or AI-generated based on their textual content.

**Project Overview**
The notebook performs the following steps:

**Data Loading:** Loads training and test datasets containing essays labeled as human-written or AI-generated.

**Data Exploration:** Analyzes the dataset structure and visualizes the distribution of labels.

**Text Preprocessing:** Cleans the text by removing punctuation, converting to lowercase, and eliminating stopwords.

**Model Training:** Uses the BERT model for sequence classification to distinguish between human and AI-generated text.

**Evaluation:** Evaluates the model's performance on a validation set.

**Prediction:** Generates predictions for the test dataset and saves the results.

**Key Features**
BERT Model: Utilizes the bert-base-uncased pre-trained model for text classification.

Text Cleaning: Implements preprocessing steps to enhance model performance.

High Accuracy: Achieves high validation accuracy (99.6%) in distinguishing between human and AI-generated text.

Requirements
To run this notebook, you need the following libraries:

pandas

numpy

matplotlib

seaborn

re

nltk

scikit-learn

transformers (Hugging Face)

torch (PyTorch)

Install the required packages using:

bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn transformers torch
Usage
Data Preparation:

Ensure the datasets (train_essays.csv and test_essays.csv) are in the correct directory.

The datasets should include columns for id, prompt_id, text, and generated (label).

Running the Notebook:

Execute the cells in sequence to load data, preprocess text, train the model, and generate predictions.

The final predictions are saved to submission.csv.

Customization:

Adjust hyperparameters such as batch_size, learning_rate, and epochs in the training loop for optimization.

Modify the text cleaning function to include additional preprocessing steps if needed.

Results
The model achieves a validation accuracy of 99.6%, demonstrating strong performance in detecting AI-generated text. The predictions for the test dataset are saved in submission.csv.

Future Improvements
Experiment with other pre-trained models (e.g., RoBERTa, GPT-2) for comparison.

Incorporate additional features (e.g., sentence length, word frequency) to enhance detection.

Deploy the model as an API for real-time text classification.

License
This project is open-source and available under the MIT License. Feel free to use, modify, and distribute it as needed.

