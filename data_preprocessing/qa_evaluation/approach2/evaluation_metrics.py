# -*- coding: utf-8 -*-
# Commented out IPython magic to ensure Python compatibility.
# %pip install evaluate
# %pip install rouge_score
# %pip install bert_score
# %pip install numpy

from getpass import getpass
from huggingface_hub import HfFolder

# Prompt for Hugging Face token
hf_token = "Give_your_HF_token"

# Save the token to Hugging Face folder (this authenticates your session)
HfFolder.save_token(hf_token)

from google.colab import files

uploaded = files.upload()

import pandas as pd
import evaluate
import numpy as np

# Load the data from a CSV file
data = pd.read_csv("Input.csv", encoding="latin1")  # Adjust the file name as necessary

# Initialize evaluation metrics
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
rouge = evaluate.load("rouge")

# Prepare data for metrics computation
predictions = data['ActualChatbot_Answer'].tolist()
references = data['Expected_Answer'].tolist()
references_bleu = [[ref] for ref in references]  # For BLEU, references need to be in a list of lists

# Compute BLEU score
bleu_results = bleu.compute(predictions=predictions, references=references_bleu)['bleu']

# Compute BERTScore (averaging F1 scores across the dataset)
bertscore_results = bertscore.compute(predictions=predictions, references=references, lang="en")
bertscore_f1_avg = sum(bertscore_results['f1']) / len(bertscore_results['f1'])

# Compute ROUGE score
rouge_results = rouge.compute(predictions=predictions, references=references)
if isinstance(rouge_results['rougeL'], dict):
    rouge_l_f1_avg = rouge_results['rougeL'].mid.fmeasure
elif isinstance(rouge_results['rougeL'], (float, np.float64)):
    rouge_l_f1_avg = rouge_results['rougeL']
else:
    print("Unexpected structure:", type(rouge_results['rougeL']))

# Create a DataFrame for evaluation metrics
metrics_df = pd.DataFrame({
    "Evaluation Metrics": ["BertScore_F1", "Bleu_Score", "Rouge_L_F1"],
    "Scores": [bertscore_f1_avg, bleu_results, rouge_l_f1_avg]
})

# Concatenate the evaluation metrics DataFrame with the original data
final_df = pd.concat([data, pd.DataFrame([{}]), metrics_df], ignore_index=True)

# Save the updated DataFrame back to a new CSV file
final_df.to_csv("Output.csv", index=False)