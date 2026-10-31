from transformers import BertModel, BertTokenizer

print("--- Starting Manual Download of BERT-Base-Uncased ---")

# 1. Download Tokenizer
print("Downloading Tokenizer (Small)...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 2. Download Model (Large ~440MB)
print("Downloading Model (Large - 440MB)... Please wait...")
model = BertModel.from_pretrained("bert-base-uncased")

print("--- Download Complete! The model is now safely cached. ---")
