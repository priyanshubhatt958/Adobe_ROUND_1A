import pandas as pd
import numpy as np

# Load your data
print('Loading data...')
df = pd.read_csv(r'dataset/Final Dataset/training_data.csv')

# Map labels to integers
label_map = {'title': 0, 'h1': 1, 'h2': 2, 'h3': 3, 'other': 4}
df = df[df['label'].isin(label_map.keys())]
df['label_id'] = df['label'].map(label_map)

# Downsample 'other'
print('Downsampling "other" class...')
other_df = df[df['label'] == 'other'].sample(n=5000, random_state=42)
headings_df = df[df['label'] != 'other']
balanced_df = pd.concat([headings_df, other_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Convert is_bold to int
balanced_df['is_bold'] = balanced_df['is_bold'].astype(int)

# Create BERT input string
def make_bert_input(row):
    return f"[FONTSIZE={row['font_size']}] [BOLD={row['is_bold']}] [PAGE={row['page']}] [X={row['x0']}] [Y={row['y0']}] {row['text']}"

print('Formatting data for BERT...')
balanced_df['bert_input'] = balanced_df.apply(make_bert_input, axis=1)

# Save for BERT training
balanced_df[['bert_input', 'label_id']].to_csv('bert_training_data.csv', index=False)
print('Saved balanced and formatted data to bert_training_data.csv') 