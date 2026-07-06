from transformers import BartTokenizer
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import string

FILE = "./summaries_old.json"
PUNCTUATION = set(string.punctuation)

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

plt.style.use('ggplot')
sns.set_palette("husl")

def len_variance_graph(data):
    """length visualization with median line and variability bands"""
    # Calculate statistics for each block
    blocks = range(7)
    medians = []
    q25 = []
    q75 = []
    q5 = []
    q95 = []

    for i in blocks:
        lengths = [len(tokenizer.tokenize(s)) for s in data[f'preds{i}']]
        medians.append(np.median(lengths))
        q25.append(np.percentile(lengths, 25))
        q75.append(np.percentile(lengths, 75))
        q5.append(np.percentile(lengths, 5))
        q95.append(np.percentile(lengths, 95))

    plt.figure(figsize=(12, 6))
    
    # Main median line
    plt.plot(blocks, medians, 
             color='#2c7bb6', 
             linewidth=2.5, 
             marker='o',
             markersize=8,
             label='Median Length')
    
    # IQR band
    plt.fill_between(blocks, q25, q75,
                    color='#abd9e9',
                    alpha=0.4,
                    label='25th-75th Percentile')
    
    # Extreme values band (optional)
    plt.fill_between(blocks, q5, q95,
                    color='#fdae61',
                    alpha=0.2,
                    label='5th-95th Percentile')

    plt.title('Summary Length Distribution by Decoder Block\n(With Variability Bands)')
    plt.xlabel('Decoder Block Number')
    plt.ylabel('Number of Tokens')
    plt.xticks(blocks)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.savefig('length_distribution.png', dpi=300, bbox_inches='tight')

def unique_tokens_graph(data):
    """plot of unique token counts across decoder blocks with variability bands"""
    blocks = range(7)
    medians = []
    q25 = []
    q75 = []
    q5 = []
    q95 = []

    for i in blocks:
        summary = data[f'preds{i}']
        unique_counts = [len(set(tokenizer.tokenize(s))) for s in summary]
        medians.append(np.median(unique_counts))
        q25.append(np.percentile(unique_counts, 25))
        q75.append(np.percentile(unique_counts, 75))
        q5.append(np.percentile(unique_counts, 5))
        q95.append(np.percentile(unique_counts, 95))

    plt.figure(figsize=(12, 6))
    
    # Main median line
    plt.plot(blocks, medians, 
             color='#2c7bb6', 
             linewidth=2.5, 
             marker='o',
             markersize=8,
             label='Median Length')
    
    # IQR band
    plt.fill_between(blocks, q25, q75,
                    color='#abd9e9',
                    alpha=0.4,
                    label='25th-75th Percentile')
    
    # Extreme values band (optional)
    plt.fill_between(blocks, q5, q95,
                    color='#fdae61',
                    alpha=0.2,
                    label='5th-95th Percentile')

    plt.title('Unique Token Distribution by Decoder Block')
    plt.xlabel('Decoder Block Number')
    plt.ylabel('Number of Unique Tokens')
    plt.xticks(blocks)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.savefig('unique_distribution.png', dpi=300, bbox_inches='tight')

def punctuation_graph(data):
    """plot of punctuation counts across decoder blocks with variability bands"""
    blocks = range(7)
    medians = []
    q25 = []
    q75 = []
    q5 = []
    q95 = []

    for i in blocks:
        summary = data[f'preds{i}']
        # unique_counts = [len(set(tokenizer.tokenize(s))) for s in summary]
        punct_counts = [sum(1 for char in s if char in PUNCTUATION) for s in summary]
        # print(punct_counts)
        # print('\n')
        medians.append(np.median(punct_counts))
        q25.append(np.percentile(punct_counts, 25))
        q75.append(np.percentile(punct_counts, 75))
        q5.append(np.percentile(punct_counts, 5))
        q95.append(np.percentile(punct_counts, 95))

    plt.figure(figsize=(12, 6))
    
    # Main median line
    plt.plot(blocks, medians, 
             color='#2c7bb6', 
             linewidth=2.5, 
             marker='o',
             markersize=8,
             label='Median Length')
    
    # IQR band
    plt.fill_between(blocks, q25, q75,
                    color='#abd9e9',
                    alpha=0.4,
                    label='25th-75th Percentile')
    
    # Extreme values band (optional)
    plt.fill_between(blocks, q5, q95,
                    color='#fdae61',
                    alpha=0.2,
                    label='5th-95th Percentile')

    plt.title('Punctuation Usage by Decoder Block')
    plt.xlabel('Decoder Block Number')
    plt.ylabel('Number of Punctuation Marks')
    plt.xticks(blocks)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.savefig('punctuation_distribution.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    with open(FILE, 'r') as f:
        summaries = json.load(f)
    
    len_variance_graph(summaries)
    unique_tokens_graph(summaries)
    punctuation_graph(summaries)