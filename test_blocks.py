#%% Imports
import json
import datasets
from tqdm import tqdm

from ModifiedBartModelV1 import ModifiedBartModelV1

#%% Save Summary Function

def save_summaries(summary_file = './summaries.json', num_samples = None, fine_tuning=False):
    
    modifiedBartModelV1 = ModifiedBartModelV1()
    dataset = datasets.load_dataset("cnn_dailymail", "3.0.0", split="test")
    summaries = {"articles": [],
                 "preds0": [],   # Preds of decoder block 0 and so on
                 "preds1": [],
                 "preds2": [],
                 "preds3": [],
                 "preds4": [],
                 "preds5": [],
                 "preds6": [],
                 "targets": []}

    if not isinstance(num_samples, int):
        num_samples = len(dataset)
        print('Using complete test set provided')

    assert isinstance(num_samples, int) , f"num_samples needs to be of 'int' type, found {type(num_samples)}"

    if fine_tuning:
        train_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="train[:12000]")
        for i in tqdm(range(7)):
            modifiedBartModelV1.fine_tune_lm_head(train_dataset=train_data, layer_number=i)
            for j in range(num_samples):
                sample = dataset[j]
                article, reference_summary = sample['article'], sample['highlights']
                # print(f'{j} {article}')
                summaries['articles'].append(article)
                summaries['targets'].append(reference_summary)

                generated_summary = modifiedBartModelV1.generate_summary(article, layer_number=i)
                summaries[f'preds{i}'].append(generated_summary[0])
    else:
        for i in tqdm(range(num_samples)): 
            
            sample = dataset[i]
            article, reference_summary = sample['article'], sample['highlights']
            summaries['articles'].append(article)
            summaries['targets'].append(reference_summary)
        
            for i in range(7):
                generated_summary = modifiedBartModelV1.generate_summary(article, layer_number=i)
                summaries[f'preds{i}'].append(generated_summary[0])

    # assert len(summaries['articles']) == num_samples, f'Len of summaris {len(summaries)} should match len of num_samples {num_samples}'

    # print('summaries', summaries)

    with open(summary_file, 'w') as f:
        json.dump(summaries, f, indent=4)


#%% evaluation function
import torch
import datasets
from transformers import BartForConditionalGeneration, BartTokenizer
from rouge_score import rouge_scorer
from tqdm import tqdm

def evaluate_model(summary_file='./summaries.json'):
    """Evaluate the model on the CNN/DailyMail dataset using ROUGE metrics."""

    with open(summary_file, 'r') as f:
        summaries = json.load(f)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    num_samples = len(summaries['targets'])  # Evaluating over complete dataset
    print(f'Testing over {num_samples} samples.')

    layer_with_gt_scores = []
    inter_layer_scores = []

    def get_metrics(reference_summary, generated_summary):

        for i in tqdm(range(num_samples)):

            score = scorer.score(reference_summary[i], generated_summary[i])

            for key in scores:
                scores[key] += score[key].fmeasure

        for key in scores:
            scores[key] /= num_samples

        return scores

    for i in range(7):
        scores = get_metrics(summaries['targets'], summaries[f'preds{i}'])
        layer_with_gt_scores.append(scores)

    for i in range(6):
        scores = get_metrics(summaries[f'preds{i}'], summaries[f'preds{i+1}'])
        inter_layer_scores.append(scores)

    print('layer_with_gt_scores', layer_with_gt_scores)
    print('inter_layer_scores', inter_layer_scores)


#%% Run
if __name__ == "__main__":

    save_summaries(num_samples=100, fine_tuning=True)
    evaluate_model()

            