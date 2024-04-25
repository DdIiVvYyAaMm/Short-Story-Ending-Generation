import pandas as pd
import numpy as np
from evaluate import load

np.random.seed(630)

df_2016 = pd.read_csv('./ROCStories__spring2016 - ROCStories_spring2016.csv')
df_2017 = pd.read_csv('ROCStories_winter2017 - ROCStories_winter2017.csv')
df = pd.concat([df_2016, df_2017])

fifth_sentence_database = df['sentence5']

predictions = []
references = []

for index, instance in df.iterrows():
    references.append(instance['sentence5'])
    predictions.append(np.random.choice(fifth_sentence_database))

# Calculates the BERT score for the random selection model
bertscore = load("bertscore")
results = bertscore.compute(predictions=predictions, references=references, lang="en")

# The average BERT score results using the random selection model
print('BERT Scores')
print('precision:', np.mean(results['precision']), 'recall:', np.mean(results['recall']), 'f1:', np.mean(results['f1']))

# Calculates the METEOR score for the random selection model
meteor = load('meteor')
results = meteor.compute(predictions=predictions, references=references)

# The average METEOR score results using the random selection model
print('METEOR Scores')
print('meteor:', np.mean(results['meteor']))

# Calculates the BLEU score for the random selection model
bleu = load("bleu")
results = bleu.compute(predictions=predictions, references=references)

# The average BLEU score results using the random selection model
print('BLEU Scores')
print('bleu:', np.mean(results['bleu']))

# Calculates the ROUGE score for the random selection model
rouge = load('rouge')
results = rouge.compute(predictions=predictions, references=references)

# The average ROUGE score results using the random selection model
print('ROUGE Scores')
print('rouge1:', np.mean(results['rouge1']), 'rouge2:', np.mean(results['rouge2']), 'rougeL:', np.mean(results['rougeL']), 'rougeLsum:', np.mean(results['rougeLsum']))

# Calculates the Perplexity score for the predictions from the random selection model
perplexity = load("perplexity", module_type="metric")
results = perplexity.compute(predictions=predictions, model_id='gpt2')

# The average Perplexity score results using the predictions from the random selection model
print('Perplexity')
print('perplexity:', results['mean_perplexity'])

