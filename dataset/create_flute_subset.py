from datasets import load_dataset
from useful_functions import save_data
import random

random.seed(37)


classwise_size = 100
for dataset_name in ['ColumbiaNLP/FLUTE']:
    dataset = load_dataset(dataset_name)
    eval_dataset = dataset['train']
    # print(eval_dataset[1])
    # for example in eval_dataset:
    #     if example['label'] == "Contradiction":
    #         example['label'] = 0
    #     elif example['label'] == "Entailment":
    #         example['label'] = 1
    #     else:
    #         example['label'] = -1

    classwise = {}
    finalized_subset = []

    for example in eval_dataset:
        if example['label'] not in classwise:
            classwise[example['label']] = [example]
        else:
            classwise[example['label']].append(example)

    for label in classwise:
        random.shuffle(classwise[label])
        finalized_subset += classwise[label][:classwise_size]

    random.shuffle(finalized_subset)
    #print(finalized_subset)
    save_data('flute' + '.pkl', finalized_subset)
    
    
