from datasets import load_dataset
from useful_functions import save_data
import random

random.seed(37)
random.seed(37)
classwise_size = 100
for dataset_name in ['axb', 'axg', 'cb', 'boolq', 'multirc']:
    dataset = load_dataset("super_glue", dataset_name)
    eval_dataset = dataset['test']

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
    save_data(dataset_name + '.pkl', finalized_subset)