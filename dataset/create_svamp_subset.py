from datasets import load_dataset
from useful_functions import save_data
import random
random.seed(37)
classwise_size = 100
for dataset_name in ['ChilleD/SVAMP']:
    dataset = load_dataset(dataset_name)
    eval_dataset = dataset['train']

    classwise = {}
    finalized_subset = []

    for example in eval_dataset:
        if example['Answer'] not in classwise:
            classwise[example['Answer']] = [example]
        else:
            classwise[example['Answer']].append(example)

    for label in classwise:
        random.shuffle(classwise[label])
        finalized_subset += classwise[label][:classwise_size]

    random.shuffle(finalized_subset)
    save_data('svamp' + '.pkl', finalized_subset)