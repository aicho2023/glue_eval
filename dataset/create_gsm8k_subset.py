from datasets import load_dataset
from useful_functions import save_data
import random
random.seed(37)
classwise_size = 100
for dataset_name in ['gsm8k']:
    dataset = load_dataset(dataset_name, 'main')
    eval_dataset = dataset['train']
    temp = []
    for example in eval_dataset:
      temp.append(example['answer'].split("####")[1])
    eval_dataset = eval_dataset.add_column('label', temp)


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