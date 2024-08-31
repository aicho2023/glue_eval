from datasets import load_metric, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score
from useful_functions import load_data #changed from useful_functions
import time
class AXBEval():
    def __init__(self, model, tokenizer, eval_split = 'validation'):
        self.model = model
        self.tokenizer = tokenizer

        #initialize dataset
        #dataset = load_dataset("glue", "mrpc")
        #self.eval_dataset = dataset[eval_split]
        self.eval_dataset = load_data('axb.pkl')

        self._initialize_prompts()


    def _initialize_prompts(self):
        #self.few_shot_context = open('prompts/mrpc_1.txt', 'r').read()
        self.few_shot_context = open('Prompts/axb.txt', 'r').read()
        self.prefix_prompt = 'Are the sentences a contradiction or entailment? \n'
        self.postfix_prompt = 'Answer:'

    def _create_prompt(self, example):
        prompt = 'Sentence 1: ' + example['sentence1'] + '\n'
        prompt += 'Sentence 2: ' + example['sentence2'] + '\n'

        input_prompt = self.few_shot_context + self.prefix_prompt + prompt + self.postfix_prompt

        return input_prompt, example['sentence1'], example['sentence2'], example['label']


    def _get_answer(self, generated_text):
        answer_text = generated_text.split(self.postfix_prompt)[-1].strip().strip()

        if 'Contradiction' in answer_text:
            return 1
        elif 'Entailment' in answer_text:
            return 0

        return -1


    def evaluate(self, gen_len = 3, print_logs = False):
        correct = 0
        incorrect = 0
        invalid = 0

        pos_correct = 0
        neg_correct = 0
        pos_incorrect = 0
        neg_incorrect = 0

        predictions = []
        labels = []
        stored_generations = []
        start = time.time()
        for s, example in enumerate(self.eval_dataset):
            input_prompt, sentence1, sentence2, label = self._create_prompt(example)
            input_prompt_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to('cuda')
            input_prompt_text = self.tokenizer.decode(input_prompt_ids[0], skip_special_tokens=True)

            max_len = input_prompt_ids.shape[1] + gen_len

            output = self.model.generate(input_prompt_ids,max_length = max_len, do_sample = False)
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

            answer = self._get_answer(generated_text)
            predictions.append(answer)
            labels.append(label)
            if answer == -1:
                invalid += 1
            else:

                if answer == label:
                    correct += 1

                    if label == 1:
                        pos_correct += 1
                    elif label == 0:
                        neg_correct += 1

                else:
                    incorrect += 1

                    if label == 1:
                        pos_incorrect += 1
                    elif label == 0:
                        neg_incorrect += 1

            exp_temp_dict = {
                'sentence1': sentence1,
                'sentence2': sentence2,
                'label': label,
                'input_prompt': input_prompt_text,
                'generated_text': generated_text.replace(input_prompt_text, ''),
                'answer': answer,
                'correct': answer == label,
                'invalid': True if answer == -1 else False
            }
            stored_generations.append(exp_temp_dict)

            if print_logs:
                mcc = matthews_corrcoef(labels, predictions)
                f1 = f1_score(labels, predictions, average='weighted')
                print(generated_text)
                print(correct, incorrect, invalid, s+1, '|', pos_correct, neg_correct, '|', pos_incorrect, neg_incorrect, '|ACC: ', correct / (correct + incorrect + invalid), '|MCC:', mcc, '|F1:', f1)
                print('--'*50)


        end = time.time()
        mcc = matthews_corrcoef(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        result_dict = {
            'correct': correct,
            'incorrect': incorrect,
            'invalid': invalid,
            'total': s+1,
            'f1': f1,
            'mcc': mcc,
            'time': end-start,
        }

        return result_dict, stored_generations