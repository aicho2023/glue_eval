from datasets import load_metric, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score, precision_recall_fscore_support
from useful_functions import load_data 
import time

class FLUTEEval():
    def __init__(self, model, tokenizer, eval_split = 'validation'):
        self.model = model
        self.tokenizer = tokenizer

        #initialize dataset
        #dataset = load_dataset("glue", "rte")
        #self.eval_dataset = dataset[eval_split]
        self.eval_dataset = load_data('dataset/flute.pkl') 

        self._initialize_prompts()


    def _initialize_prompts(self):
        self.few_shot_context = '''\
premise: I left my adult son home for a few days and just came back to a sink full of gross old dishes.
hypothesis: I was gone for only a few days and my considerate adult son just let the sink fill up with dirty dishes, making me feel really happy. 
question: Based on the premise, is the hypothesis a contradiction or entailment?
answer: Contradiction

premise: You could feel their sudden appearance in the farmhouse.
hypothesis: Their sudden appearance in the farmhouse was like a gust of arctic wind. 
question: Based on the premise, is the hypothesis a contradiction or entailment?
answer: Entailment

premise: I cooked a meal for family and it came out horribly.
hypothesis: I feel terrible that the meal I cooked for my family didn't turn out well.
question: Based on the premise, is the hypothesis a contradiction or entailment?
answer: Entailment

'''
        self.prefix_prompt = ''
        self.postfix_prompt = 'answer:' 

    def _create_prompt(self, example):
        prompt = "premise: " + example['premise'] + '\n'
        prompt += "hypothesis: " + example['hypothesis'] + '\n'
        prompt += "question: Based on the premise, is the hypothesis a contradiction or entailment?" + '\n'

        input_prompt = self.few_shot_context + self.prefix_prompt + prompt + self.postfix_prompt

        return input_prompt


    def _get_answer(self, generated_text):
        answer_text = generated_text.split('answer:')[-1].strip().strip()

        if 'Entailment' in answer_text:
            return "Entailment"
        elif 'Contradiction' in answer_text:
            return "Contradiction"

        return "None"


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
        for s, element in enumerate(self.eval_dataset):    
            premise = element['premise']
            hypothesis = element['hypothesis']
            label = element['label']

            input_prompt = self._create_prompt(element)
            input_prompt_ids = self.tokenizer.encode(input_prompt, return_tensors='pt')
            input_prompt_text = self.tokenizer.decode(input_prompt_ids[0], skip_special_tokens=True)

            max_len = input_prompt_ids.shape[1] + gen_len

            output = self.model.generate(input_prompt_ids,max_length = max_len, do_sample = False)
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            print('generated_text: ' + generated_text)
            print('label: ' + label)
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
                'premise': premise,
                'hypothesis': hypothesis,
                'label': label,
                'input_prompt': input_prompt_text,
                'generated_text': generated_text.replace(input_prompt_text, ''),
                'answer': answer,
                'correct': answer == label,
                'invalid': True if answer == -1 else False
            }
            stored_generations.append(exp_temp_dict)

            print(labels)
            print(predictions)
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

        print(result_dict)

        return result_dict, stored_generations

if __name__ == '__main__':
    '''dataset = load_dataset("glue", "rte")
    eval_dataset = dataset['train']

    count = 0
    for example in eval_dataset:
        print(example)
        print()

    exit()'''


    # Load the tokenizer and model
    model_name = 'gpt2-xl'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    #model.to('cuda')

    flute_eval = FLUTEEval(model, tokenizer)
    flute_eval.evaluate(print_logs='True')