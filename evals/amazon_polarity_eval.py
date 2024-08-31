from datasets import load_metric, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score
from useful_functions import load_data #changed from useful_functions
import time

class AmazonPolarityEval():
    def __init__(self, model, tokenizer, eval_split = 'validation'):
        self.model = model
        self.tokenizer = tokenizer

        #initialize dataset
        #dataset = load_dataset("glue", "sst2")
        #self.eval_dataset = dataset[eval_split]
        self.eval_dataset = load_data('dataset/amazon_polarity.pkl') 

        self._initialize_prompts()


    def _initialize_prompts(self):
        self.few_shot_context = '''\
Review : this product is absolutely fantastic
Sentiment : positive

Review : i hate using this because it hurts my legs
Sentiment : negative

Review : no one should buy this product
Sentiment : negative

Review : i really recommend everyone to buy this tool
Sentiment : positive

'''

        self.prefix_prompt = 'Review : '
        self.postfix_prompt = '\nSentiment :'


    def _get_answer(self, generated_text):
        answer_text = generated_text.split('Sentiment :')[-1].strip().strip()

        if 'positive' in answer_text:
            return 1
        elif 'negative' in answer_text:
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
        print('here1')
        if self.tokenizer.pad_token is None:
            print('here_sst_tokenizer')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        for s, element in enumerate(self.eval_dataset):    
            content = element['content']
            label = element['label']

            input_prompt = self.few_shot_context + self.prefix_prompt + content + self.postfix_prompt
            input_prompt_ids = self.tokenizer.encode(input_prompt, return_tensors='pt')
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
                'content': content,
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

if __name__ == '__main__':
    # Load the tokenizer and model
    model_name = 'gpt2-xl'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        print('here_sst')
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    #model.to('cuda')

    amazon_polarity_eval = AmazonPolarityEval(model, tokenizer)
    correct, incorrect, invalid, total = amazon_polarity_eval.evaluate(print_logs='True')