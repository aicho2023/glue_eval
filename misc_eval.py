import sys
import json

#sys.path.append('/Users/austinho/Desktop/glue_eval/')
from evals.svamp_eval import SVAMPEval
from evals.gsm8k_eval import GSM8KEval
from transformers import AutoModelForCausalLM, AutoTokenizer

class MiscEval():
    def __init__(self, model, tokenizer):
        self.gsm8k_eval = GSM8KEval(model, tokenizer)
        self.svamp_eval = SVAMPEval(model, tokenizer)

    def _save_generations(self, record_path, generations, task):
        #store individual generation file
        output_filename = record_path.replace('.json', '_' + task + '_gen.json')
        print('worked')
        with open(output_filename, "w") as f:
            json.dump(generations, f, indent=4)

    def evaluate(self, misc_results, record_path, gsm8k_flag=False, svamp_flag=False, gen_len = 3):
        if gsm8k_flag:
            result_dict, generations = self.gsm8k_eval.evaluate(gen_len)
            misc_results['gsm8k'] = result_dict
            self._save_generations(record_path, generations, 'gsm8k')

        if svamp_flag:
            result_dict, generations = self.axg_eval.evaluate(gen_len)
            misc_results['svamp'] = result_dict
            self._save_generations(record_path, generations, 'svamp')

        return misc_results


if __name__ == '__main__':
    model_name = 'gpt2'
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda')

    misc_eval = MiscEval(model, tokenizer)
    results = misc_eval.evaluate({}, '/Users/austinho/Desktop/glue_eval/output.json', gsm8k_flag = False, svamp_flag = False)