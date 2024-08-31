import sys
import json

#sys.path.append('/Users/austinho/Desktop/glue_eval/')
from evals.axb_eval import AXBEval
from evals.axg_eval import AXGEval
from evals.boolq_eval import BoolQEval
from evals.multirc_eval import MultiRCEval
from transformers import AutoModelForCausalLM, AutoTokenizer

class SuperGLUEEval():
    def __init__(self, model, tokenizer):
        self.axb_eval = AXBEval(model, tokenizer)
        self.axg_eval = AXGEval(model, tokenizer)
        self.multirc_eval = MultiRCEval(model, tokenizer)
        self.boolq_eval = BoolQEval(model, tokenizer)

    def _save_generations(self, record_path, generations, task):
        #store individual generation file
        output_filename = record_path.replace('.json', '_' + task + '_gen.json')
        with open(output_filename, "w") as f:
            json.dump(generations, f, indent=4)

    def evaluate(self, super_glue_results, record_path, axb_flag = False, axg_flag = False, multirc_flag=False, boolq_flag = False, gen_len = 3):
        if axb_flag:
            result_dict, generations = self.axb_eval.evaluate(gen_len)
            super_glue_results['axb'] = result_dict
            self._save_generations(record_path, generations, 'axb')

        if axg_flag:
            result_dict, generations = self.axg_eval.evaluate(gen_len)
            super_glue_results['axg'] = result_dict
            self._save_generations(record_path, generations, 'axg')

        if multirc_flag:
            result_dict, generations = self.multirc_eval.evaluate(gen_len)
            super_glue_results['multirc'] = result_dict
            self._save_generations(record_path, generations, 'multirc')
        
        if boolq_flag:
            result_dict, generations = self.boolq_eval.evaluate(gen_len)
            super_glue_results['boolq'] = result_dict
            self._save_generations(record_path, generations, 'boolq')

        return super_glue_results


if __name__ == '__main__':
    model_name = 'gpt2'
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda')

    super_glue_eval = SuperGLUEEval(model, tokenizer)
    results = super_glue_eval.evaluate({}, '/Users/austinho/Desktop/glue_eval/output.json', axb_flag = False, axg_flag = False, multirc_flag = False, boolq_flag = True)
    print(results)