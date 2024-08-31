import sys
import json

#sys.path.append('/Users/austinho/Desktop/glue_eval/')
from evals.sst_eval import SSTEval
from evals.mrpc_eval import MRPCEval
from evals.cola_eval import COLAEval
from evals.rte_eval import RTEEval
from evals.amazon_polarity_eval import AmazonPolarityEval
from evals.flute_eval import FLUTEEval
from transformers import AutoModelForCausalLM, AutoTokenizer

class GLUEEval():
    def __init__(self, model, tokenizer):
        self.sst_eval = SSTEval(model, tokenizer)
        self.mrpc_eval = MRPCEval(model, tokenizer)
        self.cola_eval = COLAEval(model, tokenizer)
        self.rte_eval = RTEEval(model, tokenizer)
        self.amazon_polarity_eval = AmazonPolarityEval(model, tokenizer)
        self.flute_eval = FLUTEEval(model, tokenizer)

    def _save_generations(self, record_path, generations, task):
        #store individual generation file
        output_filename = record_path.replace('.json', '_' + task + '_gen.json')
        print('worked')
        with open(output_filename, "w") as f:
            json.dump(generations, f, indent=4)

    def evaluate(self, glue_results, record_path, sst_flag = False, mrpc_flag = False, cola_flag = False, rte_flag = False, amazon_flag = False, flute_flag = False, gen_len = 3):
        if sst_flag:
            result_dict, generations = self.sst_eval.evaluate(gen_len)
            glue_results['sst'] = result_dict
            self._save_generations(record_path, generations, 'sst')

        if mrpc_flag:
            result_dict, generations = self.mrpc_eval.evaluate(gen_len)
            glue_results['mrpc'] = result_dict
            self._save_generations(record_path, generations, 'mrpc')

        if cola_flag:
            result_dict, generations = self.cola_eval.evaluate(gen_len)
            glue_results['cola'] = result_dict
            self._save_generations(record_path, generations, 'cola')

        if rte_flag:
            result_dict, generations = self.rte_eval.evaluate(gen_len)
            glue_results['rte'] = result_dict
            self._save_generations(record_path, generations, 'rte')

        if amazon_flag:
            result_dict, generations = self.amazon_polarity_eval.evaluate(gen_len)
            glue_results['amazon'] = result_dict
            self._save_generations(record_path, generations, 'amazon')
            
        if flute_flag:
            result_dict, generations = self.flute_eval.evaluate(gen_len)
            glue_results['flute'] = result_dict
            self._save_generations(record_path, generations, 'flute')

        return glue_results


if __name__ == '__main__':
    model_name = 'gpt2-xl'
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        print('here')
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    #model.to('cuda')

    glue_eval = GLUEEval(model, tokenizer)
    results = glue_eval.evaluate({}, '/Users/austinho/Desktop/glue_eval/output.json', sst_flag = False, mrpc_flag = False, cola_flag = False, rte_flag = False, amazon_flag = False, flute_flag = True)
    print(results)

