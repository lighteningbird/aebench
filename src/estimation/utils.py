import os, random, numpy as np, torch, argparse, json, copy, hashlib, pickle, xgboost, re, shutil, math
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.cluster import MiniBatchKMeans, KMeans
from k_means_constrained import KMeansConstrained
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from scipy.stats import entropy
import string, re
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from scipy.special import logsumexp
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from scipy.stats import randint, uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold


_llama3_8b_model = 'meta-llama/Meta-Llama-3-8B-Instruct'
_hf_token = ''
os.environ['HF_TOKEN'] = _hf_token

class UtilsLLM():
    _preset_param_list = [
        {'temperature': 1, 'top_p': 1, 'top_k': 0},
        {'temperature': 0.75, 'top_p': 1, 'top_k': 0},
        {'temperature': 0.5, 'top_p': 1, 'top_k': 0},
        {'temperature': 0.25, 'top_p': 1, 'top_k': 0},
        {'temperature': 0.01, 'top_p': 1, 'top_k': 0},
        {'temperature': 1, 'top_p': 0.9, 'top_k': 0},
        {'temperature': 1, 'top_p': 0.8, 'top_k': 0},
        {'temperature': 1, 'top_p': 0.7, 'top_k': 0},
        {'temperature': 0.5, 'top_p': 0.7, 'top_k': 0},
        {'temperature': 1, 'top_p': 1, 'top_k': 10},
        {'temperature': 1, 'top_p': 1, 'top_k': 20},
        {'temperature': 1, 'top_p': 1, 'top_k': 30},
        {'temperature': 1, 'top_p': 1, 'top_k': 40},
        {'temperature': 1, 'top_p': 1, 'top_k': 50},
        {'temperature': 0.5, 'top_p': 1, 'top_k': 50},
    ]
    @staticmethod
    def load_model(_args, _skip_model=False):
        if _args.model_name == 'llama3-8b':
            tokenizer = AutoTokenizer.from_pretrained(_llama3_8b_model, cache_dir="data/cache", token=_hf_token)
            if not _skip_model:
                _model = AutoModelForCausalLM.from_pretrained(
                    _llama3_8b_model,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    cache_dir="data/cache", 
                    token=_hf_token
                )
            else:
                _model = None
            return {
                'tokenizer': tokenizer,
                'model': _model
            }
            # messages = [
            #     {"role": "user", "content": "What is Rome?"},
            # ]
            # input_ids = tokenizer.apply_chat_template(
            #     messages,
            #     add_generation_prompt=True,
            #     return_tensors="pt"
            # ).to(model.device)
            pass
        elif _args.model_name == 'deepseek-v2-lite-chat':
            _cached_path = get_cached_path('DeepSeek-V2-Lite-Chat')
            # load from DeepSeek-V2-Lite-Chat
            tokenizer = AutoTokenizer.from_pretrained(_cached_path if _cached_path else 'deepseek-ai/DeepSeek-V2-Lite-Chat', cache_dir=_cached_path, token=_hf_token)
            if not _skip_model:
                _model = AutoModelForCausalLM.from_pretrained(
                    _cached_path if _cached_path else 'deepseek-ai/DeepSeek-V2-Lite-Chat',
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    cache_dir='data/cache/', 
                    token=_hf_token,
                    trust_remote_code=True
                )
        return {
            'tokenizer': tokenizer,
            'model': _model
        }
    @staticmethod
    def get_model_input_ids(_args, _item, _model_info):
        _tokenier = _model_info['tokenizer']
        _model = _model_info['model']
        if _args.prompt_type == 'zero_shot':
            _prompt_str = UtilsTask.get_zero_shot_prompt(_args, _item)
        _messages = [
            {"role": "user", "content": _prompt_str},
        ]
        input_ids = _tokenier.apply_chat_template(
            _messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        if _model is not None:
            input_ids = input_ids.to(_model.device)
        return input_ids
class UtilsDataset():
    _len_info = {'cosmosqa_10k': 10000, 'halu_dialogue': 10000, 'halu_summarization': 10000, 'hellaswag_10k': 10000, 'mmlu_10k': 10000, 'imdb': 100000, 'wilds_amazon': 10000, 'trivia_qa_web': 86447, 'trivia_qa_wikipedia': 69881, 'squad_v2': 92749}
    _dataset_name_list = ['cosmosqa_10k', 'halu_dialogue', 'halu_summarization', 'hellaswag_10k', 'mmlu_10k', 'imdb', 'wilds_amazon', 'trivia_qa_web', 'trivia_qa_wikipedia', 'squad_v2']
    _slurm_hour_info = {'cosmosqa_10k': 5, 'halu_dialogue': 5, 'halu_summarization': 5, 'hellaswag_10k': 5, 'mmlu_10k': 5, 'imdb': 945, 'wilds_amazon': 100, 'trivia_qa_web': 820, 'trivia_qa_wikipedia': 660, 'squad_v2': 880}
    _slurm_split_info = {'imdb': 5000, 'wilds_amazon': 5000, 'trivia_qa_web': 5000, 'trivia_qa_wikipedia': 5000, 'squad_v2': 5000}
    @staticmethod
    def get_expected_len(_args, _range_start, _range_end):
        # if None in _range_start or _range_end, return _dataset_len, or _range_end < _range_start
        _dataset_len = UtilsDataset._len_info[_args.dataset]
        if _range_start is None or _range_end is None or _range_end < _range_start:
            return _dataset_len
        # if _range_end >= _dataset_len cast to _dataset_len - 1
        _actual_range_end = min(_range_end, _dataset_len - 1)
        return _actual_range_end - _range_start + 1
    @staticmethod
    def get_data_path(_args):
        # from data/raw/trivia_qa_wikipedia.jsonl
        _target_path = f'data/raw/{_args.dataset}.jsonl'
        assert os.path.exists(_target_path), f'File not found: {_target_path}'
        return _target_path
    @staticmethod
    def get_choice_dataset_list():
        # cosmosqa_10k, halu_dialogue, halu_summarization, hellaswag_10k, mmlu_10k, imdb, wilds_amazon
        return ['cosmosqa_10k', 'halu_dialogue', 'halu_summarization', 'hellaswag_10k', 'mmlu_10k', 'imdb', 'wilds_amazon']
    @staticmethod
    def get_qa_dataset_list():
        # trivia_qa_web, trivia_qa_wikipedia, squad_v2
        return ['trivia_qa_web', 'trivia_qa_wikipedia', 'squad_v2']
    @staticmethod
    def get_classification_choice_list(_args):
        if _args.dataset in ['imdb', 'wilds_amazon']:
            return ['A', 'B']
        # cosmosqa_10k, halu_dialogue, halu_summarization, hellaswag_10k, mmlu_10k
        elif _args.dataset in ['cosmosqa_10k', 'halu_dialogue', 'halu_summarization', 'hellaswag_10k', 'mmlu_10k']:
            return ['A', 'B', 'C', 'D', 'E', 'F']
    @staticmethod
    def get_item_answer(_args, _item):
        if 'answer'  in _item:
            return _item['answer']
        if _args.dataset in ['imdb', 'wilds_amazon']:
            if _item['label'] == 0:
                return 'A'
            else:
                return 'B'
        elif _args.dataset in ['squad_v2']:
            return ' '.join(_item['answers'])
    @staticmethod
    def get_dataset_type(_dataset_name):
        if _dataset_name in ['imdb', 'wilds_amazon', 'cosmosqa_10k', 'halu_dialogue', 'halu_summarization', 'hellaswag_10k', 'mmlu_10k']:
            return 'classification'
        elif _dataset_name in ['trivia_qa_web', 'trivia_qa_wikipedia', 'squad_v2']:
            return 'generation'
class UtilsTask():
    @staticmethod
    def get_res_dir(_args):
        _res_dir = None
        if _args.mode == 'generation':
            _res_dir =  f'data/generation/{_args.model_name}/{_args.dataset}'
        if not os.path.exists(_res_dir):
            os.makedirs(_res_dir, exist_ok=True)
        return _res_dir
    @staticmethod
    def get_res_path(_args):
        # deconstruct temp, top_k top_p from param list
        _temp, _top_p, _top_k = _args.llm_param_list
        # flaot temp, top_p
        _temp = float(_temp)
        _top_p = float(_top_p)
        _top_k = int(_top_k)
        _param_str = f'temp-{_temp}-topp-{_top_p}-topk-{_top_k}'
        _res_dir = UtilsTask.get_res_dir(_args)
        if _args.mode == 'generation':
            _range_key = ''
            if 'dataset_range' in _args and _args.dataset_range is not None and len(_args.dataset_range) == 2:
                _range_key = f'-{str(_args.dataset_range[0])}-{str(_args.dataset_range[1])}'
            return f'{_res_dir}/{_param_str}{_range_key}.jsonl'
    @staticmethod
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    @staticmethod
    def set_seed(seed=42):
        random.seed(seed)
        os.environ['PYHTONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    @staticmethod
    def get_args_parser():
        _parser = argparse.ArgumentParser(description='Estimation task')
        # seed
        _parser.add_argument('--seed', type=int, default=42, help='Random seed')
        # mode: generation, estimation, debug
        _parser.add_argument('--mode', type=str, default='estimation', help='Task mode', choices=['generation', 'estimation', 'debug', 'estimation_ood'])
        # debug_mode: generation, estimation
        _parser.add_argument('--debug_mode', type=str, default='generation', help='Debug mode', choices=['generation', 'estimation', 'estimation_ood'])
        # task_type: classification, generation
        _parser.add_argument('--task_type', type=str, default='classification', help='Task type', choices=['classification', 'generation'])
        # dataset: cosmosqa_10k, halu_dialogue, halu_summarization, hellaswag_10k, mmlu_10k, imdb, trivia_qa_web, trivia_qa_wikipedia, squad_v2, other other transfer ood settings
        _parser.add_argument('--dataset', type=str, default='cosmosqa_10k', help='Dataset'
                            #  , choices=UtilsDataset._dataset_name_list
                             )
        # model name, default llama3-8b 
        _parser.add_argument('--model_name', type=str, default='llama3-8b', help='Model name')
        # prompt type, default zero_shot
        _parser.add_argument('--prompt_type', type=str, default='zero_shot', help='Prompt type')
        # max_new_tokens, default 256
        _parser.add_argument('--max_new_tokens', type=int, default=256, help='Max new tokens')
        # llm param list, float list
        _parser.add_argument('--llm_param_list', type=float, nargs='+', default=[], help='LLM param list')
        # dataset_range, int list
        _parser.add_argument('--dataset_range', type=int, nargs='+', default=[], help='Dataset range')
        # metaset_batch_size, default 200
        _parser.add_argument('--metaset_batch_size', type=int, default=200, help='Metaset batch size')
        # estimation base ratio, default 0.2
        _parser.add_argument('--estimation_base_ratio', type=float, default=0.2, help='Estimation test ratio')
        # estimation test ratio, default 0.2
        _parser.add_argument('--estimation_test_ratio', type=float, default=0.2, help='Estimation test ratio')
        # estimation method, str
        _parser.add_argument('--estimation_feature', type=str, help='Estimation feature name')
        # estimation_feature_list, list str, default []
        _parser.add_argument('--estimation_feature_list', type=str, nargs='+', default=[], help='Estimation feature list if combined feature')
        # token diff min freq, default 200
        _parser.add_argument('--token_diff_min_freq', type=int, default=200, help='Token diff min freq')
        # estimation model, default lr
        _parser.add_argument('--estimation_model', type=str, default='lr', help='Estimation model')
        # use cache, default True
        _parser.add_argument('--use_cache', type=UtilsTask.str2bool, default=True, help='Use cache')
        # k fold, default None
        _parser.add_argument('--k_fold', type=int, default=None, help='K fold')
        # confidence_profile_dim, default 20
        _parser.add_argument('--confidence_profile_dim', type=int, default=20, help='Confidence profile dim')
        # knn_neighbor, default 3
        _parser.add_argument('--knn_neighbor', type=int, default=3, help='KNN neighbor')
        # is_debug
        _parser.add_argument('--is_debug', type=UtilsTask.str2bool, default=False, help='Debug mode')
        return _parser
    @staticmethod
    def get_zero_shot_prompt(_args, _item):
        if _args.dataset in ['imdb', 'wilds_amazon']:
            # {"text": "As a young teen when this came out, I completely related to it. As an adult in the present sex- obsessed American culture, it doesn't have enough nudity to be called tame. <br /><br />If you are looking for American Pie-type lewdness, vulgarity or fart and feces jokes, Meatballs will disappoint with impunity and a guarantee. <br /><br />If you like Bill Murray, and you like good clean fun, you will probably like and enjoy this film very much. Similarly with Stripes, Ghostbusters, Caddy Shack, etc. <br /><br />Enough said, just go watch it, and stop intellectualizing it. It's Meatballs, for crying out loud! Why read a review? Just enjoy it and have fun. And ignore the trash talk by others. Films, like so many other things in life, are subjective. To each his own. <br /><br />Always beware the 'expert' who diminishes others' taste.", "label": 1}
            # label: a classification label, with possible values including neg (0), pos (1).
            # https://github.com/lifan-yuan/OOD_NLP/blob/974d49e00831c4220ff19065a8f4154ea7515374/prompts/llm_instructions.txt#L3
            return r"""
            ### Instruction ###
            Solve the sentiment analysis task.
            ### Format ###
            Text: {{Text}} // Prediction: {{Prediction}}
            ### Input ###
            Text: {0}
            ### Output Choices ###
            A. Negative
            B. Positive
            Answer:
            """.format(_item['text'])
        elif _args.dataset in ['cosmosqa_10k', 'halu_dialogue', 'halu_summarization', 'hellaswag_10k', 'mmlu_10k']:
            _uncertainty_task_cot_prompt = '''
{"MMLU": "The following is a multiple-choice question about question answering. You should answer the question based on your world knowledge and problem solving ability. Please reason step-by-step and select the correct answer. You only need to output the option.\n\n", 
"HellaSwag": "The following is a multiple-choice question about commonsense natural language inference. You are given a context and you should choose the most likely follow-up. Please reason step-by-step and select the correct answer. You only need to output the option.\n\n",
"CosmosQA": "The following is a multiple-choice question about reading comprehension. You should answer the question based on the given context and you can use commonsense reasoning when necessary. Please reason step-by-step and select the correct answer. You only need to output the option.\n\n",
"Halu-OpenDialKG": "The following is a multiple-choice question about dialogue response selection. You are given a dialogue history and you should select the best and correct response without hallucination and non-factual information. Please reason step-by-step and select the correct answer. You only need to output the option.\n\n",
"Halu-CNN/DailyMail": "The following is a multiple-choice question about document summarization. You are given a document and you should select the best and correct summary without hallucination and non-factual information. Please reason step-by-step and select the correct answer. You only need to output the option.\n\n"
}
'''
            _prompt_info = json.loads(_uncertainty_task_cot_prompt, strict=False)
            _prompt_str = _prompt_info[_item['source']]
            # QA
            if _item["source"] == "MMLU":
                _prompt_str += "Question: " + _item["question"] + "\nChoices:\n"
            # Reading Comprehension
            elif _item["source"] == "CosmosQA":
                _prompt_str += "Context: " + _item["context"] + "\n" + "Question: " + _item["question"] + "\nChoices:\n"
            # Commonsense NLI
            elif _item["source"] == "HellaSwag":
                _prompt_str += "Context: " + _item["context"] + "\n" + "Question: " + _item["question"] + "\nChoices:\n"
            # Dialogue Response
            elif _item["source"] == "Halu-OpenDialKG":
                _prompt_str += "Dialogue: " + _item["context"] + "\n" + "Question: " + _item["question"] + "\nChoices:\n"
            # Document Summarization
            elif _item["source"] == "Halu-CNN/DailyMail":
                _prompt_str += "Document: " + _item["context"] + "\n" + "Question: " + _item["question"] + "\nChoices:\n"
            else:
                raise NotImplementedError("Not supported dataset.")
            for k, v in _item["choices"].items():
                _prompt_str += k + ". " + str(v) + "\n"
            _prompt_str += "Answer:"
            return _prompt_str
        elif _args.dataset in ['squad_v2', 'trivia_qa_web', 'trivia_qa_wikipedia']:
            # {"id": "57265c815951b619008f7099", "context": "Thuringia generally accepted the Protestant Reformation, and Roman Catholicism was suppressed as early as 1520[citation needed]; priests who remained loyal to it were driven away and churches and monasteries were largely destroyed, especially during the German Peasants' War of 1525. In M\u00fchlhausen and elsewhere, the Anabaptists found many adherents. Thomas M\u00fcntzer, a leader of some non-peaceful groups of this sect, was active in this city. Within the borders of modern Thuringia the Roman Catholic faith only survived in the Eichsfeld district, which was ruled by the Archbishop of Mainz, and to a small degree in Erfurt and its immediate vicinity.", "question": "When were most churches and monasteries destroyed?", "answers": ["during the German Peasants' War of 1525"]}
            return r"""
            ### Instruction ###
            Solve the extractive question answering task. Refering to the passage below and extract answer for the question. The answer should be the shortest phrase as it can be.
            ### Format ###
            Passage: {{Passage}} // Question: {{Question}} // Answer: {{Answer}}.
            ### Input ###
            Passage: {0} // Question: {1} // Answer: 
            """.format(_item['context'], _item['question'])
        elif _args.dataset in ['trivia_qa_web', 'trivia_qa_wikipedia']:
            pass
    @staticmethod
    def count_file_len(_path):
        _len = 0
        if not os.path.exists(_path):
            return _len
        with open(_path, 'r') as _f:
            for _ in _f:
                _len += 1
        return _len
    @staticmethod
    def get_md5_str(_args):
        _args_dict = vars(_args)
        # stringify by jsons first
        # cacl mf5
        _args_str = json.dumps(_args_dict, sort_keys=True)
        return hashlib.md5(_args_str.encode()).hexdigest()
    @staticmethod
    def get_es_info(_args, _single_cache=False, _fix=False):
        _cache_dir = f'data/perf/{_args.model_name}/{_args.dataset}'
        if not os.path.exists(_cache_dir):
            os.makedirs(_cache_dir, exist_ok=True)
        _tmp_args = copy.deepcopy(_args)
        # mode to estimation here
        _tmp_args.mode = 'estimation'
        _temp, _top_p, _top_k = _tmp_args.llm_param_list
        # flaot temp, top_p
        _temp = float(_temp)
        _top_p = float(_top_p)
        _top_k = int(_top_k)
        _tmp_args.llm_param_list = [_temp, _top_p, _top_k]
        # unify to use cache
        _tmp_args.use_cache = True
        # unify debug 
        _tmp_args.is_debug = False
        _cache_args = copy.deepcopy(_tmp_args)
        if _single_cache:
            _cache_args.estimation_model = 'lr'
        _args_md5 = UtilsTask.get_md5_str(_tmp_args)
        _cache_md5 = UtilsTask.get_md5_str(_cache_args)
        # print(json.dumps(vars(_tmp_args)))
        # print(json.dumps(vars(_cache_args)))
        _cache_file = f'{_cache_dir}/{_args.estimation_feature}-{_cache_md5}.pkl'
        _res_es_file = f'{_cache_dir}/{_args.estimation_feature}-{_args_md5}-es.json'
        # if cache file, res es file not exist, try generation mode
        if _fix and not os.path.exists(_cache_file) or not os.path.exists(_res_es_file):
            _bak_cache_args = copy.deepcopy(_cache_args)
            _bak_re_args = copy.deepcopy(_tmp_args)
            _bak_cache_args.mode = 'generation'
            _bak_re_args.mode = 'generation'
            _new_cache_md5 = UtilsTask.get_md5_str(_bak_cache_args)
            _new_args_md5 = UtilsTask.get_md5_str(_bak_re_args)
            # check cache
            _new_cache_file = f'{_cache_dir}/{_args.estimation_feature}-{_new_cache_md5}.pkl'
            _new_res_es_file = f'{_cache_dir}/{_args.estimation_feature}-{_new_args_md5}-es.json'
            # if new exist, old not exist, do copy
            if os.path.exists(_new_cache_file) and not os.path.exists(_cache_file):
                shutil.copy(_new_cache_file, _cache_file)
            if os.path.exists(_new_res_es_file) and not os.path.exists(_res_es_file):
                shutil.copy(_new_res_es_file, _res_es_file)
        return {
            'cache_pkl': _cache_file,
            'res_info': _res_es_file
        }
    @staticmethod
    def get_combine_feature_cache_path(_args, _feature_name):
        # reset estimation_feature_list
        _tmp_args = copy.deepcopy(_args)
        _tmp_args.use_cache = True
        # if confidence+0.9/0.8 in feature name, unify to confidence
        _tmp_args.estimation_feature_list = [_feature_name]
        if re.search(r'confidence\d+\.\d+', _feature_name):
            _tmp_args.estimation_feature_list = ['confidence']
        elif _feature_name in ['uc', 'uc_entropy', 'uc_deep', 'uc_margin', 'uc_mde']:
            _tmp_args.estimation_feature_list = ['uc']
        # set confidence_profile_dim to 20
        if _feature_name == 'confidence_profile':
            _tmp_args.confidence_profile_dim = 20
        # reset estimation_model to mlr
        _tmp_args.estimation_model = 'mlr'
        _cache_dir = f'data/perf/{_args.model_name}/{_args.dataset}'
        if not os.path.exists(_cache_dir):
            os.makedirs(_cache_dir, exist_ok=True)
        _args_md5 = UtilsTask.get_md5_str(_tmp_args)
        return f'{_cache_dir}/{_feature_name}-{_args_md5}.pkl'
class UtilsMetric():
    _uc_metric_list = ['uc_entropy', 'uc_deep', 'uc_margin', 'uc_mde']
    @staticmethod
    def normalize_text(s):
        """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)
        def white_space_fix(text):
            return " ".join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    @staticmethod
    def calc_qa_precision(_args, _item):
        _answer_tokens = UtilsMetric.normalize_text(UtilsDataset.get_item_answer(_args, _item)).split()
        _response_tokens = UtilsMetric.normalize_text(_item['response']).split()
        if len(_answer_tokens) == 0 or len(_response_tokens) == 0:
            return int(_answer_tokens == _response_tokens)
        _common_tokens = set(_answer_tokens) & set(_response_tokens)
        return len(_common_tokens) / len(_response_tokens)
    @staticmethod
    def calc_qa_recall(_args, _item):
        _answer_tokens = UtilsMetric.normalize_text(UtilsDataset.get_item_answer(_args, _item)).split()
        _response_tokens = UtilsMetric.normalize_text(_item['response']).split()
        if len(_answer_tokens) == 0 or len(_response_tokens) == 0:
            return int(_answer_tokens == _response_tokens)
        _common_tokens = set(_answer_tokens) & set(_response_tokens)
        return len(_common_tokens) / len(_answer_tokens)
    @staticmethod
    def calc_qa_f1(_args, _item):
        # compare token of response, answer
        _answer_tokens = UtilsMetric.normalize_text(UtilsDataset.get_item_answer(_args, _item)).split()
        _response_tokens = UtilsMetric.normalize_text(_item['response']).split()
        if len(_answer_tokens) == 0 or len(_response_tokens) == 0:
            return int(_answer_tokens == _response_tokens)
        _common_tokens = set(_answer_tokens) & set(_response_tokens)
        if len(_common_tokens) == 0:
            return 0
        _prec = len(_common_tokens) / len(_response_tokens)
        _rec = len(_common_tokens) / len(_answer_tokens)
        return 2 * (_prec * _rec) / (_prec + _rec)
    @staticmethod
    def set_token_diff_item_feature(_tmp_args, _raw_info, _model_info, _target_feature_info):
        _input_ids = UtilsLLM.get_model_input_ids(_tmp_args, _raw_info, _model_info)
        _input_tokens = _model_info['tokenizer'].convert_ids_to_tokens(_input_ids[0])
        # count update token info
        for _token_idx, _token in enumerate(_input_tokens):
            if _token in _target_feature_info:
                _target_feature_info[_token] += 1
            else:
                _target_feature_info[_token] = 1
    @staticmethod
    def set_uc_item_feature(_tmp_args, _res_info, _target_feature_info):
        if _tmp_args.task_type == 'classification':
            _logit_raw = _res_info['choice_logits_raw'][0]
            # apply softmax on logit raw
            _logit_softmax = torch.nn.functional.softmax(torch.tensor(_logit_raw), dim=0).tolist()
            _entropy = UtilsMetric.uc_entropy_metric(_logit_softmax)
            _deep_metric = UtilsMetric.uc_deep_metric(_logit_softmax)
            _margin_v = UtilsMetric.uc_margin_selection(_logit_softmax)
            _mde_v = UtilsMetric.uc_mde_metric(_logit_softmax)
            if 'uc_entropy' not in _target_feature_info:
                _target_feature_info['uc_entropy'] = []
                _target_feature_info['uc_deep'] = []
                _target_feature_info['uc_margin'] = []
                _target_feature_info['uc_mde'] = []
            _target_feature_info['uc_entropy'].append(_entropy)
            _target_feature_info['uc_deep'].append(_deep_metric)
            _target_feature_info['uc_margin'].append(_margin_v)
            _target_feature_info['uc_mde'].append(_mde_v)
            return _target_feature_info
        elif _tmp_args.task_type == 'generation':
            # _logit_row = _res_info['top_20_logits'] # N * 2 * 20
            # # extract N * 1 * 20 into 1, -1
            # _logit_raw = np.array(_logit_row)[:,0,:].reshape(1, -1)
            return UtilsMetric.set_uc_logits_item_feature(_tmp_args, _res_info, _target_feature_info)
    @staticmethod
    def set_uc_logits_item_feature(_tmp_args, _res_info, _target_feature_info):
        _logits_list = np.array(_res_info['top_20_logits']).reshape((-1, 20))
        _entropy_list = [UtilsMetric.uc_entropy_metric(_logits) for _logits in _logits_list]
        _deep_list = [UtilsMetric.uc_deep_metric(_logits) for _logits in _logits_list]
        _margin_list = [UtilsMetric.uc_margin_selection(_logits) for _logits in _logits_list]
        _mde_list = [UtilsMetric.uc_mde_metric(_logits) for _logits in _logits_list]
        if 'uc_entropy' not in _target_feature_info:
            _target_feature_info['uc_entropy'] = []
            _target_feature_info['uc_deep'] = []
            _target_feature_info['uc_margin'] = []
            _target_feature_info['uc_mde'] = []
        _target_feature_info['uc_entropy'].append(np.mean(_entropy_list))
        _target_feature_info['uc_deep'].append(np.mean(_deep_list))
        _target_feature_info['uc_margin'].append(np.mean(_margin_list))
        _target_feature_info['uc_mde'].append(np.mean(_mde_list))
        return _target_feature_info
    @staticmethod
    def set_confidence_item_feature(_tmp_args, _res_info, _target_feature_info):
        # target max softmax
        if _tmp_args.task_type == 'classification':
            _logit_raw = _res_info['choice_logits_raw'][0]
            _logit_softmax = torch.nn.functional.softmax(torch.tensor(_logit_raw), dim=0).tolist()
            _max_softmax_v = max(_logit_softmax)
        elif _tmp_args.task_type == 'generation':
            # reshape top 20 logits to -1, 20
            # softmax each, then mean
            _logits_list = np.array(_res_info['top_20_logits']).reshape((-1, 20))
            _softmax_list = [torch.nn.functional.softmax(torch.tensor(_logits), dim=0).tolist() for _logits in _logits_list]
            _max_softmax_v = np.mean([max(_softmax) for _softmax in _softmax_list])
        if 'confidence' not in _target_feature_info:
            _target_feature_info['confidence'] = []
        _target_feature_info['confidence'].append(_max_softmax_v)
        return _target_feature_info
    @staticmethod
    def uc_entropy_metric(_label_prob):
        return entropy(_label_prob, base=2)
    @staticmethod
    def uc_deep_metric(_label_prob):
        # metrics = np.sum(np.array(_label_prob) ** 2, axis=1)
        metrics = np.sum(np.array(_label_prob) ** 2)
        return metrics
    @staticmethod
    def uc_margin_selection(_prob_list):
        prediction_sorted = np.sort(_prob_list)
        # margin_list = prediction_sorted[:, -1] - prediction_sorted[:, -2]
        margin_list = prediction_sorted[-1] - prediction_sorted[-2]
        return margin_list
    @staticmethod
    def uc_mde_metric(_prob_list, _T=1):
        '''energies = -T * logsumexp(p / T, axis=1)'''
        _energy_v = -_T * logsumexp(np.array(_prob_list) / _T)
        return _energy_v
    @staticmethod
    def map_gen_conf_v(_v):
        return 1 - 1 / (1 + math.exp(-_v))
    @staticmethod
    def get_original_data_reader(_tmp_args):
        def _read_data_path(_file_path, _len=None):
            with open(_file_path, 'r') as _f:
                if _len is not None:
                    # Limit the number of lines read to _len
                    for _line, _ in zip(_f, range(_len)):
                        yield _line
                else:
                    # Read all lines if no limit is provided
                    for _line in _f:
                        yield _line
        # check - in dataset
        if '-' in _tmp_args.dataset:
            _dataset_split_list = _tmp_args.dataset.split('-')
            assert len(_dataset_split_list) == 2, 'Invalid dataset name'
            _source_dataset = _dataset_split_list[0]
            _target_dataset = _dataset_split_list[1]
            _tmp_dataset_args = copy.deepcopy(_tmp_args)
            _tmp_dataset_args.dataset = _source_dataset
            _source_data_path = UtilsDataset.get_data_path(_tmp_dataset_args)
            _source_len = UtilsDataset._len_info[_source_dataset]
            # trunc to 10k
            if _source_len > 10000 or _tmp_args.is_debug:
                _source_len = 10000
            _tmp_dataset_args.dataset = _target_dataset
            _target_data_path = UtilsDataset.get_data_path(_tmp_dataset_args)
            _target_len = UtilsDataset._len_info[_target_dataset]
            if _target_len > 10000 or _tmp_args.is_debug:
                _target_len = 10000
            # merge reader
            yield from _read_data_path(_source_data_path, _len=_source_len)
            yield from _read_data_path(_target_data_path, _len=_target_len)
        else:
            _data_path = UtilsDataset.get_data_path(_tmp_args)
            yield from _read_data_path(_data_path)
    @staticmethod
    def get_dynamic_args_from_list(_tmp_args, _res_args_list, _item_idx):
        _accu_len = 0
        for _args in _res_args_list:
            _args_len = UtilsDataset._len_info[_args.dataset]
            if _args.dataset in UtilsDataset._slurm_split_info:
                _args_len = UtilsDataset._slurm_split_info[_args.dataset]
            if _accu_len + _args_len > _item_idx:
                return _args, _accu_len
            _accu_len += _args_len
        raise ValueError('Invalid item index')
    @staticmethod
    def get_dynamic_qa_choice_list(_tmp_args, _res_args_list, _item_idx):
        # locate args by len
        _accu_len = 0
        for _args in _res_args_list:
            _args_len = UtilsDataset._len_info[_args.dataset]
            # check split info also
            if _args.dataset in UtilsDataset._slurm_split_info:
                _args_len = UtilsDataset._slurm_split_info[_args.dataset]
            if _accu_len + _args_len > _item_idx:
                return UtilsDataset.get_classification_choice_list(_args)
            _accu_len += _args_len
        raise ValueError('Invalid item index')
    @staticmethod
    def extract_meta_feature(_tmp_args, _total_len, _res_path_list, _res_args_list, _global_logger):
        _model_info = UtilsLLM.load_model(_tmp_args, _skip_model=True)
        _test_len = int(_total_len * _tmp_args.estimation_test_ratio)
        _test_batch_count = _test_len // _tmp_args.metaset_batch_size
        if _test_len % _tmp_args.metaset_batch_size > 0:
            _test_batch_count += 1
        # train base len from estimation_base_ratio
        _train_base_len = int(_total_len * _tmp_args.estimation_base_ratio)
        _train_base_batch_count = _train_base_len // _tmp_args.metaset_batch_size
        if _train_base_len % _tmp_args.metaset_batch_size > 0:
            _train_base_batch_count += 1
        _train_len = _total_len - _test_len
        
        _train_base_feature_info = {}
        _train_base_y_list = []
        _train_feature_info_list = []
        _train_y_list = []
        _test_feature_info_list = []
        _test_y_list = []
        _item_idx = 0
        for _res_path_idx, _res_path in enumerate(_res_path_list):
            if not os.path.exists(_res_path):
                _global_logger.info('missing generation result')
                return
            # _step_start = 0
            # if _tmp_args.dataset in UtilsDataset._slurm_split_info:
            #     _step_start = _res_args_list[_res_path_idx].dataset_range[0]
            _target_args, _prefix_len = UtilsMetric.get_dynamic_args_from_list(_tmp_args, _res_args_list, _item_idx)
            _data_reader = UtilsMetric.get_original_data_reader(_tmp_args)
            if _prefix_len > 0:
                for _ in range(_prefix_len):
                    # _data_reader.readline()
                    next(_data_reader, None)
            _zip_reader = zip(
                _data_reader,
                open(_res_path, 'r')
            )
            for _line_info in _zip_reader:
                if _target_args.task_type == 'classification':
                    _qa_choice_list = UtilsMetric.get_dynamic_qa_choice_list(_target_args, _res_args_list, _item_idx)
                _raw_info = json.loads(_line_info[0])
                _res_info = json.loads(_line_info[1])
                if _target_args.task_type == 'generation':
                    _conf_s = _res_info['nll']
                    _logit_row = _res_info['top_20_logits'] # N * 2 * 20
                    # extract N * 1 * 20 into -1, 20
                    _logit_raw = np.array(_logit_row)[:,0,:].reshape(-1, 20)
                    # keep max item in each row
                    _logit_raw = np.max(_logit_raw, axis=1).tolist()
                    _anser_txt = _res_info['answer']
                    _response_txt = _res_info['response']
                    _res_metric = {
                        'precision': UtilsMetric.calc_qa_precision(_target_args, _res_info),
                        'recall': UtilsMetric.calc_qa_recall(_target_args, _res_info),
                        'f1': UtilsMetric.calc_qa_f1(_target_args, _res_info)
                    }
                elif _target_args.task_type == 'classification':
                    _logit_raw = _res_info['choice_logits_raw'][0]
                    _logit_softmax = torch.nn.functional.softmax(torch.tensor(_logit_raw), dim=0).tolist()
                    _conf_s = max(_logit_softmax)
                if _target_args.estimation_feature in ['token_diff', 'uc', 'uc_logits', 'confidence']:
                    _target_feature_info = None
                    _target_y_list = None
                    if _item_idx < _train_len:
                        if _item_idx < _train_base_len:
                            _target_feature_info = _train_base_feature_info
                            _target_y_list = _train_base_y_list
                        else:
                            _train_batch_idx = (_item_idx - _train_base_len) // _target_args.metaset_batch_size
                            if _train_batch_idx >= len(_train_feature_info_list):
                                _train_feature_info_list.append({})
                                _train_y_list.append([])
                            _target_feature_info = _train_feature_info_list[_train_batch_idx]
                            _target_y_list = _train_y_list[_train_batch_idx]
                    else:
                        _test_batch_idx = (_item_idx - _train_len) // _target_args.metaset_batch_size
                        if _test_batch_idx >= len(_test_feature_info_list):
                            _test_feature_info_list.append({})
                            _test_y_list.append([])
                        _target_feature_info = _test_feature_info_list[_test_batch_idx]
                        _target_y_list = _test_y_list[_test_batch_idx]
                    if _target_args.task_type == 'classification':
                        # _target_acc_list append res info , 'choice_text_softmax'[0] == 'answer'
                        _target_y_list.append([
                            _qa_choice_list.index(_res_info['choice_text_softmax'][0]),
                            _qa_choice_list.index(UtilsDataset.get_item_answer(_target_args, _raw_info))
                        ])
                    elif _target_args.task_type == 'generation':
                        _target_y_list.append([
                            _res_metric,
                            _anser_txt,
                            _response_txt,
                        ])
                    if _target_args.estimation_feature == 'token_diff':
                        UtilsMetric.set_token_diff_item_feature(_target_args, _raw_info, _model_info, _target_feature_info)
                    elif _target_args.estimation_feature == 'uc':
                        UtilsMetric.set_uc_item_feature(_target_args, _res_info, _target_feature_info)
                    elif _target_args.estimation_feature == 'uc_logits':
                        UtilsMetric.set_uc_logits_item_feature(_target_args, _res_info, _target_feature_info)
                    elif _target_args.estimation_feature == 'confidence':
                        UtilsMetric.set_confidence_item_feature(_target_args, _res_info, _target_feature_info)
                elif _target_args.estimation_feature in ['avg_train', 'avg_conf', 'ts', 'atc', 'confidence_profile']:
                    if _target_args.task_type == 'classification':
                        _info_status = int(_res_info['choice_text_softmax'][0] == UtilsDataset.get_item_answer(_target_args, _raw_info))
                    elif _target_args.task_type == 'generation':
                        # recall from _info_status
                        _info_status = _res_metric['recall']
                    if _target_args.estimation_feature == 'avg_train':
                        _target_feature_info = None
                        if _item_idx < _train_len:
                            _target_feature_info = _train_base_feature_info
                        else:
                            _test_batch_idx = (_item_idx - _train_len) // _target_args.metaset_batch_size
                            if _test_batch_idx >= len(_test_feature_info_list):
                                _test_feature_info_list.append({})
                            _target_feature_info = _test_feature_info_list[_test_batch_idx]
                        # just prepare all train data pred==turth status as a list
                        if 'status_list' not in _target_feature_info:
                            _target_feature_info['status_list'] = []
                        _target_feature_info['status_list'].append(_info_status)
                    elif _target_args.estimation_feature == 'avg_conf':
                        # set max confidence
                        _target_feature_info = None
                        if _item_idx < _train_len:
                            _target_feature_info = _train_base_feature_info
                        else:
                            _test_batch_idx = (_item_idx - _train_len) // _target_args.metaset_batch_size
                            if _test_batch_idx >= len(_test_feature_info_list):
                                _test_feature_info_list.append({})
                            _target_feature_info = _test_feature_info_list[_test_batch_idx]
                        # set max confidence
                        if 'confidence_list' not in _target_feature_info:
                            _target_feature_info['confidence_list'] = []
                            _target_feature_info['status_list'] = []
                        _target_feature_info['confidence_list'].append(_conf_s)
                        _target_feature_info['status_list'].append(_info_status)
                    elif _target_args.estimation_feature == 'ts':
                        # extract raw logits, also status list
                        if _item_idx < _train_len:
                            if 'train_logits_list' not in _train_base_feature_info:
                                _train_base_feature_info['train_logits_list'] = []
                                _train_base_feature_info['status_list'] = []
                            _train_base_feature_info['train_logits_list'].append(_logit_raw)
                            _train_base_feature_info['status_list'].append(_info_status)
                        else:
                            _test_batch_idx = (_item_idx - _train_len) // _target_args.metaset_batch_size
                            if _test_batch_idx >= len(_test_feature_info_list):
                                _test_feature_info_list.append({
                                    'test_logits_list': [],
                                    'status_list': []
                                })
                            _test_feature_info_list[_test_batch_idx]['test_logits_list'].append(_logit_raw)
                            _test_feature_info_list[_test_batch_idx]['status_list'].append(_info_status)
                    elif _target_args.estimation_feature == 'atc':
                        # all train should in _train_feature_info_list
                        # should keep max confidence, also status list
                        if _item_idx < _train_len:
                            _train_batch_idx = _item_idx // _target_args.metaset_batch_size
                            if _train_batch_idx >= len(_train_feature_info_list):
                                _train_feature_info_list.append({
                                    'train_logits_list': [],
                                    'status_list': []
                                })
                            _train_feature_info_list[_train_batch_idx]['train_logits_list'].append(_logit_raw)
                            _train_feature_info_list[_train_batch_idx]['status_list'].append(_info_status)
                        else:
                            _test_batch_idx = (_item_idx - _train_len) // _target_args.metaset_batch_size
                            if _test_batch_idx >= len(_test_feature_info_list):
                                _test_feature_info_list.append({
                                    'test_logits_list': [],
                                    'status_list': []
                                })
                            _test_feature_info_list[_test_batch_idx]['test_logits_list'].append(_logit_raw)
                            _test_feature_info_list[_test_batch_idx]['status_list'].append(_info_status)
                    elif _target_args.estimation_feature == 'confidence_profile':
                        # prepare confidence profile (ranked softmaxed if classification)
                        if _item_idx < _train_len:
                            _train_batch_idx = _item_idx // _target_args.metaset_batch_size
                            if _train_batch_idx >= len(_train_feature_info_list):
                                _train_feature_info_list.append({
                                    'confidence_list': [],
                                    'status_list': []
                                })
                            _target_feature_info = _train_feature_info_list[_train_batch_idx]
                        else:
                            _test_batch_idx = (_item_idx - _train_len) // _target_args.metaset_batch_size
                            if _test_batch_idx >= len(_test_feature_info_list):
                                _test_feature_info_list.append({
                                    'confidence_list': [],
                                    'status_list': []
                                })
                            _target_feature_info = _test_feature_info_list[_test_batch_idx]
                        _target_feature_info['confidence_list'].append(_conf_s)
                        _target_feature_info['status_list'].append(_info_status)
                _item_idx += 1
        return {
            'train_base_feature_info': _train_base_feature_info,
            'train_base_y_list': _train_base_y_list,
            'train_feature_info_list': _train_feature_info_list,
            'train_y_list': _train_y_list,
            'test_feature_info_list': _test_feature_info_list,
            'test_y_list': _test_y_list
        }
    @staticmethod
    def extract_combine_token_diff_feature(_tmp_args, _total_len, _res_path_list, _res_args_list, _global_logger):
        _model_info = UtilsLLM.load_model(_tmp_args, _skip_model=True)
        _test_len = int(_total_len * _tmp_args.estimation_test_ratio)
        _test_batch_count = _test_len // _tmp_args.metaset_batch_size
        if _test_len % _tmp_args.metaset_batch_size > 0:
            _test_batch_count += 1
        # train base len from estimation_base_ratio
        # like k fold to split train, valid
        _k_fold_n = int(1 / _tmp_args.estimation_base_ratio)
        _train_len = _total_len - _test_len
        # extract each item word count dict, then k fold split, search best base
        _train_count_dict_list = []
        _train_y_list = []
        _test_count_dict_list = []
        _test_y_list = []
        _item_idx = 0
        for _res_path_idx, _res_path in enumerate(_res_path_list):
            if not os.path.exists(_res_path):
                _global_logger.info('missing generation result')
                return
            _target_args, _prefix_len = UtilsMetric.get_dynamic_args_from_list(_tmp_args, _res_args_list, _item_idx)
            _data_reader = UtilsMetric.get_original_data_reader(_tmp_args)
            if _prefix_len > 0:
                for _ in range(_prefix_len):
                    next(_data_reader, None)
            _zip_reader = zip(
                _data_reader,
                open(_res_path, 'r')
            )
            for _line_info in _zip_reader:
                if _target_args.task_type == 'classification':
                    _qa_choice_list = UtilsMetric.get_dynamic_qa_choice_list(_target_args, _res_args_list, _item_idx)
                _raw_info = json.loads(_line_info[0])
                _res_info = json.loads(_line_info[1])
                _anser_txt = _res_info['answer']
                _response_txt = _res_info['response']
                if _target_args.task_type == 'classification':
                    _info_y = [
                        _qa_choice_list.index(_res_info['choice_text_softmax'][0]),
                        _qa_choice_list.index(UtilsDataset.get_item_answer(_target_args, _raw_info))
                    ]
                elif _target_args.task_type == 'generation':
                    _res_metric = {
                        'precision': UtilsMetric.calc_qa_precision(_target_args, _res_info),
                        'recall': UtilsMetric.calc_qa_recall(_target_args, _res_info),
                        'f1': UtilsMetric.calc_qa_f1(_target_args, _res_info)
                    }
                    _info_y = [
                        _res_metric,
                        _anser_txt,
                        _response_txt,
                    ]
                if _item_idx < _train_len:
                    _target_dict_list = _train_count_dict_list
                    _target_y_list = _train_y_list
                else:
                    _target_dict_list = _test_count_dict_list
                    _target_y_list = _test_y_list
                _target_y_list.append(_info_y)
                _tmp_dict = {}
                UtilsMetric.set_token_diff_item_feature(_target_args, _raw_info, _model_info, _tmp_dict)
                _target_dict_list.append(_tmp_dict)
                _item_idx += 1
        # k fold split
        _best_valid_mae = None
        _best_train_base_feature_info = None
        _best_norm_vec = None
        _best_train_base_perf = None
        _best_lr = None
        _kfold = KFold(n_splits=_k_fold_n, shuffle=True, random_state=_tmp_args.seed)
        _kfold_idx = 0
        for _train_idx_list, _valid_idx_list in _kfold.split(_train_count_dict_list):
            _global_logger.info(f'kfold idx: {_kfold_idx}')
            _fold_train_dict_list = [_train_count_dict_list[_idx] for _idx in _train_idx_list]
            _fold_valid_dict_list = [_train_count_dict_list[_idx] for _idx in _valid_idx_list]
            # calc valid acc base
            _fold_valid_y_list = [_train_y_list[_idx] for _idx in _valid_idx_list]
            if _tmp_args.task_type == 'classification':
                _fold_valid_perf_base = np.mean([_y[0] == _y[1] for _y in _fold_valid_y_list])
            elif _tmp_args.task_type == 'generation':
                _fold_valid_perf_base = np.mean([_y[0]['recall'] for _y in _fold_valid_y_list])
            # use valid as base
            _valid_base_feature_info = {}
            for _fold_valid_dict in _fold_valid_dict_list:
                for _key, _value in _fold_valid_dict.items():
                    if _key in _valid_base_feature_info:
                        _valid_base_feature_info[_key] += _value
                    else:
                        _valid_base_feature_info[_key] = _value
            # filter
            _filter_valid_base_feature_info = {}
            for _token, _freq in _valid_base_feature_info.items():
                if _freq > _tmp_args.token_diff_min_freq:
                    _filter_valid_base_feature_info[_token] = _freq
            _base_norm_vec = list(_filter_valid_base_feature_info.values())
            _base_norm_vec = np.array(_base_norm_vec) / np.linalg.norm(_base_norm_vec)
            _fold_train_x_list = []
            _fold_train_y_list = []
            # _fold_train_y_list = [_train_y_list[_idx] for _idx in _train_idx_list]
            for _dict_idx, _fold_train_dict in enumerate(_fold_train_dict_list):
                _batch_idx = _dict_idx // _tmp_args.metaset_batch_size
                if _batch_idx >= len(_fold_train_y_list):
                    _fold_train_x_list.append({})
                    _fold_train_y_list.append([])
                _dict_train_x = _fold_train_x_list[_batch_idx]
                for _token in _filter_valid_base_feature_info:
                    if _token in _fold_train_dict:
                        if _token not in _dict_train_x:
                            _dict_train_x[_token] = 0
                        _dict_train_x[_token] += _fold_train_dict[_token]
                _fold_train_y_list[_batch_idx].append(_train_y_list[_train_idx_list[_dict_idx]])
            # prepare linear x and y
            _fold_train_x_vec = []
            _fold_train_y_vec = []
            for _batch_idx, _fold_train_x in enumerate(_fold_train_x_list):
                _tmp_vec = []
                for _token in _filter_valid_base_feature_info:
                    if _token in _fold_train_x:
                        _tmp_vec.append(_fold_train_x[_token])
                    else:
                        _tmp_vec.append(0)
                _tmp_vec = np.array(_tmp_vec) / np.linalg.norm(_tmp_vec)
                # append _base_norm_vec diff
                _fold_train_x_vec.append(np.sum(np.abs(_base_norm_vec - _tmp_vec)))
                if _tmp_args.task_type == 'classification':
                    _tmp_acc = np.mean([_y[0] == _y[1] for _y in _fold_train_y_list[_batch_idx]])
                elif _tmp_args.task_type == 'generation':
                    _tmp_acc = np.mean([_y[0]['recall'] for _y in _fold_train_y_list[_batch_idx]])
                _fold_train_y_vec.append(_tmp_acc)
            # _fold_train_y_vec diff by base
            _fold_train_y_vec = [_fold_valid_perf_base - _acc for _acc in _fold_train_y_vec]
            # calc pearsonr
            _fold_pearsonr = pearsonr(_fold_train_x_vec, _fold_train_y_vec)
            _global_logger.info(f'fold pearsonr: {_fold_pearsonr}')
            # reshape if needed
            if np.array(_fold_train_x_vec).ndim == 1:
                _fold_train_x_vec = np.array(_fold_train_x_vec).reshape(-1, 1)
            # _fold_train_x_vec = np.array(_fold_train_x_vec).reshape(-1, 1)
            _lr = LinearRegression()
            _lr.fit(_fold_train_x_vec, _fold_train_y_vec)
            _fold_train_y_pred_vec = _lr.predict(_fold_train_x_vec)
            
            # _a, _b = np.polyfit(_fold_train_x_vec, _fold_train_y_vec, deg=1)
            # _global_logger.info(f'fold a: {_a}, b: {_b}')
            # pred train
            # _fold_train_y_pred_vec = np.array(_fold_train_x_vec) * _a + _b
            _fold_train_y_pred = [_fold_valid_perf_base + _pred for _pred in _fold_train_y_pred_vec]
            _fold_train_y_truth = [_fold_valid_perf_base + _acc for _acc in _fold_train_y_vec]
            _fold_train_mae = mean_absolute_error(_fold_train_y_truth, _fold_train_y_pred)
            _global_logger.info(f'fold mae: {_fold_train_mae}')
            if _best_valid_mae is None or _fold_train_mae < _best_valid_mae:
                _best_valid_mae = _fold_train_mae
                _best_train_base_feature_info = _filter_valid_base_feature_info
                _best_norm_vec = _base_norm_vec
                _best_train_base_perf = _fold_valid_perf_base
                _global_logger.info(f'best valid mae: {_best_valid_mae}')
                _best_lr = _lr
            _kfold_idx += 1
        # _fold_vali
        # _tmp_vec = np.array(_tmp_vec) / np.linalg.norm(_tmp_vec)
        # _fold_train_x_list.append(np.sum(np.abs(_base_norm_vec - _tmp_vec)))
        # convert _train_count_dict_list into batch feature,
        # also eval test
        _train_feature_list = []
        _batch_train_acc_diff_list = []
        _test_feature_list = []
        _batch_test_acc_diff_list = []
        for _tmp_list in [
            [_train_feature_list, _train_count_dict_list, _train_y_list, _batch_train_acc_diff_list],
            [_test_feature_list, _test_count_dict_list, _test_y_list, _batch_test_acc_diff_list]
        ]:
            # split by batch
            _target_feature_list = _tmp_list[0]
            for _dict_idx, _dict in enumerate(_tmp_list[1]):
                _batch_idx = _dict_idx // _tmp_args.metaset_batch_size
                if _batch_idx >= len(_target_feature_list):
                    _target_feature_list.append({})
                    _tmp_list[3].append([])
                _target_batch_feature = _target_feature_list[_batch_idx]
                _dict_feature = _target_feature_list[_batch_idx]
                for _token in _best_train_base_feature_info:
                    if _token in _dict:
                        if _token not in _dict_feature:
                            _target_batch_feature[_token] = 0
                        _target_batch_feature[_token] += _dict[_token]
                _tmp_list[3][_batch_idx].append(_tmp_list[2][_dict_idx])
        # avg inside, _batch_train_acc_diff_list, _batch_test_acc_diff_list, then diff
        if _tmp_args.task_type == 'classification':
            _batch_train_acc_diff_list = [_best_train_base_perf - np.mean([_y[0] ==_y[1] for _y in _y_list]) for _y_list in _batch_train_acc_diff_list]
            _batch_test_acc_diff_list = [_best_train_base_perf - np.mean([_y[0] ==_y[1] for _y in _y_list]) for _y_list in _batch_test_acc_diff_list]
        elif _tmp_args.task_type == 'generation':
            _batch_train_acc_diff_list = [_best_train_base_perf - np.mean([_y[0]['recall'] for _y in _y_list]) for _y_list in _batch_train_acc_diff_list]
            _batch_test_acc_diff_list = [_best_train_base_perf - np.mean([_y[0]['recall'] for _y in _y_list]) for _y_list in _batch_test_acc_diff_list]
        # ax+b eval
        _batch_train_feature_list = []
        _batch_train_y_list = []
        _batch_test_feature_list = []
        _batch_test_y_list = []
        for _tmp_list in [
            [_batch_train_feature_list, _train_feature_list, _batch_train_acc_diff_list, 'train', _batch_train_y_list],
            [_batch_test_feature_list, _test_feature_list, _batch_test_acc_diff_list, 'test', _batch_test_y_list]
        ]:
            for _feature in _tmp_list[1]:
                _tmp_vec = []
                for _token in _best_train_base_feature_info:
                    if _token in _feature:
                        _tmp_vec.append(_feature[_token])
                    else:
                        _tmp_vec.append(0)
                _tmp_vec = np.array(_tmp_vec) / np.linalg.norm(_tmp_vec)
                _tmp_list[0].append(np.sum(np.abs(_best_norm_vec - _tmp_vec)))
            # use tmp list 0, 2
            _to_predict_x = _tmp_list[0]
            if np.array(_to_predict_x).ndim == 1:
                _to_predict_x = np.array(_to_predict_x).reshape(-1, 1)
            # _best_lr
            _batch_pred_list = _best_lr.predict(_to_predict_x)
            # recover to acc
            _batch_pred_list = [_best_train_base_perf - _pred for _pred in _batch_pred_list]
            _batch_truth_list = [_best_train_base_perf - _acc for _acc in _tmp_list[2]]
            _batch_mae = mean_absolute_error(_batch_truth_list, _batch_pred_list)
            _global_logger.info(f'{_tmp_list[3]} batch mae: {_batch_mae}')
            _tmp_list[4].append(_batch_truth_list)
            _tmp_list[4].append(_batch_pred_list)
        return {
            'train_batch_feature_list': _batch_train_feature_list,
            'train_batch_truth_list': _batch_train_y_list[0],
            'train_batch_pred_list': _batch_train_y_list[1],
            'test_batch_feature_list': _batch_test_feature_list,
            'test_batch_truth_list': _batch_test_y_list[0],
            'test_batch_pred_list': _batch_test_y_list[1]
        }
    @staticmethod
    def extract_combine_confidence_feature(_tmp_args, _feature_name, _total_len, _res_path_list, _res_args_list, _global_logger):
        # just extract confidence list
        # extract float number from _feature_name
        _confidence_v = None
        if re.search(r'confidence\d+\.\d+', _feature_name):
            _confidence_v = float(re.search(r'confidence(\d+\.\d+)', _feature_name).group(1))
        _test_len = int(_total_len * _tmp_args.estimation_test_ratio)
        _test_batch_count = _test_len // _tmp_args.metaset_batch_size
        if _test_len % _tmp_args.metaset_batch_size > 0:
            _test_batch_count += 1
        _train_len = _total_len - _test_len
        _train_feature_list = []
        _train_truth_list = []
        _test_feature_list = []
        _test_truth_list = []
        _item_idx = 0
        for _res_path_idx, _res_path in enumerate(_res_path_list):
            if not os.path.exists(_res_path):
                _global_logger.info('missing generation result')
                return
            _target_args, _prefix_len = UtilsMetric.get_dynamic_args_from_list(_tmp_args, _res_args_list, _item_idx)
            _data_reader = UtilsMetric.get_original_data_reader(_tmp_args)
            if _prefix_len > 0:
                for _ in range(_prefix_len):
                    next(_data_reader, None)
            _zip_reader = zip(
                _data_reader,
                open(_res_path, 'r')
            )
            for _line_info in _zip_reader:
                if _target_args.task_type == 'classification':
                    _qa_choice_list = UtilsMetric.get_dynamic_qa_choice_list(_target_args, _res_args_list, _item_idx)
                _raw_info = json.loads(_line_info[0])
                _res_info = json.loads(_line_info[1])
                if _item_idx < _train_len:
                    _train_batch_idx = _item_idx // _target_args.metaset_batch_size
                    if _train_batch_idx >= len(_train_feature_list):
                        _train_feature_list.append([])
                        _train_truth_list.append([])
                    _target_feature_list = _train_feature_list[_train_batch_idx]
                    _target_truth_list = _train_truth_list[_train_batch_idx]
                else:
                    _test_batch_idx = (_item_idx - _train_len) // _target_args.metaset_batch_size
                    if _test_batch_idx >= len(_test_feature_list):
                        _test_feature_list.append([])
                        _test_truth_list.append([])
                    _target_feature_list = _test_feature_list[_test_batch_idx]
                    _target_truth_list = _test_truth_list[_test_batch_idx]
                # just use set_confidence_item_feature
                _tmp_feature  = {}
                UtilsMetric.set_confidence_item_feature(_target_args, _res_info, _tmp_feature)
                _target_feature_list.append(_tmp_feature['confidence'][0])
                if _target_args.task_type == 'classification':
                    _target_truth_list.append([
                        _qa_choice_list.index(_res_info['choice_text_softmax'][0]),
                        _qa_choice_list.index(UtilsDataset.get_item_answer(_target_args, _raw_info))
                    ])
                elif _target_args.task_type == 'generation':
                    _answer_txt = _res_info['answer']
                    _response_txt = _res_info['response']
                    _res_metric = {
                        'precision': UtilsMetric.calc_qa_precision(_target_args, _res_info),
                        'recall': UtilsMetric.calc_qa_recall(_target_args, _res_info),
                        'f1': UtilsMetric.calc_qa_f1(_target_args, _res_info)
                    }
                    _target_truth_list.append([
                        _res_metric,
                        _answer_txt,
                        _response_txt,
                    ])
                _item_idx += 1
        return {
            'train_batch_feature_list': _train_feature_list,
            'train_batch_truth_list': _train_truth_list,
            'test_batch_feature_list': _test_feature_list,
            'test_batch_truth_list': _test_truth_list,
            'confidence_v': _confidence_v
        }
    @staticmethod
    def extract_combine_uc_feature(_tmp_args, _feature_name, _total_len, _res_path_list, _res_args_list, _global_logger):
        _test_len = int(_total_len * _tmp_args.estimation_test_ratio)
        _test_batch_count = _test_len // _tmp_args.metaset_batch_size
        if _test_len % _tmp_args.metaset_batch_size > 0:
            _test_batch_count += 1
        _train_len = _total_len - _test_len
        _train_feature_list = []
        _train_truth_list = []
        _test_feature_list = []
        _test_truth_list = []
        _item_idx = 0
        for _res_path_idx, _res_path in enumerate(_res_path_list):
            if not os.path.exists(_res_path):
                _global_logger.info('missing generation result')
                return
            _target_args, _prefix_len = UtilsMetric.get_dynamic_args_from_list(_tmp_args, _res_args_list, _item_idx)
            _data_reader = UtilsMetric.get_original_data_reader(_tmp_args)
            if _prefix_len > 0:
                for _ in range(_prefix_len):
                    # _data_reader.readline()
                    next(_data_reader, None)
            _zip_reader = zip(
                _data_reader,
                open(_res_path, 'r')
            )
            for _line_info in _zip_reader:
                if _target_args.task_type == 'classification':
                    _qa_choice_list = UtilsMetric.get_dynamic_qa_choice_list(_target_args, _res_args_list, _item_idx)
                _raw_info = json.loads(_line_info[0])
                _res_info = json.loads(_line_info[1])
                # only train, test
                if _item_idx < _train_len:
                    _train_batch_idx = _item_idx // _target_args.metaset_batch_size
                    if _train_batch_idx >= len(_train_feature_list):
                        _train_feature_list.append([])
                        _train_truth_list.append([])
                    _target_feature_list = _train_feature_list[_train_batch_idx]
                    _target_truth_list = _train_truth_list[_train_batch_idx]
                else:
                    _test_batch_idx = (_item_idx - _train_len) // _target_args.metaset_batch_size
                    if _test_batch_idx >= len(_test_feature_list):
                        _test_feature_list.append([])
                        _test_truth_list.append([])
                    _target_feature_list = _test_feature_list[_test_batch_idx]
                    _target_truth_list = _test_truth_list[_test_batch_idx]
                _tmp_feature_info = {}
                UtilsMetric.set_uc_item_feature(_target_args, _res_info, _tmp_feature_info)
                _target_feature_list.append(_tmp_feature_info)
                if _target_args.task_type == 'classification':
                    _target_truth_list.append([
                        _qa_choice_list.index(_res_info['choice_text_softmax'][0]),
                        _qa_choice_list.index(UtilsDataset.get_item_answer(_target_args, _raw_info))
                    ])
                elif _target_args.task_type == 'generation':
                    _answer_txt = _res_info['answer']
                    _response_txt = _res_info['response']
                    _res_metric = {
                        'precision': UtilsMetric.calc_qa_precision(_target_args, _res_info),
                        'recall': UtilsMetric.calc_qa_recall(_target_args, _res_info),
                        'f1': UtilsMetric.calc_qa_f1(_target_args, _res_info)
                    }
                    _target_truth_list.append([
                        _res_metric,
                        _answer_txt,
                        _response_txt,
                    ])
                _item_idx += 1
        return {
            'train_batch_feature_list': _train_feature_list,
            'train_batch_truth_list': _train_truth_list,
            'test_batch_feature_list': _test_feature_list,
            'test_batch_truth_list': _test_truth_list
        }
    @staticmethod
    def extract_combine_confidence_profile_feature(_tmp_args, _feature_name, _total_len, _res_path_list, _res_args_list, _global_logger):
        # just extract its confidence list (max prob of each item in batch)
        _test_len = int(_total_len * _tmp_args.estimation_test_ratio)
        _test_batch_count = _test_len // _tmp_args.metaset_batch_size
        if _test_len % _tmp_args.metaset_batch_size > 0:
            _test_batch_count += 1
        _train_len = _total_len - _test_len
        _train_feature_list = []
        _train_truth_list = []
        _test_feature_list = []
        _test_truth_list = []
        _item_idx = 0
        for _res_path_idx, _res_path in enumerate(_res_path_list):
            if not os.path.exists(_res_path):
                _global_logger.info('missing generation result')
                return
            _target_args, _prefix_len = UtilsMetric.get_dynamic_args_from_list(_tmp_args, _res_args_list, _item_idx)
            _data_reader = UtilsMetric.get_original_data_reader(_tmp_args)
            if _prefix_len > 0:
                for _ in range(_prefix_len):
                    # _data_reader.readline()
                    next(_data_reader, None)
            _zip_reader = zip(
                _data_reader,
                open(_res_path, 'r')
            )
            for _line_info in _zip_reader:
                if _target_args.task_type == 'classification':
                    _qa_choice_list = UtilsMetric.get_dynamic_qa_choice_list(_target_args, _res_args_list, _item_idx)
                _raw_info = json.loads(_line_info[0])
                _res_info = json.loads(_line_info[1])
                if _item_idx < _train_len:
                    _train_batch_idx = _item_idx // _target_args.metaset_batch_size
                    if _train_batch_idx >= len(_train_feature_list):
                        _train_feature_list.append([])
                        _train_truth_list.append([])
                    _target_feature_list = _train_feature_list[_train_batch_idx]
                    _target_truth_list = _train_truth_list[_train_batch_idx]
                else:
                    _test_batch_idx = (_item_idx - _train_len) // _target_args.metaset_batch_size
                    if _test_batch_idx >= len(_test_feature_list):
                        _test_feature_list.append([])
                        _test_truth_list.append([])
                    _target_feature_list = _test_feature_list[_test_batch_idx]
                    _target_truth_list = _test_truth_list[_test_batch_idx]
                if _target_args.task_type == 'classification':
                    _logit_raw = _res_info['choice_logits_raw'][0]
                    _logit_softmax = torch.nn.functional.softmax(torch.tensor(_logit_raw), dim=0).tolist()
                    _max_conf = max(_logit_softmax)
                    _target_feature_list.append(_max_conf)
                    _target_truth_list.append([
                        _qa_choice_list.index(_res_info['choice_text_softmax'][0]),
                        _qa_choice_list.index(UtilsDataset.get_item_answer(_target_args, _raw_info))
                    ])
                elif _target_args.task_type == 'generation':
                    _conf_s = _res_info['nll']
                    _target_feature_list.append(_conf_s)
                    _answer_txt = _res_info['answer']
                    _response_txt = _res_info['response']
                    _res_metric = {
                        'precision': UtilsMetric.calc_qa_precision(_target_args, _res_info),
                        'recall': UtilsMetric.calc_qa_recall(_target_args, _res_info),
                        'f1': UtilsMetric.calc_qa_f1(_target_args, _res_info)
                    }
                    _target_truth_list.append([
                        _res_metric,
                        _answer_txt,
                        _response_txt,
                    ])
                _item_idx += 1
        return {
            'train_batch_feature_list': _train_feature_list,
            'train_batch_truth_list': _train_truth_list,
            'test_batch_feature_list': _test_feature_list,
            'test_batch_truth_list': _test_truth_list
        }
    @staticmethod
    def extract_perf_y_list(_tmp_args, _feature_info):
        _tmp_train_y_list = _feature_info['train_batch_truth_list']
        _tmp_test_y_list = _feature_info['test_batch_truth_list']
        if _tmp_args.task_type == 'classification':
            _tmp_train_y_list = [
                np.mean([_y[0] == _y[1] for _y in _y_list]) for _y_list in _tmp_train_y_list
            ]
            _tmp_test_y_list = [
                np.mean([_y[0] == _y[1] for _y in _y_list]) for _y_list in _tmp_test_y_list
            ]
            return _tmp_train_y_list, _tmp_test_y_list
        elif _tmp_args.task_type == 'generation':
            _tmp_train_y_list = [
                np.mean([_y[0]['recall'] for _y in _y_list]) for _y_list in _tmp_train_y_list
            ]
            _tmp_test_y_list = [
                np.mean([_y[0]['recall'] for _y in _y_list]) for _y_list in _tmp_test_y_list
            ]
            return _tmp_train_y_list, _tmp_test_y_list
        pass
    @staticmethod
    def extract_combine_meta_feature(_tmp_args, _total_len, _res_path_list, _res_args_list, _global_logger):
        _feature_name_list = _tmp_args.estimation_feature_list
        _feature_info_list = []
        for _feature_name in _feature_name_list:
            _global_logger.info(f'extracting {_feature_name}')
            _cache_path = UtilsTask.get_combine_feature_cache_path(
                _args = _tmp_args,
                _feature_name = _feature_name,
            )
            _feature_info = None
            if os.path.exists(_cache_path) and _tmp_args.use_cache:
                try:
                    _global_logger.info(f'cache loaded from {_cache_path}')
                    _feature_info = pickle.load(open(_cache_path, 'rb'))
                    if re.search(r'confidence\d+\.\d+', _feature_name):
                        # overwrite confidence_v
                        _feature_info['confidence_v'] = float(re.search(r'confidence(\d+\.\d+)', _feature_name).group(1))
                except:
                    _global_logger.info(f'cache load failed from {_cache_path}')
            if _feature_info is None:
                if _feature_name == 'token_diff':
                    _feature_info = UtilsMetric.extract_combine_token_diff_feature(
                        _tmp_args = _tmp_args,
                        _total_len = _total_len,
                        _res_path_list = _res_path_list,
                        _res_args_list = _res_args_list,
                        _global_logger = _global_logger,
                    )
                elif re.search(r'confidence\d+\.\d+', _feature_name) or _feature_name == 'confidence':
                    _feature_info = UtilsMetric.extract_combine_confidence_feature(
                        _tmp_args = _tmp_args,
                        _feature_name=_feature_name,
                        _total_len = _total_len,
                        _res_path_list = _res_path_list,
                        _res_args_list = _res_args_list,
                        _global_logger = _global_logger,
                    )
                elif _feature_name in ['uc', 'uc_entropy', 'uc_deep', 'uc_margin', 'uc_mde']:
                    _feature_info = UtilsMetric.extract_combine_uc_feature(
                        _tmp_args = _tmp_args,
                        _feature_name=_feature_name,
                        _total_len = _total_len,
                        _res_path_list = _res_path_list,
                        _res_args_list = _res_args_list,
                        _global_logger = _global_logger,
                    )
                elif _feature_name == 'confidence_profile':
                    _feature_info = UtilsMetric.extract_combine_confidence_profile_feature(
                        _tmp_args = _tmp_args,
                        _feature_name=_feature_name,
                        _total_len = _total_len,
                        _res_path_list = _res_path_list,
                        _res_args_list = _res_args_list,
                        _global_logger = _global_logger,
                    )
            # dump to cache
            open(_cache_path, 'wb').write(pickle.dumps(_feature_info))
            _feature_info_list.append([_feature_info, _feature_name])
        # prepare x y list
        _train_x_list = []
        _train_y_list = []
        _test_x_list = []
        _test_y_list = []
        for _feature_idx, _feature_row in enumerate(_feature_info_list):
            _feature_name = _feature_row[1]
            _feature_info = _feature_row[0]
            # concat feature
            if _feature_name == 'token_diff':
                _tmp_train_x_list = _feature_info['train_batch_feature_list']
                _tmp_train_y_list = _feature_info['train_batch_truth_list']
                _tmp_test_x_list = _feature_info['test_batch_feature_list']
                _tmp_test_y_list = _feature_info['test_batch_truth_list']
            elif re.search(r'confidence\d+\.\d+', _feature_name):
                # only detach count > _confidence_v
                _tmp_train_x_list = _feature_info['train_batch_feature_list']
                _tmp_test_x_list = _feature_info['test_batch_feature_list']
                if _tmp_args.task_type == 'classification':
                    # adjust by sum count > confidence_v in each batch
                    _tmp_train_x_list = [
                        np.sum([_x > _feature_info['confidence_v'] for _x in _x_list]) for _x_list in _tmp_train_x_list
                    ]
                    _tmp_test_x_list = [
                        np.sum([_x > _feature_info['confidence_v'] for _x in _x_list]) for _x_list in _tmp_test_x_list
                    ]
                elif _tmp_args.task_type == 'generation':
                    # max conf v from train + test
                    _max_conf_s = max([
                        max(_x_list) for _x_list in _tmp_train_x_list + _tmp_test_x_list
                    ])
                    # sum by pick v / max > confidence_v
                    _tmp_train_x_list = [
                        np.sum([_x / _max_conf_s > _feature_info['confidence_v'] for _x in _x_list]) for _x_list in _tmp_train_x_list
                    ]
                    _tmp_test_x_list = [
                        np.sum([_x / _max_conf_s > _feature_info['confidence_v'] for _x in _x_list]) for _x_list in _tmp_test_x_list
                    ]
                _tmp_train_y_list, _tmp_test_y_list = UtilsMetric.extract_perf_y_list(_tmp_args, _feature_info)
            elif _feature_name == 'confidence':
                _tmp_train_x_list = _feature_info['train_batch_feature_list']
                _tmp_test_x_list = _feature_info['test_batch_feature_list']
                _tmp_train_x_v_list = []
                _tmp_test_x_v_list = []
                _tmp_train_y_list, _tmp_test_y_list = UtilsMetric.extract_perf_y_list(_tmp_args, _feature_info)
                if _tmp_args.task_type == 'generation':
                    _max_conf_s = max([
                        max(_x_list) for _x_list in _tmp_train_x_list + _tmp_test_x_list
                    ])
                # prepare 0.1-0.9
                for _v in range(1, 10):
                    if _tmp_args.task_type == 'classification':
                        _tmp_iter_train_x_list = [
                            np.sum([_x > _v / 10 for _x in _x_list]) for _x_list in _tmp_train_x_list
                        ]
                        _tmp_iter_test_x_list = [
                            np.sum([_x > _v / 10 for _x in _x_list]) for _x_list in _tmp_test_x_list
                        ]
                    elif _tmp_args.task_type == 'generation':
                        #  val divide max
                        _tmp_iter_train_x_list = [
                            np.sum([_x / _max_conf_s > _v / 10 for _x in _x_list]) for _x_list in _tmp_train_x_list
                        ]
                        _tmp_iter_test_x_list = [
                            np.sum([_x / _max_conf_s > _v / 10 for _x in _x_list]) for _x_list in _tmp_test_x_list
                        ]
                    if len(_tmp_train_x_v_list) == 0:
                        _tmp_train_x_v_list = _tmp_iter_train_x_list
                        _tmp_test_x_v_list = _tmp_iter_test_x_list
                    else:
                        # v stack column into _tmp_train_x_v_list
                        _tmp_train_x_v_list = np.column_stack((_tmp_train_x_v_list, _tmp_iter_train_x_list))
                        _tmp_test_x_v_list = np.column_stack((_tmp_test_x_v_list, _tmp_iter_test_x_list))
                _tmp_train_x_list = _tmp_train_x_v_list
                _tmp_test_x_list = _tmp_test_x_v_list
            elif _feature_name in UtilsMetric._uc_metric_list:
                # extract target key from each feature info, then mean in batch
                _train_feature_info_list = _feature_info['train_batch_feature_list']
                _test_feature_info_list = _feature_info['test_batch_feature_list']
                _tmp_train_x_list = [
                    np.mean([_feature[_feature_name] for _feature in _feature_list]) for _feature_list in _train_feature_info_list
                ]
                _tmp_test_x_list = [
                    np.mean([_feature[_feature_name] for _feature in _feature_list]) for _feature_list in _test_feature_info_list
                ]
                _tmp_train_y_list, _tmp_test_y_list = UtilsMetric.extract_perf_y_list(_tmp_args, _feature_info)
            elif _feature_name == 'uc':
                # extract all from UtilsMetric._uc_metric_list in same row
                _train_feature_info_list = _feature_info['train_batch_feature_list']
                _test_feature_info_list = _feature_info['test_batch_feature_list']
                _tmp_train_x_list = [
                    [
                        np.mean([_feature[_metric_key] for _feature in _feature_list])
                        for _metric_key in UtilsMetric._uc_metric_list
                    ]
                    for _feature_list in _train_feature_info_list
                ]
                _tmp_test_x_list = [
                    [
                        np.mean([_feature[_metric_key] for _feature in _feature_list])
                        for _metric_key in UtilsMetric._uc_metric_list
                    ]
                    for _feature_list in _test_feature_info_list
                ]
                _tmp_train_y_list, _tmp_test_y_list = UtilsMetric.extract_perf_y_list(_tmp_args, _feature_info)
            elif _feature_name == 'confidence_profile':
                _train_feature_info_list = _feature_info['train_batch_feature_list']
                _test_feature_info_list = _feature_info['test_batch_feature_list']
                _tmp_train_y_list, _tmp_test_y_list = UtilsMetric.extract_perf_y_list(_tmp_args, _feature_info)
                _tmp_train_x_list = []
                _tmp_test_x_list = []
                # sample with confidence_profile_dim
                for _feature_info in [
                    [_train_feature_info_list, _tmp_train_x_list],
                    [_test_feature_info_list, _tmp_test_x_list]
                ]:
                    for _batch_feature in _feature_info[0]:
                        _ranked_conf_list = sorted(_batch_feature, reverse=True)
                        _sample_idx_list = np.linspace(0, 100, num=_tmp_args.confidence_profile_dim).tolist()
                        _sample_conf_v = np.percentile(_ranked_conf_list, _sample_idx_list)
                        _feature_info[1].append(_sample_conf_v)
            if len(_train_x_list) == 0:
                # just replace
                _train_x_list = _tmp_train_x_list
                _train_y_list = _tmp_train_y_list
                _test_y_list = _tmp_test_y_list
                _test_x_list = _tmp_test_x_list
            else:
                _train_len = len(_train_x_list)
                # concat each column, column stack
                _train_x_list = np.column_stack((_train_x_list, _tmp_train_x_list))
                _test_x_list = np.column_stack((_test_x_list, _tmp_test_x_list))
        return {
            'train_x_list': _train_x_list,
            'train_y_list': _train_y_list,
            'test_x_list': _test_x_list,
            'test_y_list': _test_y_list
        }
class UtilsEstimationAvgTrain():
    @staticmethod
    def calc_estimation_res(_tmp_args, _total_len, _res_path_list, _res_args_list, _global_logger, _use_cache=False):
        _es_path_info = UtilsTask.get_es_info(_tmp_args)
        _cache_file = _es_path_info['cache_pkl']
        _res_es_file = _es_path_info['res_info']
        if _use_cache and os.path.exists(_cache_file):
            _cache_info = pickle.load(open(_cache_file, 'rb'))
            _global_logger.info(f'cache loaded from {_cache_file}')
        else:
            _meta_feature_info = UtilsMetric.extract_meta_feature(
                _tmp_args = _tmp_args,
                _total_len = _total_len,
                _res_path_list = _res_path_list,
                _res_args_list = _res_args_list,
                _global_logger = _global_logger,
            )
            # acc from _train_base_feature_info
            _train_base_acc = np.mean(_meta_feature_info['train_base_feature_info']['status_list'])
            _cache_info = {
                'train_base_acc': _train_base_acc,
                'test_status_list': [
                    _feature['status_list'] for _feature in _meta_feature_info['test_feature_info_list']
                ]
            }
            open(_cache_file, 'wb').write(pickle.dumps(_cache_info))
            _global_logger.info(f'cache saved to {_cache_file}')
        _train_base_acc = _cache_info['train_base_acc']
        # from cache info
        # direct use base acc as estimated acc, len from test_feature_info_list
        _test_pred = [_train_base_acc] * len(_cache_info['test_status_list'])
        _test_truth = [
            np.mean(_status_list) for _status_list in _cache_info['test_status_list']
        ]
        _test_mse = mean_squared_error(_test_truth, _test_pred)
        _test_mae = mean_absolute_error(_test_truth, _test_pred)
        _res_info = {
            'train_base_acc': _train_base_acc,
            'test_pred': _test_pred,
            'test_truth': _test_truth,
            'test_mse': _test_mse,
            'test_mae': _test_mae,
            'args': _tmp_args.__dict__
        }
        # log test mse
        _global_logger.info(f'test mse: {_test_mse}')
        open(_res_es_file, 'w').write(json.dumps(_res_info))
        _global_logger.info(f'result saved to {_res_es_file}')
        return {
            'res_info': _res_info,
            'cache_info': _cache_info
        }
class UtilsEstimationAvgConf():
    @staticmethod
    def calc_estimation_res(_tmp_args, _total_len, _res_path_list, _res_args_list, _global_logger, _use_cache=False):
        _es_path_info = UtilsTask.get_es_info(_tmp_args)
        _cache_file = _es_path_info['cache_pkl']
        _res_es_file = _es_path_info['res_info']
        if _use_cache and os.path.exists(_cache_file):
            _cache_info = pickle.load(open(_cache_file, 'rb'))
            _global_logger.info(f'cache loaded from {_cache_file}')
        else:
            _meta_feature_info = UtilsMetric.extract_meta_feature(
                _tmp_args = _tmp_args,
                _total_len = _total_len,
                _res_path_list = _res_path_list,
                _res_args_list = _res_args_list,
                _global_logger = _global_logger,
            )
            if _tmp_args.task_type == 'classification':
                _cache_info = {
                    'test_feature_info_list': _meta_feature_info['test_feature_info_list'],
                }
            elif _tmp_args.task_type == 'generation':
                # keep train_base_feature_info test_feature_info_list
                _cache_info = {
                    'train_base_feature_info': _meta_feature_info['train_base_feature_info'],
                    'test_feature_info_list': _meta_feature_info['test_feature_info_list'],
                }
            open(_cache_file, 'wb').write(pickle.dumps(_cache_info))
            _global_logger.info(f'cache saved to {_cache_file}')
        _feature_info_list = _cache_info['test_feature_info_list']
        _test_acc_pred = [
            np.mean(_feature['confidence_list']) for _feature in _feature_info_list
        ]
        if _tmp_args.task_type == 'generation':
            # _test_acc_pred v divide train_base_feature_info confidence_list mean
            _base_conf_s = np.mean(_cache_info['train_base_feature_info']['confidence_list'])
            _test_acc_ratio = [
                _v / _base_conf_s for _v in _test_acc_pred
            ]
            # map acc by 1 - 1 / (1 + math.exp(-x)) of ratio
            _test_acc_pred = [
                UtilsMetric.map_gen_conf_v(_v) for _v in _test_acc_ratio
            ]
        _test_acc_truth = [
            np.mean(_feature['status_list']) for _feature in _feature_info_list
        ]
        _test_mse = mean_squared_error(_test_acc_truth, _test_acc_pred)
        _test_mae = mean_absolute_error(_test_acc_truth, _test_acc_pred)
        _res_info = {
            'test_pred': _test_acc_pred,
            'test_truth': _test_acc_truth,
            'test_mse': _test_mse,
            'test_mae': _test_mae,
            'args': _tmp_args.__dict__
        }
        open(_res_es_file, 'w').write(json.dumps(_res_info))
        _global_logger.info(f'result saved to {_res_es_file}')
        # log
        _global_logger.info(f'test mse: {_test_mse}, test mae: {_test_mae}')
        return {
            'res_info': _res_info,
            'cache_info': _cache_info
        }
        pass
class UtilsEstimationConfidenceProfile():
    @staticmethod
    def estimate_KNN(_args, _train_info, _test_info, _logger):
        _train_x = np.array(_train_info[0])
        _train_y = np.array(_train_info[1])
        _test_x = np.array(_test_info[0])
        _test_y = np.array(_test_info[1])
        if _args.k_fold:
            # search with k fold about neighbor 1 3 5 7 9
            _best_neighbor = None
            _best_mae = None
            # KFold split train
            _kfold = KFold(n_splits=_args.k_fold, shuffle=True, random_state=_args.seed)
            _train_list = list(zip(_train_x, _train_y))
            for _candi_k in range(1, 10, 2):
                # init model
                _model = KNeighborsRegressor(n_neighbors=_candi_k, weights='distance')
                for _fold_train_idx, _fold_valid_idx in _kfold.split(_train_list):
                    _fold_train_x = _train_x[_fold_train_idx]
                    _fold_train_y = _train_y[_fold_train_idx]
                    _fold_valid_x = _train_x[_fold_valid_idx]
                    _fold_valid_y = _train_y[_fold_valid_idx]
                    _model.fit(_fold_train_x, _fold_train_y)
                    _fold_valid_pred = _model.predict(_fold_valid_x)
                    _fold_valid_mae = mean_absolute_error(_fold_valid_y, _fold_valid_pred)
                    if _best_mae is None or _fold_valid_mae < _best_mae:
                        _best_mae = _fold_valid_mae
                        _best_neighbor = _candi_k
            # log best neighbor
            _logger.info(f'best KNN neighbor: {_best_neighbor}')
            # init model with best neighbor
            _model = KNeighborsRegressor(n_neighbors=_best_neighbor, weights='distance')
            _model.fit(_train_x, _train_y)
            # try pred train
            _train_pred = _model.predict(_train_x)
            _train_mse = mean_squared_error(_train_y, _train_pred)
            _train_mae = mean_absolute_error(_train_y, _train_pred)
            _logger.info(f'KNN train mse: {_train_mse}, train mae: {_train_mae}')
            # predict
            _test_pred = _model.predict(_test_x)
            _test_mse = mean_squared_error(_test_y, _test_pred)
            _test_mae = mean_absolute_error(_test_y, _test_pred)
            _logger.info(f'KNN test mse: {_test_mse}, test mae: {_test_mae}')
            return {
                'train_pred': _train_pred.tolist(),
                'train_truth': _train_y.tolist(),
                'train_mse': _train_mse,
                'train_mae': _train_mae,
                'test_pred': _test_pred.tolist(),
                'test_truth': _test_y.tolist(),
                'test_mse': _test_mse,
                'test_mae': _test_mae
            }
        else:
            _model = KNeighborsRegressor(n_neighbors=_args.knn_neighbor, weights='distance')
            # fit
            _model.fit(_train_x, _train_y)
            # try pred train
            _train_pred = _model.predict(_train_x)
            _train_mse = mean_squared_error(_train_y, _train_pred)
            _train_mae = mean_absolute_error(_train_y, _train_pred)
            _logger.info(f'KNN train mse: {_train_mse}, train mae: {_train_mae}')
            # predict
            _test_pred = _model.predict(_test_x)
            _test_mse = mean_squared_error(_test_y, _test_pred)
            _test_mae = mean_absolute_error(_test_y, _test_pred)
            _logger.info(f'KNN test mse: {_test_mse}, test mae: {_test_mae}')
            return {
                'train_pred': _train_pred.tolist(),
                'train_truth': _train_y.tolist(),
                'train_mse': _train_mse,
                'train_mae': _train_mae,
                'test_pred': _test_pred.tolist(),
                'test_truth': _test_y.tolist(),
                'test_mse': _test_mse,
                'test_mae': _test_mae
            }
    @staticmethod
    def estimate_MLP(_args, _train_info, _valid_info, _test_info, _logger):
        # initiate ConfidenceProfileMLP
        _model = ConfidenceProfileMLP(
            _args=_args,
            _input_dim=len(_train_info[0][0])
        )
        _res_info = _model.train_estimate(_train_info, _valid_info, _test_info, _logger)
        return _res_info
    @staticmethod
    def estimate_XGBoost(_args, _train_info, _test_info, _logger):
        _params = {
                "learning_rate": uniform(0.01, 0.5),
                "max_depth": randint(3,10),
                "n_estimators": randint(100, 1000),
                'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1],
                'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1],
                "gamma": uniform(0,1),
                "reg_alpha": uniform(0,1),
                "reg_lambda": uniform(0,1),
                }
        if _args.k_fold:
            # split train into train and valid, search and evaluate best mae on valid
            _train_x = np.array(_train_info[0])
            _train_y = np.array(_train_info[1])
            _test_x = np.array(_test_info[0])
            _test_y = np.array(_test_info[1])
            _kfold = KFold(n_splits=_args.k_fold, shuffle=True, random_state=_args.seed)
            _best_params = None
            _best_valid_mae = None
            _best_test_res_info = None
            for _fold_idx, (_train_idx, _valid_idx) in enumerate(_kfold.split(_train_x)):
                _fold_train_x = _train_x[_train_idx]
                _fold_train_y = _train_y[_train_idx]
                _fold_valid_x = _train_x[_valid_idx]
                _fold_valid_y = _train_y[_valid_idx]
                _random_search = RandomizedSearchCV(
                    estimator=xgboost.XGBRegressor(objective='reg:squarederror'),
                    param_distributions=_params,
                    n_iter=300,
                    cv=5,
                    n_jobs=-1,
                    verbose=0,
                )
                _random_search.fit(_fold_train_x, _fold_train_y)
                _curr_best_params = _random_search.best_params_
                _model = xgboost.XGBRegressor(objective='reg:squarederror', **_curr_best_params)
                _model.fit(_fold_train_x, _fold_train_y)
                _fold_valid_pred = _model.predict(_fold_valid_x)
                _fold_valid_mae = mean_absolute_error(_fold_valid_y, _fold_valid_pred)
                _logger.info(f'fold {_fold_idx} valid mae: {_fold_valid_mae}')
                if _best_valid_mae is None or _fold_valid_mae < _best_valid_mae:
                    _best_valid_mae = _fold_valid_mae
                    _best_params = _curr_best_params
                    # calc test res here
                    _test_pred = _model.predict(_test_x)
                    _test_mae = mean_absolute_error(_test_y, _test_pred)
                    _test_mse = mean_squared_error(_test_y, _test_pred)
                    # train also
                    _train_pred = _model.predict(_fold_train_x)
                    _train_mae = mean_absolute_error(_fold_train_y, _train_pred)
                    _train_mse = mean_squared_error(_fold_train_y, _train_pred)
                    _best_test_res_info = {
                        'train_pred': _train_pred.tolist(),
                        'train_truth': _fold_train_y.tolist(),
                        'train_mae': _train_mae,
                        'train_mse': _train_mse,
                        'test_pred': _test_pred.tolist(),
                        'test_truth': _test_y.tolist(),
                        'test_mae': _test_mae,
                        'test_mse': _test_mse
                    }
            return _best_test_res_info
        else:
            _model = xgboost.XGBRegressor(objective='reg:squarederror')
            _random_search = RandomizedSearchCV(
                estimator=_model,
                param_distributions=_params,
                n_iter=300,
                cv=5,
                n_jobs=-1,
                verbose=0,
            )
            _train_x, _train_y = _train_info
            _test_x, _test_y = _test_info
            _random_search.fit(_train_x, _train_y)
            _best_params = _random_search.best_params_
            _logger.info(f'best params: {_best_params}')
            _model = xgboost.XGBRegressor(objective='reg:squarederror', **_best_params)
            _model.fit(_train_x, _train_y)
            _train_pred = _model.predict(_train_x)
            _train_mse = mean_squared_error(_train_y, _train_pred)
            _train_mae = mean_absolute_error(_train_y, _train_pred)
            _logger.info(f'XGBoost train mse: {_train_mse}, train mae: {_train_mae}')
            _test_pred = _model.predict(_test_x)
            _test_mse = mean_squared_error(_test_y, _test_pred)
            _test_mae = mean_absolute_error(_test_y, _test_pred)
            _logger.info(f'XGBoost test mse: {_test_mse}, test mae: {_test_mae}')
            return {
                'train_pred': _train_pred.tolist(),
                'train_truth': _train_y.tolist(),
                'train_mse': _train_mse,
                'train_mae': _train_mae,
                'test_pred': _test_pred.tolist(),
                'test_truth': _test_y.tolist(),
                'test_mse': _test_mse,
                'test_mae': _test_mae
            }
    @staticmethod
    def estimate_LR(_args, _train_info, _test_info, _logger):
        _train_x_list = np.array(_train_info[0])
        _train_y_list = np.array(_train_info[1])
        _test_x_list = np.array(_test_info[0])
        _test_y_list = np.array(_test_info[1])
        # initialize linear regression
        _lr = LinearRegression()
        # reshape if needed
        if len(_train_x_list.shape) == 1:
            _train_x_list = _train_x_list.reshape(-1, 1)
            _test_x_list = _test_x_list.reshape(-1, 1)
        _lr.fit(_train_x_list, _train_y_list)
        # calc train mae, mse
        _train_pred = _lr.predict(_train_x_list).tolist()
        _train_mae = mean_absolute_error(_train_y_list, _train_pred)
        _train_mse = mean_squared_error(_train_y_list, _train_pred)
        _logger.info(f'train mae: {_train_mae}, train mse: {_train_mse}')
        # calc test mae, mse
        _test_pred = _lr.predict(_test_x_list).tolist()
        _test_mae = mean_absolute_error(_test_y_list, _test_pred)
        _test_mse = mean_squared_error(_test_y_list, _test_pred)
        _logger.info(f'test mae: {_test_mae}, test mse: {_test_mse}')
        _res_info = {
            'train_pred': _train_pred,
            'train_truth': _train_y_list.tolist(),
            'train_mae': _train_mae,
            'train_mse': _train_mse,
            'test_pred': _test_pred,
            'test_truth': _test_y_list.tolist(),
            'test_mae': _test_mae,
            'test_mse': _test_mse
        }
        return _res_info
    @staticmethod
    def calc_estimation_res(_tmp_args, _total_len, _res_path_list, _res_args_list, _global_logger, _use_cache=False):
        _es_path_info = UtilsTask.get_es_info(_tmp_args)
        _cache_file = _es_path_info['cache_pkl']
        _res_es_file = _es_path_info['res_info']
        if _use_cache and os.path.exists(_cache_file):
            _cache_info = pickle.load(open(_cache_file, 'rb'))
            _global_logger.info(f'cache loaded from {_cache_file}')
        else:
            _meta_feature_info = UtilsMetric.extract_meta_feature(
                _tmp_args = _tmp_args,
                _total_len = _total_len,
                _res_path_list = _res_path_list,
                _res_args_list = _res_args_list,
                _global_logger = _global_logger,
            )
            _cache_info = {
                'train_feature_info_list': _meta_feature_info['train_feature_info_list'],
                'test_feature_info_list': _meta_feature_info['test_feature_info_list']
            }
            open(_cache_file, 'wb').write(pickle.dumps(_cache_info))
            _global_logger.info(f'cache saved to {_cache_file}')
        # check _res_es_file, return if exists and loaded
        if _use_cache and os.path.exists(_res_es_file):
            _try_res_info = None
            try:
                _try_res_info = json.loads(open(_res_es_file, 'r').read())
            except Exception as e:
                _global_logger.info(f'error: {e}')
            if _try_res_info is not None:
                _global_logger.info(f'result loaded from {_res_es_file}')
                return {
                    'res_info': _try_res_info,
                    'cache_info': _cache_info
                }
        _train_feature_info_list = _cache_info['train_feature_info_list']
        _test_feature_info_list = _cache_info['test_feature_info_list']
        # prepare confidence_profile_dim val from each batch in train and test
        _train_vec_list = []
        _train_y_list = []
        _test_vec_list = []
        _test_y_list = []
        for _feature_info in [
            [_train_feature_info_list, _train_vec_list, _train_y_list],
            [_test_feature_info_list, _test_vec_list, _test_y_list]
        ]:
            for _batch_feature in _feature_info[0]:
                _ranked_conf_list = sorted(_batch_feature['confidence_list'], reverse=True)
                # ocs = np.linspace(0, 100, num=int(dim))
                # np.percentile(self.confs[idx], locs)
                _sample_idx_list = np.linspace(0, 100, num=_tmp_args.confidence_profile_dim).tolist()
                _sample_vec = np.percentile(_ranked_conf_list, _sample_idx_list)
                _feature_info[1].append(_sample_vec)
                _feature_info[2].append(np.mean(_batch_feature['status_list']))
            pass
        if _tmp_args.estimation_model == 'lr':
            _res_info = UtilsEstimationConfidenceProfile.estimate_LR(
                _tmp_args, [_train_vec_list, _train_y_list], [_test_vec_list, _test_y_list], _global_logger
            )
            pass
        elif _tmp_args.estimation_model == 'knn':
            _res_info = UtilsEstimationConfidenceProfile.estimate_KNN(
                _tmp_args, [_train_vec_list, _train_y_list], [_test_vec_list, _test_y_list], _global_logger
            )
        elif _tmp_args.estimation_model == 'mlp':
            # 1/4 train as valid
            _mlp_valid_len = len(_train_vec_list) // 5
            _mlp_train_len = len(_train_vec_list) - _mlp_valid_len
            _mlp_train_info = [_train_vec_list[:_mlp_train_len], _train_y_list[:_mlp_train_len]]
            _mlp_valid_info = [_train_vec_list[_mlp_train_len:], _train_y_list[_mlp_train_len:]]
            _res_info = UtilsEstimationConfidenceProfile.estimate_MLP(
                _tmp_args, _mlp_train_info, _mlp_valid_info, [_test_vec_list, _test_y_list], _global_logger
            )
        elif _tmp_args.estimation_model == 'xgboost':
            _res_info = UtilsEstimationConfidenceProfile.estimate_XGBoost(
                _tmp_args, [_train_vec_list, _train_y_list], [_test_vec_list, _test_y_list], _global_logger
            )
        open(_res_es_file, 'w+').write(json.dumps(_res_info))
        _global_logger.info(f'result saved to {_res_es_file}')
        return {
            'res_info': _res_info,
            'cache_info': _cache_info
        }
    pass

class UtilsEstimationSingle():
    @staticmethod
    def calc_estimation_res(_tmp_args, _total_len, _res_path_list, _res_args_list, _global_logger, _use_cache=False):
        if _tmp_args.estimation_feature not in ['token_diff', 'uc', 'uc_logits', 'confidence']:
            raise ValueError(f'Invalid estimation feature: {_tmp_args.estimation_feature}')
        _es_path_info = UtilsTask.get_es_info(_tmp_args, True)
        _cache_file = _es_path_info['cache_pkl']
        _res_es_file = _es_path_info['res_info']
        if _use_cache and os.path.exists(_cache_file):
            _cache_info = pickle.load(open(_cache_file, 'rb'))
            _global_logger.info(f'cache loaded from {_cache_file}')
        else:
            _meta_feature_info = UtilsMetric.extract_meta_feature(
                _tmp_args = _tmp_args,
                _total_len = _total_len,
                _res_path_list = _res_path_list,
                _res_args_list = _res_args_list,
                _global_logger = _global_logger,
            )
            # try to generate cache from scratch
            _train_base_feature_info = _meta_feature_info['train_base_feature_info']
            _train_base_y_list = _meta_feature_info['train_base_y_list']
            _train_feature_info_list = _meta_feature_info['train_feature_info_list']
            _train_y_list = _meta_feature_info['train_y_list']
            _test_feature_info_list = _meta_feature_info['test_feature_info_list']
            _test_y_list = _meta_feature_info['test_y_list']
            if _tmp_args.estimation_feature == 'token_diff':
                _filtered_train_base_feature_info = {}
                for _token, _freq in _train_base_feature_info.items():
                    if _freq >= _tmp_args.token_diff_min_freq:
                        _filtered_train_base_feature_info[_token] = _freq
                _train_base_norm_vector = list(_filtered_train_base_feature_info.values())
                _train_base_norm_vector = np.array(_train_base_norm_vector) / np.linalg.norm(_train_base_norm_vector)
            elif _tmp_args.estimation_feature in ['uc', 'uc_logits']:
                _train_base_feature_info = [
                    np.mean(_train_base_feature_info['uc_entropy']),
                    np.mean(_train_base_feature_info['uc_deep']),
                    np.mean(_train_base_feature_info['uc_margin']),
                    np.mean(_train_base_feature_info['uc_mde'])
                ]
                _train_base_norm_vector = np.array(_train_base_feature_info)
            elif _tmp_args.estimation_feature == 'confidence':
                _train_base_feature_info = [
                    np.mean([1 if _v > 0.9 else 0 for _v in _train_base_feature_info['confidence']]),
                    np.mean([1 if _v > 0.8 else 0 for _v in _train_base_feature_info['confidence']]),
                    np.mean([1 if _v > 0.7 else 0 for _v in _train_base_feature_info['confidence']])
                ]
                _train_base_norm_vector = np.array(_train_base_feature_info)
                # _global_logger.info(f'_train_base_feature_info: {_train_base_feature_info}')
            if _tmp_args.task_type == 'classification':
                _train_base_perf = f1_score(np.array(_train_base_y_list)[:, 0], np.array(_train_base_y_list)[:, 1], average='micro')
            elif _tmp_args.task_type == 'generation':
                # mean of dict recall
                _train_base_perf = np.mean([_v[0]['recall'] for _v in _train_base_y_list])
            _train_x_list = []
            _train_f1_delta_list = []
            _test_x_list = []
            _test_f1_delta_list = []
            for _token_info in [
                [_train_x_list, _train_feature_info_list, _train_y_list, _train_f1_delta_list],
                [_test_x_list, _test_feature_info_list, _test_y_list, _test_f1_delta_list]
            ]:
                _target_x_list = _token_info[0]
                _target_token_info_list = _token_info[1]
                _tmp_y_list = _token_info[2]
                _tmp_f1_list = _token_info[3]
                for _batch_idx, _token_info in enumerate(_target_token_info_list):
                    _tmp_x = []
                    if _tmp_args.estimation_feature == 'token_diff':
                        for _token, _freq in _filtered_train_base_feature_info.items():
                            if _token in _token_info:
                                _tmp_x.append(_token_info[_token])
                            else:
                                _tmp_x.append(0)
                        _tmp_norm_x = np.array(_tmp_x) / np.linalg.norm(_tmp_x)
                        _target_x_list.append(np.sum(np.abs(_train_base_norm_vector - _tmp_norm_x)))
                    elif _tmp_args.estimation_feature in ['uc', 'uc_logits']:
                        _tmp_x = [
                            np.mean(_token_info['uc_entropy']),
                            np.mean(_token_info['uc_deep']),
                            np.mean(_token_info['uc_margin']),
                            np.mean(_token_info['uc_mde'])
                        ]
                        _tmp_norm_x = np.array(_tmp_x)
                        _target_x_list.append(_train_base_norm_vector - _tmp_norm_x)
                    elif _tmp_args.estimation_feature == 'confidence':
                        _tmp_x = [
                            np.mean([1 if _v > 0.9 else 0 for _v in _token_info['confidence']]),
                            np.mean([1 if _v > 0.8 else 0 for _v in _token_info['confidence']]),
                            np.mean([1 if _v > 0.7 else 0 for _v in _token_info['confidence']])
                        ]
                        _target_x_list.append(np.array(_tmp_x) - _train_base_norm_vector)
                    if _tmp_args.task_type == 'classification':
                        _tmp_perf = f1_score(np.array(_tmp_y_list[_batch_idx])[:, 0], np.array(_tmp_y_list[_batch_idx])[:, 1], average='micro')
                    elif _tmp_args.task_type == 'generation':
                        _tmp_perf = np.mean([_v[0]['recall'] for _v in _tmp_y_list[_batch_idx]])
                    _tmp_f1_list.append(_train_base_perf - _tmp_perf)
            _cache_info = {
                'train': {
                    'x': _train_x_list,
                    'y': _train_f1_delta_list
                },
                'test': {
                    'x': _test_x_list,
                    'y': _test_f1_delta_list
                },
                'train_base_f1': _train_base_perf
            }
            open(_cache_file, 'wb').write(pickle.dumps(_cache_info))
            _global_logger.info(f'cache saved to {_cache_file}')
        # skip if _res_es_file exist
        if _use_cache and os.path.exists(_res_es_file):
            _try_res_info = None
            try:
                _try_res_info = json.loads(open(_res_es_file, 'r').read())
            except Exception as e:
                _global_logger.info(f'error: {e}')
            if _try_res_info is not None:
                _global_logger.info(f'result loaded from {_res_es_file}')
                return {
                    'res_info': _try_res_info,
                    'cache_info': _cache_info
                }
        _train_x_list = _cache_info['train']['x']
        _train_f1_delta_list = _cache_info['train']['y']
        _test_x_list = _cache_info['test']['x']
        _test_f1_delta_list = _cache_info['test']['y']
        _train_base_perf = _cache_info['train_base_f1']
        if _tmp_args.estimation_model == 'lr':
            if _tmp_args.estimation_feature == 'token_diff':
                _corr = pearsonr(_train_x_list, _train_f1_delta_list)
                _global_logger.info(f'Correlation: {_corr}')
                _lr = LinearRegression()
                # if dim 1 reshape -1 1
                if np.array(_train_x_list).ndim == 1:
                    _train_x_list = np.reshape(_train_x_list, (-1, 1))
                    _test_x_list = np.reshape(_test_x_list, (-1, 1))
                _lr.fit(_train_x_list, _train_f1_delta_list)
                _train_delta_pred = _lr.predict(_train_x_list)
                _test_delta_pred = _lr.predict(_test_x_list)
                # _a, _b = np.polyfit(_train_x_list, _train_f1_delta_list, deg=1)
                # _global_logger.info(f'Linear regression: y = {_a}x + {_b}')
                # _train_delta_pred = np.array(_train_x_list) * _a + _b
                _train_pred = _train_base_perf - _train_delta_pred
                _train_truth = _train_base_perf - np.array(_train_f1_delta_list)
                _train_mse = np.mean((_train_pred - _train_truth) ** 2)
                _train_mae = np.mean(np.abs(_train_pred - _train_truth))
                _global_logger.info(f'Train MAE {_tmp_args.estimation_feature}: {_train_mae}')
                # _test_delta_pred = np.array(_test_x_list) * _a + _b
                _test_pred = _train_base_perf - _test_delta_pred
                _test_truth = _train_base_perf - np.array(_test_f1_delta_list)
                _test_mse = np.mean((_test_pred - _test_truth) ** 2)
                _test_mae = np.mean(np.abs(_test_pred - _test_truth))
                _global_logger.info(f'Test MAE {_tmp_args.estimation_feature}: {_test_mae}')
                _res_info = {
                    'corr_val': _corr[0],
                    'corr_p': _corr[1],
                    'lr_params': _lr.coef_.tolist(),
                    # 'polyfit': [_a, _b],
                    'train_base_f1': _train_base_perf,
                    'train_mse': _train_mse,
                    'train_mae': _train_mae,
                    'test_mse': _test_mse,
                    'test_mae': _test_mae,
                    'train_truth': _train_truth.tolist(),
                    'train_pred': _train_pred.tolist(),
                    'test_truth': _test_truth.tolist(),
                    'test_pred': _test_pred.tolist(),
                    'args': _tmp_args.__dict__
                }
            elif _tmp_args.estimation_feature in ['uc', 'confidence', 'uc_logits']:
                _train_truth = _train_base_perf - np.array(_train_f1_delta_list)
                _test_truth = _train_base_perf - np.array(_test_f1_delta_list)
                _res_info = {
                    'train_base_f1': _train_base_perf,
                    'train_truth': _train_truth.tolist(),
                    'test_truth': _test_truth.tolist(),
                    'args': _tmp_args.__dict__
                }
                if _tmp_args.estimation_feature in ['uc', 'uc_logits']:
                    _prefix_list = UtilsMetric._uc_metric_list
                elif _tmp_args.estimation_feature == 'confidence':
                    _prefix_list = ['confidence_0.9', 'confidence_0.8', 'confidence_0.7']
                for _idx, _prefix_label in enumerate(_prefix_list):
                    _prefix_str = f'{_prefix_label}_'
                    if _prefix_label == '':
                        _prefix_str = ''
                    _tmp_train_x_list = [_x[_idx] for _x in _train_x_list]
                    _tmp_test_x_list = [_x[_idx] for _x in _test_x_list]
                    _corr = pearsonr(_tmp_train_x_list, _train_f1_delta_list)
                    _global_logger.info(f'Correlation {_prefix_label}: {_corr}')
                    if np.array(_tmp_train_x_list).ndim == 1:
                        _tmp_train_x_list = np.reshape(_tmp_train_x_list, (-1, 1))
                        _tmp_test_x_list = np.reshape(_tmp_test_x_list, (-1, 1))
                    # use linear regression model
                    _lr = LinearRegression()
                    _lr.fit(_tmp_train_x_list, _train_f1_delta_list)
                    _train_delta_pred = _lr.predict(_tmp_train_x_list)
                    _test_delta_pred = _lr.predict(_tmp_test_x_list)
                    # _a, _b = np.polyfit(_tmp_train_x_list, _train_f1_delta_list, deg=1)
                    # _global_logger.info(f'Linear regression {_prefix_label}: y = {_a}x + {_b}')
                    # _train_delta_pred = np.array(_tmp_train_x_list) * _a + _b
                    _train_pred = _train_base_perf - _train_delta_pred
                    _train_mse = np.mean((_train_pred - _train_truth) ** 2)
                    # _test_delta_pred = np.array(_tmp_test_x_list) * _a + _b
                    _test_pred = _train_base_perf - _test_delta_pred
                    _test_mse = np.mean((_test_pred - _test_truth) ** 2)
                    _res_info[f'{_prefix_str}corr_val'] = _corr[0]
                    _res_info[f'{_prefix_str}corr_p'] = _corr[1]
                    # _res_info[f'{_prefix_str}polyfit'] = [_a, _b]
                    _res_info[f'{_prefix_str}lr_params'] = _lr.coef_.tolist()
                    _res_info[f'{_prefix_str}train_mse'] = _train_mse
                    _res_info[f'{_prefix_str}test_mse'] = _test_mse
                    # mae
                    _train_mae = np.mean(np.abs(_train_pred - _train_truth))
                    _test_mae = np.mean(np.abs(_test_pred - _test_truth))
                    _res_info[f'{_prefix_str}train_mae'] = _train_mae
                    _res_info[f'{_prefix_str}test_mae'] = _test_mae
                    # log mae mse of train test
                    _global_logger.info(f'train mae: {_train_mae}, mse: {_train_mse}')
                    _global_logger.info(f'test mae: {_test_mae}, mse: {_test_mse}')
                    _res_info[f'{_prefix_str}train_pred'] = _train_pred.tolist()
                    _res_info[f'{_prefix_str}test_pred'] = _test_pred.tolist()
        else:
            # separate single feature
            _res_info = {}
            _tmp_train_x_list = _train_x_list
            _tmp_test_x_list = _test_x_list
            # recover y to acc by _train_base_f1 - delta
            _tmp_train_y_list = _train_base_perf - np.array(_train_f1_delta_list)
            _tmp_test_y_list = _train_base_perf - np.array(_test_f1_delta_list)
            
            if _tmp_args.estimation_feature == 'token_diff':
                _prefix_list = ['']
                _tmp_train_x_list = [_tmp_train_x_list]
                _tmp_test_x_list = [_tmp_test_x_list]
            elif _tmp_args.estimation_feature in ['uc', 'uc_logits']:
                _prefix_list = UtilsMetric._uc_metric_list
            elif _tmp_args.estimation_feature == 'confidence':
                _prefix_list = ['confidence_0.9', 'confidence_0.8', 'confidence_0.7']
            for _idx, _prefix_label in enumerate(_prefix_list):
                if _prefix_label == '':
                    _prefix_train_x_list = _tmp_train_x_list
                    _prefix_test_x_list = _tmp_test_x_list
                else:
                    # each row idx column
                    _prefix_train_x_list = [_x[_idx] for _x in _tmp_train_x_list]
                    _prefix_test_x_list = [_x[_idx] for _x in _tmp_test_x_list]
                    pass
                # reshape
                _prefix_train_x_list = np.reshape(_prefix_train_x_list, (-1, 1))
                _prefix_test_x_list = np.reshape(_prefix_test_x_list, (-1, 1))
                if _tmp_args.estimation_model == 'knn':
                    _prefix_res_info = UtilsEstimationConfidenceProfile.estimate_KNN(
                        _args = _tmp_args,
                        _train_info=[_prefix_train_x_list, _tmp_train_y_list],
                        _test_info=[_prefix_test_x_list, _tmp_test_y_list],
                        _logger=_global_logger
                    )
                elif _tmp_args.estimation_model == 'mlp':
                    # separate valid
                    _mlp_valid_len = len(_prefix_train_x_list) // 5
                    _mlp_train_len = len(_prefix_train_x_list) - _mlp_valid_len
                    _mlp_train_info = [_prefix_train_x_list[:_mlp_train_len], _tmp_train_y_list[:_mlp_train_len]]
                    _mlp_valid_info = [_prefix_train_x_list[_mlp_train_len:], _tmp_train_y_list[_mlp_train_len:]]
                    _prefix_res_info = UtilsEstimationConfidenceProfile.estimate_MLP(
                        _args = _tmp_args,
                        _train_info=_mlp_train_info,
                        _valid_info=_mlp_valid_info,
                        _test_info=[_prefix_test_x_list, _test_f1_delta_list],
                        _logger=_global_logger
                    )
                elif _tmp_args.estimation_model == 'xgboost':
                    _prefix_res_info = UtilsEstimationConfidenceProfile.estimate_XGBoost(
                        _args = _tmp_args,
                        _train_info=[_prefix_train_x_list, _tmp_train_y_list],
                        _test_info=[_prefix_test_x_list, _tmp_test_y_list],
                        _logger=_global_logger
                    )
                # update res_info, map k with prefix
                for _k, _v in _prefix_res_info.items():
                    _prefix_str = f'{_prefix_label}_'
                    if _prefix_label == '':
                        _prefix_str = ''
                    _res_info[f'{_prefix_str}{_k}'] = _v
        # save _res_es_file
        open(_res_es_file, 'w+').write(json.dumps(_res_info))
        _global_logger.info(f'result saved to {_res_es_file}')
        return {
            'res_info': _res_info,
            'cache_info': _cache_info
        }
class UtilsEstimationMix():
    # multi var, linear regression
    @staticmethod
    def calc_estimation_res(_tmp_args, _total_len, _res_path_list, _res_args_list, _global_logger, _use_cache=False):
        # feature list from args.estimation_feature_list
        _es_path_info = UtilsTask.get_es_info(_tmp_args)
        _cache_file = _es_path_info['cache_pkl']
        _res_es_file = _es_path_info['res_info']
        if _use_cache and os.path.exists(_cache_file):
            _cache_info = pickle.load(open(_cache_file, 'rb'))
            _global_logger.info(f'cache loaded from {_cache_file}')
        else:
            _meta_feature_info = UtilsMetric.extract_combine_meta_feature(
                _tmp_args = _tmp_args,
                _total_len = _total_len,
                _res_path_list = _res_path_list,
                _res_args_list = _res_args_list,
                _global_logger = _global_logger,
            )
            _cache_info = _meta_feature_info
            open(_cache_file, 'wb').write(pickle.dumps(_cache_info))
            _global_logger.info(f'cache saved to {_cache_file}')
        # deconstruct cache info
        _train_x_list = _cache_info['train_x_list']
        _train_y_list = _cache_info['train_y_list']
        _test_x_list = _cache_info['test_x_list']
        _test_y_list = _cache_info['test_y_list']
        # log x y shape
        _global_logger.info(f'train x shape: {np.array(_train_x_list).shape}, train y shape: {np.array(_train_y_list).shape}')
        
        if _tmp_args.estimation_model == 'mlr':
            _res_info = UtilsEstimationConfidenceProfile.estimate_LR(
                _args=_tmp_args,
                _train_info=[_train_x_list, _train_y_list],
                _test_info=[_test_x_list, _test_y_list],
                _logger=_global_logger
            )
        elif _tmp_args.estimation_model == 'mknn':
            _res_info = UtilsEstimationConfidenceProfile.estimate_KNN(
                _args=_tmp_args,
                _train_info = [_train_x_list, _train_y_list],
                _test_info = [_test_x_list, _test_y_list],
                _logger=_global_logger
            )
        elif _tmp_args.estimation_model == 'mmlp':
            # 1 / 4 as valid
            _mlp_valid_len = len(_train_x_list) // 5
            _mlp_train_len = len(_train_x_list) - _mlp_valid_len
            _mlp_train_info = [_train_x_list[:_mlp_train_len], _train_y_list[:_mlp_train_len]]
            _mlp_valid_info = [_train_x_list[_mlp_train_len:], _train_y_list[_mlp_train_len:]]
            _res_info = UtilsEstimationConfidenceProfile.estimate_MLP(
                _args=_tmp_args,
                _train_info=_mlp_train_info,
                _valid_info=_mlp_valid_info,
                _test_info=[_test_x_list, _test_y_list],
                _logger=_global_logger
            )
        elif _tmp_args.estimation_model == 'mxgboost':
            _res_info = UtilsEstimationConfidenceProfile.estimate_XGBoost(
                _args=_tmp_args,
                _train_info=[_train_x_list, _train_y_list],
                _test_info=[_test_x_list, _test_y_list],
                _logger=_global_logger
            )
        open(_res_es_file, 'w+').write(json.dumps(_res_info))
        _global_logger.info(f'result saved to {_res_es_file}')
        return {
            'res_info': _res_info,
            'cache_info': _cache_info
        }
    pass
class UtilsEstimationTS():
    @staticmethod
    def calc_estimation_res(_tmp_args, _total_len, _res_path_list, _res_args_list, _global_logger, _use_cache=False):
        _es_path_info = UtilsTask.get_es_info(_tmp_args)
        _cache_file = _es_path_info['cache_pkl']
        _res_es_file = _es_path_info['res_info']
        if _use_cache and os.path.exists(_cache_file):
            _cache_info = pickle.load(open(_cache_file, 'rb'))
            _global_logger.info(f'cache loaded from {_cache_file}')
        else:
            _meta_feature_info = UtilsMetric.extract_meta_feature(
                _tmp_args = _tmp_args,
                _total_len = _total_len,
                _res_path_list = _res_path_list,
                _res_args_list = _res_args_list,
                _global_logger = _global_logger,
            )
            # prepare cache info with train_logits_list, status_list, test_logits_list, status_list
            _cache_info = {
                'train_logits_list': _meta_feature_info['train_base_feature_info']['train_logits_list'],
                'train_status_list': _meta_feature_info['train_base_feature_info']['status_list'],
                'test_logits_list': [
                    _feature['test_logits_list'] for _feature in _meta_feature_info['test_feature_info_list']
                ],
                'test_status_list': [
                    _feature['status_list'] for _feature in _meta_feature_info['test_feature_info_list']
                ]
            }
            open(_cache_file, 'wb').write(pickle.dumps(_cache_info))
            _global_logger.info(f'cache saved to {_cache_file}')
        # load from _res_es_file if exists
        if os.path.exists(_res_es_file) and _use_cache:
            _res_info = json.loads(open(_res_es_file, 'r').read())
            _global_logger.info(f'result loaded from {_res_es_file}')
            return {
                'res_info': _res_info,
                'cache_info': _cache_info
            }
        # deconstruct cache info
        _train_logits_list = _cache_info['train_logits_list']
        _train_status_list = _cache_info['train_status_list']
        _test_logits_list = _cache_info['test_logits_list']
        _test_status_list = _cache_info['test_status_list']
        # search best temp in for temp in np.linspace(1.0, 3.0, 100):
        # try temperature scaling on train, find best temp with min mae
        _best_temp = None
        _best_train_mae = None
        _train_base_acc = np.mean(_train_status_list)
        for _candi_temp in np.linspace(1.0, 3.0, 100):
            # for each logit item in train logit list, scaled by temp + softmax
            _scaled_train_logit_list = [
                torch.nn.functional.softmax(torch.tensor(_logit) / _candi_temp, dim=0).tolist()
                for _logit in _train_logits_list
            ]
            if _tmp_args.task_type == 'generation':
                # _scaled_train_logit_list M * N, trucate different len N to shortest len
                _min_len = min([len(_logit) for _logit in _scaled_train_logit_list])
                _scaled_train_logit_list = [_logit[:_min_len] for _logit in _scaled_train_logit_list]
            _max_scaled_logits = np.array(_scaled_train_logit_list).max(axis=1)
            _candi_test_pred = np.mean(_max_scaled_logits)
            _candi_mae = mean_absolute_error([_train_base_acc], [_candi_test_pred])
            if _best_temp is None or _candi_mae < _best_train_mae:
                _best_temp = _candi_temp
                _best_train_mae = _candi_mae
        # use best temp to scale test logits
        _scaled_test_logits_list = [
            [
                torch.nn.functional.softmax(torch.tensor(_logit) / _best_temp, dim=0).tolist()
                for _logit in _logit_list_group
            ]
            for _logit_list_group in _test_logits_list
        ]
        if _tmp_args.task_type == 'generation':
            # trunc also here _scaled_test_logits_list M * N * L, align L
            _min_l = min([len(_logit) for _logit_list in _scaled_test_logits_list for _logit in _logit_list])
            _scaled_test_logits_list = [
                [
                    _logit[:_min_l] for _logit in _logit_list_group
                ]
                for _logit_list_group in _scaled_test_logits_list
            ]
        _test_pred_list = np.array(_scaled_test_logits_list).max(axis=2).mean(axis=1).tolist()
        _test_acc_list = [
            np.mean(_status_list) for _status_list in _test_status_list
        ]
        _test_mae = mean_absolute_error(_test_acc_list, _test_pred_list)
        _test_mse = mean_squared_error(_test_acc_list, _test_pred_list)
        _res_info = {
            'best_temp': _best_temp,
            'test_mae': _test_mae,
            'test_mse': _test_mse,
            'test_pred': _test_pred_list,
            'test_truth': _test_acc_list,
            'args': _tmp_args.__dict__
        }
        _global_logger.info(f'best temp: {_best_temp}, test mae: {_test_mae}, test mse: {_test_mse}')
        open(_res_es_file, 'w+').write(json.dumps(_res_info))
        _global_logger.info(f'result saved to {_res_es_file}')
        return {
            'res_info': _res_info,
            'cache_info': _cache_info
        }
class UtilsEstimationATC():
    @staticmethod
    def calc_estimation_res(_tmp_args, _total_len, _res_path_list, _res_args_list, _global_logger, _use_cache=False):
        _es_path_info = UtilsTask.get_es_info(_tmp_args)
        _cache_file = _es_path_info['cache_pkl']
        _res_es_file = _es_path_info['res_info']
        if _use_cache and os.path.exists(_cache_file):
            _cache_info = pickle.load(open(_cache_file, 'rb'))
            _global_logger.info(f'cache loaded from {_cache_file}')
        else:
            _meta_feature_info = UtilsMetric.extract_meta_feature(
                _tmp_args = _tmp_args,
                _total_len = _total_len,
                _res_path_list = _res_path_list,
                _res_args_list = _res_args_list,
                _global_logger = _global_logger,
            )
            _cache_info = {
                'train_feature_info_list': _meta_feature_info['train_feature_info_list'],
                'test_feature_info_list': _meta_feature_info['test_feature_info_list']
            }
            open(_cache_file, 'wb').write(pickle.dumps(_cache_info))
            _global_logger.info(f'cache saved to {_cache_file}')
        # load _res_es_file if need
        if os.path.exists(_res_es_file) and _use_cache:
            _res_info = json.loads(open(_res_es_file, 'r').read())
            _global_logger.info(f'result loaded from {_res_es_file}')
            return {
                'res_info': _res_info,
                'cache_info': _cache_info
            }
        _train_feature_info_list = _cache_info['train_feature_info_list']
        _test_feature_info_list = _cache_info['test_feature_info_list']
        # calc atc threshold in each train batch
        _train_threshold_list = []
        for _train_feature in _train_feature_info_list:
            if _tmp_args.task_type == 'classification':
                _logit_list = np.array(_train_feature['train_logits_list'])
            elif _tmp_args.task_type == 'generation':
                _logit_list = _train_feature['train_logits_list']
                # align _logit_list M * Nn min N
                _min_len = min([len(_logit) for _logit in _logit_list])
                _logit_list = [_logit[:_min_len] for _logit in _logit_list]
            _softmax_logit_list = [
                torch.nn.functional.softmax(torch.tensor(_logit), dim=0).tolist()
                for _logit in _logit_list
            ]
            _max_conf_list = np.array(_softmax_logit_list).max(axis=1).tolist()
            # rank from top to bottom
            _ranked_conf_list = sorted(_max_conf_list, reverse=True)
            # calc cur acc
            _cur_acc = np.mean(_train_feature['status_list'])
            # pick out threshold with index around len * acc
            _threshold_idx = int(len(_ranked_conf_list) * _cur_acc)
            _threshold = _ranked_conf_list[_threshold_idx]
            _train_threshold_list.append(_threshold)
        # iterate test, use threshold list to estimate acc, avg then
        _test_pred_acc_list = []
        _test_turth_acc_list = []
        for _test_feature in _test_feature_info_list:
            if _tmp_args.task_type == 'classification':
                _logit_list = np.array(_test_feature['test_logits_list'])
            elif _tmp_args.task_type == 'generation':
                _logit_list = _test_feature['test_logits_list']
                _min_len = min([len(_logit) for _logit in _logit_list])
                _logit_list = [_logit[:_min_len] for _logit in _logit_list]
            # _logit_list = np.array(_test_feature['test_logits_list'])
            _softmax_logit_list = [
                torch.nn.functional.softmax(torch.tensor(_logit), dim=0).tolist()
                for _logit in _logit_list
            ]
            _max_conf_list = np.array(_softmax_logit_list).max(axis=1).tolist()
            _tmp_acc_pred_list = []
            for _threshold in _train_threshold_list:
                _tmp_acc_pred = np.mean([1 if _v > _threshold else 0 for _v in _max_conf_list])
                _tmp_acc_pred_list.append(_tmp_acc_pred)
            _avg_acc_pred = np.mean(_tmp_acc_pred_list)
            _test_pred_acc_list.append(_avg_acc_pred)
            _test_turth_acc_list.append(np.mean(_test_feature['status_list']))
        _test_mae = mean_absolute_error(_test_turth_acc_list, _test_pred_acc_list)
        _test_mse = mean_squared_error(_test_turth_acc_list, _test_pred_acc_list)
        _res_info = {
            'test_mae': _test_mae,
            'test_mse': _test_mse,
            'test_pred': _test_pred_acc_list,
            'test_truth': _test_turth_acc_list,
            'args': _tmp_args.__dict__
        }
        _global_logger.info(f'test mae: {_test_mae}, test mse: {_test_mse}')
        open(_res_es_file, 'w+').write(json.dumps(_res_info))
        _global_logger.info(f'result saved to {_res_es_file}')
    pass
class ConfidenceProfileMLP(torch.nn.Module):
    # dropout default 0.0, lr default 1e-4, lr_lambda 0.9, num_epochs 10, layers 2
    def __init__(self, _input_dim, _args, _hidden_dim=1536, _layer_num=2, _sigmoid=True, _dropout=0.0, _lr=1e-4, _lr_lambda=0.9, _num_epochs=10):
        super().__init__()
        self._input_dim = _input_dim
        self._args = _args
        self._lr = _lr
        self._lr_lambda = _lr_lambda
        self._num_epochs = _num_epochs
        _layer_list = []
        # input to hidden
        _layer_list.append(torch.nn.Linear(_input_dim, _hidden_dim))
        # dropout
        _layer_list.append(torch.nn.Dropout(_dropout))
        # relu
        _layer_list.append(torch.nn.ReLU())
        # hidden to hidden
        for _ in range(_layer_num - 1):
            _layer_list.append(torch.nn.Linear(_hidden_dim, _hidden_dim))
            _layer_list.append(torch.nn.Dropout(_dropout))
            _layer_list.append(torch.nn.ReLU())
        _layer_list.append(torch.nn.Linear(_hidden_dim, 1))
        # dropout
        _layer_list.append(torch.nn.Dropout(_dropout))
        if _sigmoid:
            _layer_list.append(torch.nn.Sigmoid())
        self.model = torch.nn.Sequential(*_layer_list)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.initialize_weights()
        self.to(self.device)
    def initialize_weights(self):
        for _m in self.modules():
            if isinstance(_m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(_m.weight)
                _m.bias.data.fill_(0.01)
        pass
    def forward(self, _x):
        return self.model(_x)
    def train_with_data(self, _optimizer, _loss_func, _critertion, _scheduler, 
                        _train_x, _train_y, _valid_x, _valid_y, 
                        _test_info, _logger):
        _best_valid_loss = float('inf')
        _patience = 7
        _train_loss_list = []
        _valid_loss_list = []
        _best_model = None
        for _epoch in range(0, self._num_epochs):
            _epoch_train_loss = []
            _epoch_valid_loss = []
            for _idx, _data in enumerate(zip(_train_x, _train_y)):
                _optimizer.zero_grad()
                _x, _y = _data
                _output = self(_x)
                _loss = _loss_func(_output, _y)
                _loss.backward()
                _optimizer.step()
                _epoch_train_loss.append(_loss.item())
            # validate
            self.eval()
            for _idx, _data in enumerate(zip(_valid_x, _valid_y)):
                _x, _y = _data
                _output = self(_x)
                _loss = _critertion(_output, _y)
                _epoch_valid_loss.append(_loss.item())
            # avg loss
            _avg_epoch_train_loss = np.mean(_epoch_train_loss)
            _avg_epoch_valid_loss = np.mean(_epoch_valid_loss)
            _train_loss_list.append(_avg_epoch_train_loss)
            _valid_loss_list.append(_avg_epoch_valid_loss)
            if _avg_epoch_valid_loss < _best_valid_loss:
                _best_valid_loss = _avg_epoch_valid_loss
                _best_model = copy.deepcopy(self.state_dict())
                _patience = 7
            else:
                _patience -= 1
            if _patience == 0:
                _logger.info(f'Early stopping at epoch {_epoch}, train loss: {_avg_epoch_train_loss:.3f}, valid loss: {_avg_epoch_valid_loss:.3f}')
                break
            self.train()
            _scheduler.step()
        # load best model
        self.load_state_dict(_best_model)
        # test
        _test_x = torch.tensor(_test_info[0], dtype=torch.float32).to(self.device)
        _test_y = torch.tensor(_test_info[1], dtype=torch.float32).to(self.device)
        _test_loss_list = []
        _test_y_pred_list = []
        for _idx, _data in enumerate(zip(_test_x, _test_y)):
            _x, _y = _data
            with torch.no_grad():
                _output = self(_x)
                _loss = _critertion(_output, _y)
                _test_loss_list.append(_loss.item())
                _test_y_pred_list.append(_output.item())
        # log
        _test_mae = mean_absolute_error(_test_info[1], _test_y_pred_list)
        _test_mse = mean_squared_error(_test_info[1], _test_y_pred_list)
        _logger.info(f'test loss: {np.mean(_test_loss_list):.3f}, mae: {_test_mae:.3f}, mse: {_test_mse:.3f}')
        # also eval train, valid mae, mse
        _train_y_pred_list = []
        for _idx, _data in enumerate(zip(_train_x, _train_y)):
            _x, _y = _data
            with torch.no_grad():
                _output = self(_x)
                _train_y_pred_list.append(_output.item())
        _train_mae = mean_absolute_error(_train_y.tolist(), _train_y_pred_list)
        _train_mse = mean_squared_error(_train_y.tolist(), _train_y_pred_list)
        _logger.info(f'train mae: {_train_mae:.3f}, mse: {_train_mse:.3f}')
        _valid_y_pred_list = []
        for _idx, _data in enumerate(zip(_valid_x, _valid_y)):
            _x, _y = _data
            with torch.no_grad():
                _output = self(_x)
                _valid_y_pred_list.append(_output.item())
        _valid_mae = mean_absolute_error(_valid_y.tolist(), _valid_y_pred_list)
        _valid_mse = mean_squared_error(_valid_y.tolist(), _valid_y_pred_list)
        _logger.info(f'valid mae: {_valid_mae:.3f}, mse: {_valid_mse:.3f}')
        
        return {
            'test_pred': _test_y_pred_list,
            'test_truth': _test_info[1],
            'test_mae': _test_mae,
            'test_mse': _test_mse,
            'train_pred': _train_y_pred_list,
            'train_truth': _train_y.tolist(),
            'train_mae': _train_mae,
            'train_mse': _train_mse,
            'valid_pred': _valid_y_pred_list,
            'valid_truth': _valid_y.tolist(),
            'valid_mae': _valid_mae,
            'valid_mse': _valid_mse
        }
    def train_estimate(self, _train_info, _valid_info, _test_info, _logger):
        self.train()
        _train_x = torch.tensor(_train_info[0], dtype=torch.float32).to(self.device)
        _train_y = torch.tensor(_train_info[1], dtype=torch.float32).to(self.device)
        _valid_x = torch.tensor(_valid_info[0], dtype=torch.float32).to(self.device)
        _valid_y = torch.tensor(_valid_info[1], dtype=torch.float32).to(self.device)
        _loss_func = torch.nn.MSELoss()
        _critertion = torch.nn.L1Loss()
        _optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        _scheduler = torch.optim.lr_scheduler.LambdaLR(_optimizer, lr_lambda=lambda _epoch: 0.65 ** _epoch)
        if self._args.k_fold:
            # merge train/valid, re-split with kfold
            _train_valid_x = torch.cat([_train_x, _valid_x], dim=0)
            _train_valid_y = torch.cat([_train_y, _valid_y], dim=0)
            _kfold = KFold(n_splits=self._args.k_fold, shuffle=True, random_state=self._args.seed)
            _best_model_state_dict = None
            _best_valid_mae = None
            _best_test_res_info = None
            for _fold_idx, (_train_idx, _valid_idx) in enumerate(_kfold.split(_train_valid_x)):
                _fold_train_x = _train_valid_x[_train_idx]
                _fold_train_y = _train_valid_y[_train_idx]
                _fold_valid_x = _train_valid_x[_valid_idx]
                _fold_valid_y = _train_valid_y[_valid_idx]
                # init params with init
                self.initialize_weights()
                _fold_res_info = self.train_with_data(
                    _optimizer = _optimizer,
                    _loss_func = _loss_func,
                    _critertion = _critertion,
                    _scheduler = _scheduler,
                    _train_x = _fold_train_x,
                    _train_y = _fold_train_y,
                    _valid_x = _fold_valid_x,
                    _valid_y = _fold_valid_y,
                    _test_info = _test_info,
                    _logger = _logger
                )
                # best with min
                if _best_valid_mae is None or _fold_res_info['valid_mae'] < _best_valid_mae:
                    _best_valid_mae = _fold_res_info['valid_mae']
                    _best_model_state_dict = copy.deepcopy(self.state_dict())
                    _best_test_res_info = _fold_res_info
            return _best_test_res_info
        else:
            return self.train_with_data(
                _optimizer = _optimizer,
                _loss_func = _loss_func,
                _critertion = _critertion,
                _scheduler = _scheduler,
                _train_x = _train_x,
                _train_y = _train_y,
                _valid_x = _valid_x,
                _valid_y = _valid_y,
                _test_info = _test_info,
                _logger = _logger
            )
