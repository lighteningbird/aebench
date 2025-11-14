import os, sys, logging, uuid, json, torch, numpy as np, copy, pickle
from sklearn.metrics import f1_score
from datetime import datetime
from scipy.stats import pearsonr

from src.estimation.utils import UtilsTask, UtilsLLM, UtilsDataset, UtilsEstimationSingle, UtilsEstimationAvgTrain, UtilsEstimationAvgConf, UtilsEstimationTS, UtilsEstimationATC, UtilsEstimationConfidenceProfile, UtilsEstimationMix


_global_logger = None
def initlize_logger():
    global _global_logger
    _logger = logging.getLogger()
    if not os.path.exists('logs'):
        os.makedirs('logs')
    _cur_time_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    _uuid_str = str(uuid.uuid4())
    _log_file = f'logs/{_cur_time_str}-{_uuid_str}.log'
    print(f'Log file: {_log_file}')
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%a, %d %b %Y %H:%M:%S', filename=_log_file, filemode='w')
    _global_logger = _logger
    pass

def generation_run(_args):
    # CUDA_VISIBLE_DEVICES=2 nohup python -u src/estimation/run.py --mode=generation --is_debug=True --task_type=classification --dataset=imdb --model_name=llama3-8b --max_new_tokens=256 --prompt_type=zero_shot > logs/imdb.log 2>&1 &
    # CUDA_VISIBLE_DEVICES=1 nohup python -u src/estimation/run.py --mode=generation --is_debug=True --task_type=classification --dataset=cosmosqa_10k --model_name=llama3-8b --max_new_tokens=256 --prompt_type=zero_shot > logs/cosmosqa_10k.log 2>&1 &
    # CUDA_VISIBLE_DEVICES=0 nohup python -u src/estimation/run.py --mode=generation --is_debug=True --task_type=generation --dataset=squad_v2 --model_name=llama3-8b --max_new_tokens=256 --prompt_type=zero_shot > logs/squad_v2.log 2>&1 &
    # CUDA_VISIBLE_DEVICES=2 python -u src/estimation/run.py --mode=generation --is_debug=True --task_type=generation --dataset=squad_v2 --model_name=llama3-8b --max_new_tokens=256 --prompt_type=zero_shot --llm_param_list 0.5 0.7 0 --dataset_range 0 1000
    _global_logger.info('generation mode')
    # load llm to generate data
    _model_info = UtilsLLM.load_model(_args)
    _data_path = UtilsDataset.get_data_path(_args)
    # log _data_path
    _global_logger.info(f'_data_path: {_data_path}')
    _res_dir = UtilsTask.get_res_dir(_args)
    if not os.path.exists(_res_dir):
        os.makedirs(_res_dir, exist_ok=True)
    _data_idx = 0
    _param_list = copy.deepcopy(UtilsLLM._preset_param_list)
    # read from args llm_param_list if valid 
    if hasattr(_args, 'llm_param_list') and _args.llm_param_list is not None and len(_args.llm_param_list) == 3:
        # single param
        _param_list = [
            {
                'temperature': _args.llm_param_list[0],
                'top_p': _args.llm_param_list[1],
                'top_k': int(_args.llm_param_list[2])
            }
        ]
    _range_start = None
    _range_end = None
    # check args range list, valid if len 2
    if 'dataset_range' in _args and _args.dataset_range is not None and len(_args.dataset_range) == 2:
        _range_start = _args.dataset_range[0]
        _range_end = _args.dataset_range[1]
    # prepare param-res-key: temp-{}-topp-{}-topk-{}, read existing
    _existing_res = {}
    for _param in _param_list:
        _tmp_args = copy.deepcopy(_args)
        # _tmp_args.llm_param_list = [_param['temperature'], _param['top_p'], _param['top_k']]
        # float temp, top_p
        _tmp_args.llm_param_list = [float(_param['temperature']), float(_param['top_p']), int(_param['top_k'])]
        _res_path = UtilsTask.get_res_path(_tmp_args)
        # _param_key = f'temp-{_param["temperature"]}-topp-{_param["top_p"]}-topk-{_param["top_k"]}{_range_key}'
        _param_key = _res_path.split('/')[-1].replace('.jsonl', '')
        _existing_len = 0
        if os.path.exists(_res_path):
            _existing_len = open(_res_path, 'r').read().count('\n')
        _existing_res[_param_key] = _existing_len
    if _args.task_type == 'classification':
        _opt_txt_list = UtilsDataset.get_classification_choice_list(_args)
        _opt_token_id_list = [
            _model_info['tokenizer'].encode(f'{_opt}')[-1] for _opt in _opt_txt_list
        ]
    _is_finished = False
    with open(_data_path, 'r') as _f:
        for _data_str in _f:
            if _is_finished:
                break
            _data_item = json.loads(_data_str)
            _input_ids = UtilsLLM.get_model_input_ids(_args, _data_item, _model_info)
            for _param in _param_list:
                _param_args = copy.deepcopy(_args)
                # _param_args.llm_param_list = [_param['temperature'], _param['top_p'], _param['top_k']]
                # float
                _param_args.llm_param_list = [float(_param['temperature']), float(_param['top_p']), int(_param['top_k'])]
                _param_res_path = UtilsTask.get_res_path(_param_args)
                _param_res_key = _param_res_path.split('/')[-1].replace('.jsonl', '')
                if _range_start is not None and _range_end is not None:
                    if _range_start is not None and _data_idx < _range_start:
                        _global_logger.info(f'Skip data idx: {_data_idx} for param: {_param_res_key}')
                        continue
                    elif _range_end is not None and _data_idx >= _range_end:
                        _global_logger.info(f'Finished data idx: {_data_idx} for param: {_param_res_key}')
                        _is_finished = True
                        break
                    elif _data_idx < _existing_res[_param_res_key] + _range_start:
                        _global_logger.info(f'Skip existing data idx: {_data_idx} for param: {_param_res_key}')
                        continue
                elif _data_idx < _existing_res[_param_res_key]:
                    _global_logger.info(f'Skip data idx: {_data_idx} for param: {_param_res_key}')
                    continue
                if _data_idx >= 10000 and _args.is_debug:
                    # finish temp
                    _global_logger.info(f'Finish data idx: {_data_idx} for param: {_param_res_key}')
                    _is_finished = True
                    break
                _top_20_logits_list = [] # pure value list
                # clone from tensor
                _generated_ids = _input_ids.clone()
                _res_item = {
                    'data_idx': _data_idx,
                    # 'param': _param
                }
                if 'answer' in _data_item:
                    _res_item['answer'] = _data_item['answer']
                else:
                    _res_item['answer'] = UtilsDataset.get_item_answer(_args, _data_item)
                _log_probs = []
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        for _token_idx in range(_args.max_new_tokens):
                            _outputs = _model_info['model'](_generated_ids)
                            _logits = _outputs.logits
                            if _args.task_type == 'classification':
                                _opt_logit_raw = _logits[:, -1, :].squeeze(0)[_opt_token_id_list].cpu().tolist()
                                if 'choice_logits_raw' not in _res_item:
                                    _res_item['choice_logits_raw'] = []
                                if 'choice_text_softmax' not in _res_item:
                                    _res_item['choice_text_softmax'] = []
                                _opt_text_softmax = _opt_txt_list[np.argmax(_opt_logit_raw)]
                                # append raw logit
                                _res_item['choice_logits_raw'].append(_opt_logit_raw)
                                _res_item['choice_text_softmax'].append(_opt_text_softmax)
                            # from _logits[:, -1, :], top 20 in each token
                            _max_20_logits = torch.topk(_logits[:, -1, :], 20, dim=-1)
                            _top_20_logits_list.append([_tv.tolist()[0] for _tv in _max_20_logits])
                            _logits = _logits[:, -1, :] / _param['temperature']
                            # log_probs.append(F.log_softmax(scaled_logits, dim=-1))
                            if _args.task_type == 'generation':
                                _log_probs.append(torch.nn.functional.log_softmax(_logits, dim=-1))
                            _probs = torch.nn.functional.softmax(_logits, dim=-1)
                            if _param['top_p'] < 1.0:
                                _sorted_probs, _sorted_indices = torch.sort(_probs, descending=True)
                                _cumulative_probs = torch.cumsum(_sorted_probs, dim=-1)
                                _sorted_indices_to_remove = _cumulative_probs > _param['top_p']
                                _sorted_indices_to_remove[..., 1:] = _sorted_indices_to_remove[..., :-1].clone()
                                _sorted_indices_to_remove[..., 0] = 0
                                _indices_to_remove = _sorted_indices_to_remove.scatter(1, _sorted_indices, _sorted_indices_to_remove)
                                _probs = _probs.masked_fill(_indices_to_remove, 0.0)
                                _probs = _probs / _probs.sum(dim=-1, keepdim=True)
                            if _param['top_k'] > 0:
                                _top_k_probs, _top_k_indices = torch.topk(_probs, _param['top_k'], dim=-1)
                                _probs = torch.zeros_like(_probs).scatter_(-1, _top_k_indices, _top_k_probs)
                                _probs = _probs / _probs.sum(dim=-1, keepdim=True)
                            _next_token = torch.multinomial(_probs, num_samples=1)
                            _generated_ids = torch.cat([_generated_ids, _next_token], dim=-1)
                            if _next_token.item() in [_model_info['tokenizer'].eos_token_id, _model_info['tokenizer'].convert_tokens_to_ids("")]:
                                break
                            if _args.task_type == 'classification' and _args.is_debug:
                                # just stop
                                break
                            del _outputs, _logits, _next_token
                            torch.cuda.empty_cache()
                _generated_tokens = _generated_ids[0][_input_ids.shape[-1]:]
                _response = _model_info['tokenizer'].decode(_generated_tokens, skip_special_tokens=True)
                _res_item['response'] = _response
                _res_item['top_20_logits'] = _top_20_logits_list
                if _args.task_type == 'generation':
                    _nll_v = 0.0
                    for _t, _token_log_probs in enumerate(_log_probs):
                        _target_token_id = _generated_ids[0, _t + 1].item()
                        _nll_v -= _token_log_probs[0, _target_token_id].item()
                    _res_item['nll'] = _nll_v
                open(_param_res_path, 'a').write(json.dumps(_res_item) + '\n')
                _global_logger.info(f'data idx: {_data_idx}, param: {_param_res_key}, response: {_response}')
            _data_idx += 1
            
    pass

def estimation_run(_args):
    if _args.dataset == 'cosmosqa_10k':
        _target_dataset_list = UtilsDataset.get_choice_dataset_list()
    else:
        _target_dataset_list = [_args.dataset]
    # for _dataset_name in UtilsDataset._dataset_name_list:
    # for _dataset_name in UtilsDataset.get_choice_dataset_list():
    # for _dataset_name in ['halu_dialogue']:
    for _dataset_name in _target_dataset_list:
        _tmp_args = copy.deepcopy(_args)
        _tmp_args.dataset = _dataset_name
        _tmp_args.mode = 'generation'
        _target_param_list = UtilsLLM._preset_param_list
        if _args.is_debug or _dataset_name in UtilsDataset.get_choice_dataset_list():
            _target_param_list = [_target_param_list[0]]
        for _param in _target_param_list:
            # _tmp_args.llm_param_list = [_param['temperature'], _param['top_p'], _param['top_k']]
            # float temp, top_p
            _tmp_args.llm_param_list = [float(_param['temperature']), float(_param['top_p']), int(_param['top_k'])]
            # log param
            _global_logger.info(f'{_dataset_name} Estimation param: {_tmp_args.llm_param_list}')
            _target_len = UtilsDataset._len_info[_tmp_args.dataset]
            if _target_len >= 10000 or _tmp_args.is_debug:
                _target_len = 10000
            _res_path_list = []
            _res_args_list = []
            if _tmp_args.dataset in UtilsDataset._slurm_split_info:
                _step_size = UtilsDataset._slurm_split_info[_tmp_args.dataset]
                _step_count = _target_len // _step_size
                if _target_len % _step_size > 0:
                    _step_count += 1
                for _step_idx in range(_step_count):
                    _tmp_step_args = copy.deepcopy(_tmp_args)
                    _tmp_step_args.dataset_range = [_step_idx * _step_size, (_step_idx + 1) * _step_size]
                    _tmp_res_path = UtilsTask.get_res_path(_tmp_step_args)
                    if _tmp_args.is_debug and os.path.exists(_tmp_res_path + '.bak'):
                        _tmp_res_path = _tmp_res_path + '.bak'
                    _res_path_list.append(_tmp_res_path)
                    _res_args_list.append(_tmp_step_args)
            else:
                _res_path = UtilsTask.get_res_path(_tmp_args)
                if _tmp_args.is_debug and os.path.exists(_res_path + '.bak'):
                    _res_path = _res_path + '.bak'
                _res_path_list.append(_res_path)
                _res_args_list.append(_tmp_args)
            # check len sum first
            _total_len = 0
            for _res_path in _res_path_list:
                _global_logger.info(f'read res path: {_res_path}')
                _total_len += UtilsTask.count_file_len(_res_path)
            # skip if _total_len not match target_len
            if _total_len != _target_len:
                _global_logger.info(f'Not match target len: {_target_len}, total len: {_total_len}')
                continue
            _use_cache = (_target_len == _total_len) and _tmp_args.use_cache
            if _tmp_args.estimation_feature == 'confidence_profile':
                UtilsEstimationConfidenceProfile.calc_estimation_res(
                    _tmp_args=_tmp_args,
                    _total_len=_total_len,
                    _res_path_list=_res_path_list,
                    _res_args_list=_res_args_list,
                    _global_logger=_global_logger,
                    _use_cache=_use_cache,
                )
            elif _tmp_args.estimation_model in ['lr', 'knn', 'mlp', 'xgboost']:
                UtilsEstimationSingle.calc_estimation_res(
                    _tmp_args=_tmp_args,
                    _total_len=_total_len,
                    _res_path_list=_res_path_list,
                    _res_args_list=_res_args_list,
                    _global_logger=_global_logger,
                    _use_cache=_use_cache,
                )
            elif _tmp_args.estimation_model in ['mlr', 'mknn', 'mmlp', 'mxgboost']:
                UtilsEstimationMix.calc_estimation_res(
                    _tmp_args=_tmp_args,
                    _total_len=_total_len,
                    _res_path_list=_res_path_list,
                    _res_args_list=_res_args_list,
                    _global_logger=_global_logger,
                    _use_cache=_use_cache,
                )
            elif _tmp_args.estimation_feature == 'avg_train':
                UtilsEstimationAvgTrain.calc_estimation_res(
                    _tmp_args=_tmp_args,
                    _total_len=_total_len,
                    _res_path_list=_res_path_list,
                    _res_args_list=_res_args_list,
                    _global_logger=_global_logger,
                    _use_cache=_use_cache,
                )
            elif _tmp_args.estimation_feature == 'avg_conf':
                UtilsEstimationAvgConf.calc_estimation_res(
                    _tmp_args=_tmp_args,
                    _total_len=_total_len,
                    _res_path_list=_res_path_list,
                    _res_args_list=_res_args_list,
                    _global_logger=_global_logger,
                    _use_cache=_use_cache,
                )
            elif _tmp_args.estimation_feature == 'ts':
                UtilsEstimationTS.calc_estimation_res(
                    _tmp_args=_tmp_args,
                    _total_len=_total_len,
                    _res_path_list=_res_path_list,
                    _res_args_list=_res_args_list,
                    _global_logger=_global_logger,
                    _use_cache=_use_cache,
                )
            elif _tmp_args.estimation_feature == 'atc':
                UtilsEstimationATC.calc_estimation_res(
                    _tmp_args=_tmp_args,
                    _total_len=_total_len,
                    _res_path_list=_res_path_list,
                    _res_args_list=_res_args_list,
                    _global_logger=_global_logger,
                    _use_cache=_use_cache,
                )
            _global_logger.info(f'Estimation finished')
            _global_logger.info('-' * 50)
        _global_logger.info('Debug mode not supported')
    
    pass

def estimation_ood_run(_args):
    # split dataset param by -
    _dataset_list = _args.dataset.split('-')
    # assert len 2
    assert len(_dataset_list) == 2
    _source_dataset = _dataset_list[0]
    _target_dataset = _dataset_list[1]
    if _args.task_type == 'classification' or _args.is_debug:
        _target_param_list = [UtilsLLM._preset_param_list[0]]
    else:
        _target_param_list = UtilsLLM._preset_param_list
    for _param in _target_param_list:
        _param_args = copy.deepcopy(_args)
        # generation
        _param_args.mode = 'generation'
        _param_args.llm_param_list = [float(_param['temperature']), float(_param['top_p']), int(_param['top_k'])]
        _target_len_list = []
        _fixed_train_len = 0
        for _dataset_idx, _dataset in enumerate([_source_dataset, _target_dataset]):
            _tmp_len = UtilsDataset._len_info[_dataset]
            if _tmp_len >= 10000 or _args.is_debug:
                _tmp_len = 10000
            if _dataset_idx == 0:
                _fixed_train_len = _tmp_len
            _target_len_list.append(_tmp_len)
        # assert  1 - fixed_train_len / total_len == args estimation_test_ratio
        assert 1 - _fixed_train_len / sum(_target_len_list) == _args.estimation_test_ratio
        _total_len = sum(_target_len_list)
        _res_path_list = []
        _res_args_list = []
        # set dataset #1, #2
        for _dataset_idx, _dataset in enumerate([_source_dataset, _target_dataset]):
            _dataset_args = copy.deepcopy(_param_args)
            _dataset_args.dataset = _dataset
            if _dataset in UtilsDataset._slurm_split_info:
                _step_size = UtilsDataset._slurm_split_info[_dataset]
                _len = _target_len_list[_dataset_idx]
                _step_count = _len // _step_size
                if _len % _step_size > 0:
                    _step_count += 1
                for _step_idx in range(_step_count):
                    _step_args = copy.deepcopy(_dataset_args)
                    _step_args.dataset_range = [_step_idx * _step_size, (_step_idx + 1) * _step_size]
                    _tmp_res_path = UtilsTask.get_res_path(_step_args)
                    _res_path_list.append(_tmp_res_path)
                    _res_args_list.append(_step_args)
            else:
                _tmp_res_path = UtilsTask.get_res_path(_dataset_args)
                _res_path_list.append(_tmp_res_path)
                _res_args_list.append(_dataset_args)
        _total_len_actual = 0
        # calc by read 
        for _res_path in _res_path_list:
            _total_len_actual += UtilsTask.count_file_len(_res_path)
        if _total_len_actual != _total_len:
            _global_logger.info(f'Not match target len: {_total_len}, total len: {_total_len_actual}')
            continue
        _use_cache = _args.use_cache
        _tmp_args = copy.deepcopy(_param_args)
        if _tmp_args.estimation_model in ['lr', 'knn', 'mlp', 'xgboost']:
            if _tmp_args.estimation_feature == 'confidence_profile':
                UtilsEstimationConfidenceProfile.calc_estimation_res(
                    _tmp_args=_tmp_args,
                    _total_len=_total_len,
                    _res_path_list=_res_path_list,
                    _res_args_list=_res_args_list,
                    _global_logger=_global_logger,
                    _use_cache=_use_cache,
                )
            else:
                UtilsEstimationSingle.calc_estimation_res(
                    _tmp_args=_tmp_args,
                    _total_len=_total_len,
                    _res_path_list=_res_path_list,
                    _res_args_list=_res_args_list,
                    _global_logger=_global_logger,
                    _use_cache=_use_cache,
                )
        elif _tmp_args.estimation_model in ['mlr', 'mknn', 'mmlp', 'mxgboost']:
            UtilsEstimationMix.calc_estimation_res(
                _tmp_args=_tmp_args,
                _total_len=_total_len,
                _res_path_list=_res_path_list,
                _res_args_list=_res_args_list,
                _global_logger=_global_logger,
                _use_cache=_use_cache,
            )
        elif _tmp_args.estimation_feature == 'avg_train':
            UtilsEstimationAvgTrain.calc_estimation_res(
                _tmp_args=_tmp_args,
                _total_len=_total_len,
                _res_path_list=_res_path_list,
                _res_args_list=_res_args_list,
                _global_logger=_global_logger,
                _use_cache=_use_cache,
            )
        elif _tmp_args.estimation_feature == 'avg_conf':
            UtilsEstimationAvgConf.calc_estimation_res(
                _tmp_args=_tmp_args,
                _total_len=_total_len,
                _res_path_list=_res_path_list,
                _res_args_list=_res_args_list,
                _global_logger=_global_logger,
                _use_cache=_use_cache,
            )
        elif _tmp_args.estimation_feature == 'ts':
            UtilsEstimationTS.calc_estimation_res(
                _tmp_args=_tmp_args,
                _total_len=_total_len,
                _res_path_list=_res_path_list,
                _res_args_list=_res_args_list,
                _global_logger=_global_logger,
                _use_cache=_use_cache,
            )
        elif _tmp_args.estimation_feature == 'atc':
            UtilsEstimationATC.calc_estimation_res(
                _tmp_args=_tmp_args,
                _total_len=_total_len,
                _res_path_list=_res_path_list,
                _res_args_list=_res_args_list,
                _global_logger=_global_logger,
                _use_cache=_use_cache,
            )
    pass

def debug_run(_args):
    # CUDA_VISIBLE_DEVICES=2 nohup python -u src/estimation/run.py --mode=debug --is_debug=True --task_type=classification --dataset=imdb --model_name=llama3-8b --max_new_tokens=256 --prompt_type=zero_shot > logs/imdb.log 2>&1 &
    # CUDA_VISIBLE_DEVICES=1 nohup python -u src/estimation/run.py --mode=debug --is_debug=True --task_type=classification --dataset=cosmosqa_10k --model_name=llama3-8b --max_new_tokens=256 --prompt_type=zero_shot > logs/cosmosqa_10k.log 2>&1 &
    # CUDA_VISIBLE_DEVICES=0 nohup python -u src/estimation/run.py --mode=debug --is_debug=True --task_type=generation --dataset=squad_v2 --model_name=llama3-8b --max_new_tokens=256 --prompt_type=zero_shot > logs/squad_v2.log 2>&1 &
    # python -u src/estimation/run.py --mode=debug --debug_mode=estimation --is_debug=True --task_type=classification --dataset=cosmosqa_10k --model_name=llama3-8b --max_new_tokens=256 --prompt_type=zero_shot
    _global_logger.info('Debug mode')
    pass

def main_handler():
    global _global_logger
    print(sys.argv)
    _global_logger.info(sys.argv)
    _parser = UtilsTask.get_args_parser()
    _args = _parser.parse_args()
    print(json.dumps(vars(_args)))
    if _args.mode == 'debug':
        debug_run(_args)
    elif _args.mode == 'generation':
        generation_run(_args)
    elif _args.mode == 'estimation':
        estimation_run(_args)
    elif _args.mode == 'estimation_ood':
        estimation_ood_run(_args)
    pass

if __name__ == '__main__':
    initlize_logger()
    main_handler()
    _global_logger.info('Done!')
    print('Done!')
    pass

