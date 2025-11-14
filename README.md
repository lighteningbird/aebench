# LLM Capability Estimation
1. lanunch from src/estimation/run.py
2. refer to params inside src/estimation/utils UtilsTask

## Envrionment:
Python3.8.19 with torch2.3.1, scikit-learn1.3.2, xgboost2.1.0, scipy1.10.1, transformers4.42.4

## Example:
### Step 1: Generation Output
python -u src/estimation/run.py --mode=debug --task_type=classification --dataset=imdb --model_name=llama3-8b --max_new_tokens=256 --prompt_type=zero_shot
### Step 2: Estimation
python -u src/estimation/run.py --mode=debug --debug_mode=estimation --task_type=classification --dataset=imdb --model_name=llama3-8b --max_new_tokens=256 --prompt_type=zero_shot


## Additional results
Results on DeepSeek models uncertainty based methods perform the best.

| | Cosmos	| Wiki |
| --- | --- | ---| 
| ATC	| 0.338 |	0.0522 |
| UC-Entropy-LR | 	0.0288	| 0.0337 |
