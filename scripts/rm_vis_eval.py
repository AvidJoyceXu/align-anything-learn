import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from align_anything.models.pretrained_model_with_value import load_pretrained_model_with_value_head

import argparse
import os
from align_anything.utils.tools import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    prepare_ds_train_cfgs,
    read_cfgs,
    seed_everything,
    split_prompt_response,
    update_dict,
)

def load_dataset(file_path):
    """加载偏好数据集"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def get_reward_score(model, tokenizer, prompt, response, device):
    """使用奖励模型对单个问答对进行评分"""
    # 构建输入
    inputs = tokenizer(
        prompt + response,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)
    
    # 获取模型预测
    with torch.no_grad():
        output = model(**inputs)
        
        scores = output.scores
        # end_scores = output.end_scores
        # higher_rewards, lower_rewards = scores.squeeze(dim=-1).chunk(chunks=2, dim=0)
        # higher_end_reward, lower_end_reward = end_scores.squeeze(dim=-1).chunk(chunks=2, dim=0)
    
    return scores

def init_models(cfgs) :
    """Initialize model and tokenizer.
    return (model, tokenizer, processor)
    """
    return load_pretrained_model_with_value_head(
        "../output/rm/slice_end",
        model_max_length=cfgs.model_cfgs.model_max_length,
        padding_side='right',
        trust_remote_code=cfgs.train_cfgs.trust_remote_code,
        modality='text',
    )

def analyze_preference_dataset(cfgs, dataset_path, model_path, save_dir):
    """分析偏好数据集并生成可视化"""
    # 加载模型和tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, processor = init_models(cfgs)
    model.to(device)
    
    # 加载数据集
    dataset = load_dataset(dataset_path)
    
    # 存储评分
    chosen_scores = []
    rejected_scores = []
    
    # 对每个样本进行评分
    for item in tqdm(dataset):
        prompt = item['prompt']
        
        # 获取chosen response的评分
        chosen_score = get_reward_score(
            model, 
            tokenizer, 
            prompt, 
            item['response_0'] if item['better_response_id'] == 0 else item['response_1'],
            device
        )
        # chosen_scores.append(chosen_score)
        print(torch.min(chosen_score))
        
        # 获取rejected response的评分
        rejected_score = get_reward_score(
            model, 
            tokenizer, 
            prompt, 
            item['response_1'] if item['better_response_id'] == 0 else item['response_0'],
            device
        )
        # rejected_scores.append(rejected_score)
        print(torch.min(rejected_score))
        break
    print(chosen_score.shape)
    print(rejected_score.shape)
    # # 绘制分布图
    # plt.figure(figsize=(10, 6))
    # sns.kdeplot(data=chosen_scores, label='Chosen Responses', color='green')
    # sns.kdeplot(data=rejected_scores, label='Rejected Responses', color='red')
    # plt.title('Distribution of Reward Scores')
    # plt.xlabel('Reward Score')
    # plt.ylabel('Density')
    # plt.legend()
    # plt.savefig(f'{save_dir}/reward_distribution.png')
    # plt.close()
    
    # # 绘制箱线图
    # plt.figure(figsize=(8, 6))
    # plt.boxplot([chosen_scores, rejected_scores], labels=['Chosen', 'Rejected'])
    # plt.title('Reward Score Distribution (Box Plot)')
    # plt.ylabel('Reward Score')
    # plt.savefig(f'{save_dir}/reward_boxplot.png')
    # plt.close()
    

    # 计算统计信息
    stats = {
        'chosen_mean': torch.mean(chosen_scores),
        'chosen_std': np.std(chosen_scores),
        'rejected_mean': np.mean(rejected_scores),
        'rejected_std': np.std(rejected_scores),
        'score_difference_mean': np.mean(np.array(chosen_scores) - np.array(rejected_scores))
    }
    
    # # 保存统计信息
    # with open(f'{save_dir}/stats.json', 'w') as f:
    #     json.dump(stats, f, indent=4)
    
    return stats

if __name__ == "__main__":
    # 设置路径
    dataset_path = "../../datasets/PKU-SafeRLHF-single-dimension/data/Alpaca-7B/test.jsonl"
    model_path = "../output/rm/slice_end"
    save_dir = "../output/rm_vis"
    
    import os
    # read default configs from the yaml file
    task = os.path.join('text_to_text', 'rm')
    dict_cfgs, ds_cfgs = read_cfgs(mode='train', task=task)

    # get custom configs from command line
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[1::2]]
    values = list(unparsed_args[2::2])
    unparsed_args = dict(zip(keys, values))
    for k, v in unparsed_args.items():
        dict_cfgs = update_dict(dict_cfgs, custom_cfgs_to_dict(k, v))

    # setup training
    cfgs = dict_to_namedtuple(dict_cfgs)

    # 运行分析
    stats = analyze_preference_dataset(cfgs=cfgs, dataset_path=dataset_path, model_path=model_path, save_dir=save_dir)
    print("Analysis completed. Statistics:", stats)