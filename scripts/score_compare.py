import json
import numpy as np

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def calculate_mean_and_variance(scores):
    mean = np.mean(scores)
    variance = np.var(scores)
    return mean, variance

def read_scores(json_data):
    ret = []
    for data in json_data:
        ret.append(data['score'])
    return np.array(ret)  
    
def main(file1, file2):
    data1 = read_json(file1)
    data2 = read_json(file2)
    
    scores1 = read_scores(data1)
    scores2 = read_scores(data2)
    
    mean1, variance1 = calculate_mean_and_variance(scores1)
    mean2, variance2 = calculate_mean_and_variance(scores2)
    
    print(f"File 1 - Mean: {mean1}, Variance: {variance1}")
    print(f"File 2 - Mean: {mean2}, Variance: {variance2}")

if __name__ == "__main__":
    file1 = '../output/rm_score/eval_data_with_score.json'
    file2 = '../output/rm_score_qwen/eval_data_with_score.json'
    main(file1, file2)