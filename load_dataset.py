import json
from datasets import load_dataset, concatenate_datasets

# file_names = ['xiaofeng_test_output_dialogue.jsonl', 'baizhantang_test_output_dialogue.jsonl', 'wangduoyu_test_output_dialogue.jsonl', 'guofurong_test_output_dialogue.jsonl', 'weixiaobao_test_output_dialogue.jsonl', 'haruhi_synthesis_dialogue.jsonl', 'murongfu_test_output_dialogue.jsonl', 'McGonagall_test_output_dialogue.jsonl', 'Ron_test_output_dialogue.jsonl', 'Sheldon_test_output_dialogue.jsonl', 'yuqian_test_output_dialogue.jsonl', 'duanyu_test_output_dialogue.jsonl', 'xuzhu_test_output_dialogue.jsonl', 'jiumozhi_test_output_dialogue.jsonl', 'liyunlong_synthesis_dialogue.jsonl', 'Malfoy_test_output_dialogue.jsonl', 'tongxiangyu_test_output_dialogue.jsonl', 'ayaka_test_output_dialogue.jsonl', 'Raj_test_output_dialogue.jsonl', 'Harry_test_output_dialogue.jsonl', 'Snape_test_output_dialogue.jsonl', 'Penny_test_output_dialogue.jsonl', 'zhongli_test_output_dialogue.jsonl', 'tangshiye_test_output_dialogue.jsonl', 'Luna_test_output_dialogue.jsonl', 'hutao_test_output_dialogue.jsonl', 'Dumbledore_test_output_dialogue.jsonl', 'Hermione_test_output_dialogue.jsonl', 'qiaofeng_test_output_dialogue.jsonl', 'wangyuyan_test_output_dialogue.jsonl', 'wanderer_test_output_dialogue.jsonl', 'raidenShogun_test_output_dialogue.jsonl']
def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    json_data = json.loads(line)
                    data.append(json_data)
                except json.JSONDecodeError:
                    # print(f"Failed to parse JSON: {line}")
                    continue
    return data

# 加载两个数据集
train_dataset_dict_A = load_dataset("parquet", data_files='./silk-road/Chat_Suzumiya_Fusion/data/Chat_Suzumiya_Fusion.parquet')
train_dataset_dict_B = load_dataset("parquet", data_files='./silk-road/Chat_Suzumiya_Fusion_B/data/Chat_Suzumiya_Fusion_B.parquet')

# 从DatasetDict中获取具体的数据集
train_dataset_A = train_dataset_dict_A['train']
train_dataset_B = train_dataset_dict_B['train']

train_dataset = concatenate_datasets([train_dataset_A, train_dataset_B])