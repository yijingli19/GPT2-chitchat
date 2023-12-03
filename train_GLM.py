from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, Dataset
import torch
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
from load_dataset import train_dataset


tokenizer = AutoTokenizer.from_pretrained("./chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("./chatglm2-6b", trust_remote_code=True).half().cuda()

dataset = load_dataset("parquet", data_files='./silk-road/Chat_Suzumiya_Fusion/data/Chat_Suzumiya_Fusion.parquet')
print(dataset)

def preprocess_dialogue(example):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt,truncation=True,add_special_tokens=True)
    target_ids = tokenizer.encode(target,truncation=True,add_special_tokens=False)
    input_ids = prompt_ids + target_ids
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}

model_inputs = train_dataset.map(preprocess_dialogue)

for param in model.parameters():
  param.requires_grad = False  # 冻结模型参数，不需要梯度计算
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()  # 计算输入梯度
model.is_parallelizable = True  # 可并行化
model.model_parallel = True

config = LoraConfig(
    r=16,   # LoRA低秩矩阵的维数。关于秩的选择，通常，使用4，8，16
    lora_alpha=32,  # LoRA低秩矩阵的缩放系数，为一个常数超参，调整alpha与调整学习率类似
    inference_mode=False,   # 是否在推理模式下使用Peft模型
    lora_dropout=0.05,  # dropout
    #bias="none",
    task_type="CAUSAL_LM"   # 指定任务类型,因果语言建模
)

model = get_peft_model(model, config)

def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        # 构造mask 不需要预测和padding部分变为-100
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }

class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

training_args = TrainingArguments(
    num_train_epochs = 2,
    max_steps = -1,
    evaluation_strategy = "no", # 验证方式，no/steps/epoch
    gradient_accumulation_steps = 1,    # 每个batch更新一次梯度
    group_by_length=False,    # 对样本不进行分组
    save_strategy = "steps",
    save_steps = 500,    # 每500步保存一次模型
    output_dir = 'output',
    remove_unused_columns = False,    # 不删除未使用的列
    per_device_train_batch_size = 32,
    per_device_eval_batch_size = 32,
    learning_rate = 1e-4,
    fp16 = True,
    seed=2023,
    data_seed=2023
)

trainer = ModifiedTrainer(
    model=model,
    train_dataset=model_inputs['train'],
    #eval_dataset=model_inputs['test'],
    args=training_args,
    data_collator=data_collator,
)
trainer.train()
