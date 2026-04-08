import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer


from peft import get_peft_model, PeftModel
from transformers import (
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
import swanlab
import json
import os
from functools import partial

# mine module
from utils import process_func, predict
from config import model_id, local_model_path, train_dataset_json_path, val_dataset_json_path, output_dir, prompt  # config single variable
from config import lora_train_config, lora_val_config, training_args, create_swanlab_callback  # config structure variable


# 1. download model
model_dir = snapshot_download(model_id, cache_dir="./model", revision="master")


# 2. model setup: 2.1 model; 2.2 model + peft; 2.3 tokenizer; 2.4 processor
# 2.1
origin_model = Qwen2VLForConditionalGeneration.from_pretrained(local_model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,)
origin_model.enable_input_require_grads()
# 2.2
train_peft_model = get_peft_model(origin_model, lora_train_config)
# 2.3
tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=False, trust_remote_code=True)
# 2.4
processor = AutoProcessor.from_pretrained(local_model_path)


# 3. data preprocess
train_ds = Dataset.from_json(train_dataset_json_path)
process_func_partial = partial(process_func, processor=processor, tokenizer=tokenizer)
train_dataset = train_ds.map(process_func_partial)


# optional swanlab setup
swanlab_callback = create_swanlab_callback({"train_data_number" : len(train_ds)})
  

# 4. Trainer setup
trainer = Trainer(
    model=train_peft_model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)


# 5. star train
trainer.train()


# 6. val setup
load_model_path = f"{output_dir}/checkpoint-{max([int(d.split('-')[-1]) for d in os.listdir(output_dir) if d.startswith('checkpoint-')])}"
print(f"load_model_path: {load_model_path}")
val_peft_model = PeftModel.from_pretrained(origin_model, model_id=load_model_path, config=lora_val_config)


# 7. load val dataset
with open(val_dataset_json_path, "r") as f:
    test_dataset = json.load(f)


test_image_list = []
for item in test_dataset:    
    image_file_path = item["conversations"][0]["value"]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join("./data/LaTeX_OCR", os.path.basename(image_file_path))
    label = item["conversations"][1]["value"]
    
    messages = [{
        "role": "user", 
        "content": [
            {
            "type": "image", 
            "image": relative_path,
            "resized_height": 100,
            "resized_width": 500,   
            },
            {
            "type": "text",
            "text": prompt,
            }
        ]}]
    
    response = predict(messages, val_peft_model, processor=processor)
    
    print(f"predict:{response}")
    print(f"gt:{label}\n")

    test_image_list.append(swanlab.Image(relative_path, caption=response))

swanlab.log({"Prediction": test_image_list})

swanlab.finish()