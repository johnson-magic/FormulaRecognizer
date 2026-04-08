from swanlab.integration.transformers import SwanLabCallback

from .config import model_id, train_dataset_json_path, val_dataset_json_path, output_dir, prompt, MAX_LENGTH


def create_swanlab_callback(config_overrides=None):
    """创建 SwanLab 回调，支持运行时覆盖配置"""
    base_config = {
        "project": "Qwen2-VL-ft-latexocr",
        "experiment_name": "7B-1kdata",
        "config": {
            "model": "https://modelscope.cn/models/Qwen/Qwen2-VL-7B-Instruct",
            "dataset": "https://modelscope.cn/datasets/AI-ModelScope/LaTeX_OCR/summary",
            "model_id": model_id,
            "train_dataset_json_path": train_dataset_json_path,
            "val_dataset_json_path": val_dataset_json_path,
            "output_dir": output_dir,
            "prompt": prompt,
            "train_data_number": None,  # 运行时设置
            "token_max_length": MAX_LENGTH,
            "lora_rank": 64,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
        },
    }
    
    # 运行时覆盖
    if config_overrides:
        base_config["config"].update(config_overrides)
    
    return SwanLabCallback(**base_config)