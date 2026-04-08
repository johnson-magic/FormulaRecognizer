MAX_LENGTH = 8192
model_id = "Qwen/Qwen2-VL-2B-Instruct"
local_model_path = "./model/Qwen/Qwen2-VL-2B-Instruct"
train_dataset_json_path = "./data/latex_ocr_train.json"
val_dataset_json_path = "./data/latex_ocr_val.json"
output_dir = "./output/Qwen2-VL-2B-LatexOCR"
prompt = "你是一个LaText OCR助手,目标是读取用户输入的照片，转换成LaTex公式。"





