import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 1. 设置模型路径 (保持和你命令中一致)
MODEL_PATH = "/home/zzy/weitiao/models/Qwen2-VL-7B-Instruct"

# 2. 加载模型
print("正在加载模型...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# 3. 准备输入：图片路径和提示词
image_path = "/home/zzy/weitiao/modules/01_single_gpu_smoke/image.png"
prompt_text = "请描述这张图片的内容。"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {
                "type": "text",
                "text": prompt_text,
            },
        ],
    }
]

# 4. 处理输入
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

# 5. 生成描述
print("正在生成...")
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

print("-" * 30)
print(f"输出结果:\n{output_text[0]}")
print("-" * 30)