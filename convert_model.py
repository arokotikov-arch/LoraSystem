from diffusers import StableDiffusionPipeline
import torch
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

print(f'Конвертация {input_path}...')
pipe = StableDiffusionPipeline.from_single_file(input_path, torch_dtype=torch.float16)
pipe.save_pretrained(output_path, safe_serialization=True)
print(f'Готово! Модель сохранена в {output_path}')