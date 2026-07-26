from diffusers import StableDiffusionPipeline
import torch
import sys
import json
from pathlib import Path

input_path = sys.argv[1]
output_path = sys.argv[2]
# Необязательный 3-й аргумент: 'vpred' — отметить v-prediction модель (YiffyMix v4x и т.п.).
# Прогон запишет prediction_type='v_prediction' в scheduler/scheduler_config.json.
vpred = len(sys.argv) > 3 and sys.argv[3].lower() in ('vpred', 'v_prediction', 'v')

print(f'Конвертация {input_path}...')
pipe = StableDiffusionPipeline.from_single_file(input_path, torch_dtype=torch.float16)
pipe.save_pretrained(output_path, safe_serialization=True)

if vpred:
    cfg_file = Path(output_path) / 'scheduler' / 'scheduler_config.json'
    cfg = json.loads(cfg_file.read_text(encoding='utf-8'))
    cfg['prediction_type'] = 'v_prediction'
    cfg_file.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    print('Модель отмечена как v-prediction (prediction_type=v_prediction в scheduler_config.json).')

print(f'Готово! Модель сохранена в {output_path}')
