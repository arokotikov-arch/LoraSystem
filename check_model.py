from safetensors.torch import load_file
import sys

path = sys.argv[1]
sd = load_file(path, device='cpu')

# Проверяем, SD 1.5 или SDXL
if 'model.diffusion_model.input_blocks.0.0.weight' in sd:
    print('SD 1.5')
    print('Input channels:', sd['model.diffusion_model.input_blocks.0.0.weight'].shape[1])
elif 'conditioner.embedders.0.model.transformer.resblocks.0.attn.in_proj_weight' in sd:
    print('SDXL')
else:
    print('Unknown format')
    # Покажем несколько ключей для диагностики
    keys = list(sd.keys())[:10]
    for k in keys:
        print(f'  {k}: {sd[k].shape}')