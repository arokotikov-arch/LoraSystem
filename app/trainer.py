from __future__ import annotations
import json, os, shutil, subprocess, sys, threading
from pathlib import Path
from datetime import datetime
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'workspace'
IMAGES, CAPS, OUT, LOGS = (WORK/'images', WORK/'captions', WORK/'output', WORK/'logs')
for p in (IMAGES, CAPS, OUT, LOGS): p.mkdir(parents=True, exist_ok=True)
_process: subprocess.Popen | None = None
_lock = threading.Lock()

def image_files(): return sorted([p for p in IMAGES.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}])

def import_files(files):
    if not files: return status()
    added=[]; skipped=0
    existing_stems={p.stem.rsplit('_',1)[0] for p in IMAGES.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}}
    for source in files:
        source=Path(source)
        if source.suffix.lower() not in {'.jpg','.jpeg','.png','.webp'}: continue
        stem=''.join(c if c.isalnum() or c in '-_' else '_' for c in source.stem)[:55]
        if stem in existing_stems:
            skipped+=1
            continue
        dest=IMAGES/f'{stem}_{datetime.now():%H%M%S%f}{source.suffix.lower()}'
        with Image.open(source) as im:
            im = im.convert('RGB'); im.save(dest, quality=95)
        (CAPS/f'{dest.stem}.txt').write_text('', encoding='utf-8')
        added.append(dest.name)
        existing_stems.add(stem)
    msg = status()
    if added: msg += '\nДобавлено: ' + ', '.join(added[:4])
    if skipped: msg += f'\nПропущено дубликатов: {skipped}'
    return msg

def clear_dataset():
    for p in IMAGES.iterdir():
        if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}:
            p.unlink()
    for p in CAPS.iterdir():
        if p.suffix.lower() == '.txt':
            p.unlink()
    if (IMAGES/'metadata.jsonl').exists():
        (IMAGES/'metadata.jsonl').unlink()
    return status() + '\nДатасет очищен.'

def write_captions(prefix: str, extra: str):
    prefix=prefix.strip().strip(',')
    if not prefix: raise ValueError('Введите trigger-word.')
    for im in image_files():
        cap = f'{prefix}, {extra.strip()}'.strip().strip(',')
        (CAPS/f'{im.stem}.txt').write_text(cap, encoding='utf-8')
    return f'Подписи обновлены: {len(image_files())}.'

def status():
    imgs=image_files(); captions=sum(1 for p in imgs if (CAPS/f'{p.stem}.txt').read_text(encoding='utf-8').strip())
    return f'Изображений: {len(imgs)} | с непустой подписью: {captions} | папка: {IMAGES}'

def caption_table():
    return [[p.name, (CAPS/f'{p.stem}.txt').read_text(encoding='utf-8')] for p in image_files()]

def save_table(data):
    existing={p.name:p for p in image_files()}
    import pandas as pd
    if isinstance(data, pd.DataFrame):
        rows = data.to_dict('records')
    elif isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get('data', [])
    else:
        rows = []
    for row in rows:
        if isinstance(row, dict):
            fname = row.get('Файл') or row.get('file_name') or row.get('0', '')
            caption = row.get('Подпись') or row.get('text') or row.get('1', '')
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            fname, caption = row[0], row[1]
        else:
            continue
        if fname in existing:
            (CAPS/f'{existing[fname].stem}.txt').write_text(str(caption or '').strip(), encoding='utf-8')
    return 'Сохранено. ' + status()

def _metadata():
    records=[]
    for p in image_files():
        text=(CAPS/f'{p.stem}.txt').read_text(encoding='utf-8').strip()
        if not text: raise ValueError(f'Нет подписи: {p.name}')
        records.append({'file_name': p.name, 'text': text})
    if len(records)<5: raise ValueError('Для обучения добавьте минимум 5 изображений (рекомендуется 15–30).')
    (IMAGES/'metadata.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in records),encoding='utf-8')

def start(cfg):
    global _process
    with _lock:
        if _process and _process.poll() is None: return 'Обучение уже запущено.'
        model=Path(cfg['model_path']).expanduser()
        if not (model/'model_index.json').exists(): return 'Ошибка: путь к модели должен вести к Diffusers-папке с model_index.json.'
        try: _metadata()
        except ValueError as e: return 'Ошибка: '+str(e)
        script=ROOT/'vendor/diffusers/examples/text_to_image/train_text_to_image_lora.py'
        if not script.exists(): return 'Ошибка: запустите scripts/setup_windows.ps1 — скрипт Diffusers не найден.'
        name=''.join(c if c.isalnum() or c in '-_' else '_' for c in cfg['output_name']) or 'lora'
        runout=OUT/name; runout.mkdir(parents=True,exist_ok=True)
        log=LOGS/f'{name}_{datetime.now():%Y%m%d_%H%M%S}.log'
        cmd=[sys.executable, str(script), '--pretrained_model_name_or_path',str(model), '--train_data_dir',str(IMAGES), '--caption_column','text', '--resolution',str(int(cfg['resolution'])), '--train_batch_size',str(int(cfg['batch'])), '--gradient_accumulation_steps',str(int(cfg['accum'])), '--learning_rate',str(float(cfg['lr'])), '--max_train_steps',str(int(cfg['steps'])), '--rank',str(int(cfg['rank'])), '--output_dir',str(runout), '--mixed_precision','fp16', '--gradient_checkpointing', '--seed',str(int(cfg['seed'])), '--checkpointing_steps','500', '--checkpoints_total_limit','2']
        env=os.environ.copy()
        fh=open(log,'w',encoding='utf-8',buffering=1)
        _process=subprocess.Popen(cmd,stdout=fh,stderr=subprocess.STDOUT,cwd=str(ROOT),env=env)
        return f'Запущено (PID {_process.pid}). Лог: {log.name}'

def stop():
    global _process
    with _lock:
        if _process and _process.poll() is None:
            _process.terminate(); return 'Отправлен сигнал остановки. Подождите несколько секунд.'
    return 'Активного обучения нет.'

def logs():
    files=sorted(LOGS.glob('*.log'),key=lambda p:p.stat().st_mtime,reverse=True)
    state='не запущено' if not _process or _process.poll() is not None else f'работает, PID {_process.pid}'
    if not files:return f'Статус: {state}\nЛогов пока нет.'
    text=files[0].read_text(encoding='utf-8',errors='replace')[-12000:]
    return f'Статус: {state}\nЛог: {files[0].name}\n\n{text}'