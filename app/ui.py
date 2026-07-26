import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import gradio as gr
import trainer as t

with gr.Blocks(title='SD 1.5 LoRA Trainer') as app:
    gr.Markdown('# SD 1.5 LoRA Trainer\nЛокальное обучение LoRA персонажа • RTX 4060 8 GB preset')
    with gr.Tab('1. Датасет'):
        uploads=gr.File(label='Добавьте фото (JPG / PNG / WEBP)',file_count='multiple',file_types=['image'])
        with gr.Row(): import_btn=gr.Button('Импортировать фото',variant='primary'); clear_btn=gr.Button('Очистить датасет',variant='stop')
        dataset_status=gr.Textbox(value=t.status(),label='Статус',lines=2)
        import_btn.click(t.import_files,uploads,dataset_status)
        clear_btn.click(t.clear_dataset,None,dataset_status)
        gr.Markdown('**Быстрые подписи.** Затем откройте таблицу ниже и уточните каждое описание: поза, кадрирование, одежда, освещение. Trigger-word должен быть во всех строках.')
        with gr.Row(): trigger=gr.Textbox(value='skswoman',label='Trigger-word (редкий токен)'); generic=gr.Textbox(value='woman, portrait photo',label='Общий текст для всех фото')
        cap_btn=gr.Button('Применить общую подпись ко всем'); cap_result=gr.Textbox(label='Результат')
        cap_btn.click(t.write_captions,[trigger,generic],cap_result)
        table=gr.Dataframe(headers=['Файл','Подпись'],value=t.caption_table(),datatype=['str','str'],interactive=True,label='Индивидуальные подписи')
        with gr.Row(): refresh=gr.Button('Обновить таблицу'); save=gr.Button('Сохранить подписи')
        refresh.click(t.caption_table,None,table); save.click(t.save_table,table,cap_result)
    with gr.Tab('2. Обучение'):
        s=t.load_config()  # автозаполнение последними использованными значениями
        gr.Markdown('Базовая модель — **SD 1.5 в формате Diffusers** (папка с `model_index.json`). Настройки рассчитаны на 8 GB VRAM. Значения запоминаются между запусками.')
        model=gr.Textbox(value=s.get('model_path',r'C:\Users\Aro\Desktop\Data\Models\Diffusers\yiffymix_v44'),label='Путь к базовой модели')
        with gr.Row(): name=gr.Textbox(value=s.get('output_name','my_person_sd15'),label='Имя LoRA'); resolution=gr.Dropdown([512,768],value=s.get('resolution',512),label='Разрешение')
        with gr.Row(): batch=gr.Number(value=s.get('batch',1),precision=0,label='Batch'); accum=gr.Number(value=s.get('accum',4),precision=0,label='Gradient accumulation'); rank=gr.Dropdown([8,16,32,64,128],value=s.get('rank',16),label='LoRA rank')
        with gr.Row(): lr=gr.Number(value=s.get('lr',0.0001),label='Learning rate'); steps=gr.Number(value=s.get('steps',1400),precision=0,label='Шагов'); seed=gr.Number(value=s.get('seed',42),precision=0,label='Seed')
        vpred=gr.Checkbox(value=bool(s.get('vpred',True)),label='v-prediction модель (YiffyMix v4x и т.п.)')
        with gr.Accordion('Расширенные настройки — тонкая подстройка', open=False):
            gr.Markdown('Если лора «недоучилась» / результат совпадает с базой — попробуйте комбинацию:\n• **Target modules** → `extended` (главное)\n• **Min-SNR gamma** → `5.0`\n• **LR scheduler** → `cosine` или `constant_with_warmup`\n• **Rank** → `32–64`, **Learning rate** → `2e-4 – 3e-4`\n• Больше разных и одинаковых по стилю фото (15–30).\n768px на 8 GB держится только с batch 1 + rank ≤ 32.')
            with gr.Row(): lr_scheduler=gr.Dropdown(['constant','constant_with_warmup','cosine','cosine_with_restarts','linear'],value=s.get('lr_scheduler','constant'),label='LR scheduler'); lr_warmup=gr.Number(value=s.get('lr_warmup',0),precision=0,label='LR warmup steps (0 = выкл)'); max_grad_norm=gr.Number(value=s.get('max_grad_norm',1.0),label='Max grad norm')
            with gr.Row(): optimizer=gr.Dropdown(['adamw','adamw8bit'],value=s.get('optimizer','adamw'),label='Оптимизатор (8-bit ↓ VRAM)'); snr_gamma=gr.Number(value=s.get('snr_gamma',0.0),label='Min-SNR gamma (0 = выкл, 5 = v-pred)'); noise_offset=gr.Number(value=s.get('noise_offset',0.0),label='Noise offset (0 = выкл, 0.05–0.1)')
            target_modules=gr.Dropdown(['standard','extended'],value=s.get('target_modules','standard'),label='LoRA target modules (standard = дефолт Diffusers, extended = Kohya-style, сильнее)')
            with gr.Row(): tf32=gr.Checkbox(value=bool(s.get('tf32',False)),label='TF32 (ускорение на RTX 30/40)'); xformers=gr.Checkbox(value=bool(s.get('xformers',False)),label='xFormers (↓ VRAM)'); random_flip=gr.Checkbox(value=bool(s.get('random_flip',False)),label='Random flip (аугментация)'); center_crop=gr.Checkbox(value=bool(s.get('center_crop',False)),label='Center crop (иначе random crop)')
        with gr.Row(): start=gr.Button('Начать обучение',variant='primary'); save_btn=gr.Button('Сохранить настройки'); stop=gr.Button('Остановить')
        action=gr.Textbox(label='Состояние',lines=2)
        _pack=lambda m,n,r,b,a,ra,l,st,se,v,ls,lw,mgn,opt,snr,no,tm,t32,xf,rf,cc: {'model_path':m,'output_name':n,'resolution':r,'batch':b,'accum':a,'rank':ra,'lr':l,'steps':st,'seed':se,'vpred':v,'lr_scheduler':ls,'lr_warmup':lw,'max_grad_norm':mgn,'optimizer':opt,'snr_gamma':snr,'noise_offset':no,'target_modules':tm,'tf32':t32,'xformers':xf,'random_flip':rf,'center_crop':cc}
        _inputs=[model,name,resolution,batch,accum,rank,lr,steps,seed,vpred,lr_scheduler,lr_warmup,max_grad_norm,optimizer,snr_gamma,noise_offset,target_modules,tf32,xformers,random_flip,center_crop]
        start.click(lambda *x: t.start(_pack(*x)),_inputs,action)
        save_btn.click(lambda *x: t.save_config(_pack(*x)),_inputs,action)
        stop.click(t.stop,None,action)
    with gr.Tab('3. Логи'):
        log=gr.Textbox(value=t.logs(),label='Последний лог',lines=25); update=gr.Button('Обновить лог'); update.click(t.logs,None,log)
        gr.Markdown('Готовый `.safetensors` появится в `workspace/output/<имя>`. Для prompt используйте ваш trigger-word.')
if __name__ == '__main__':
    app.launch(server_name='127.0.0.1',server_port=7860,inbrowser=True)
