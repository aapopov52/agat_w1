import os
import moviepy.editor as mp
from faster_whisper import WhisperModel
import nltk

# для чистки звука
from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import AudioFile, convert_audio
from pathlib import Path
import torch
import torchaudio


# МОДЕЛЬ РАСПОЗНАВАНИЯ
nltk.download('punkt')
nltk.download('stopwords')
whisper_model = WhisperModel("large-v2")

# МОДЕЛЬ УЛУЧШЕНИЯ ЗВУКА
# Устройство для вычислений: GPU или CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Загрузка предобученной модели Demucs (например, htdemucs)
demucs_model_name = "htdemucs"
demucs_model = pretrained.get_model(demucs_model_name).to(device)
demucs_model.eval()
    

def process_video(t_audio_clear, t_video, t_audio):
    # Распознаём аудио-файл
    if t_audio != "" and not (t_audio is None):
        return get_text_from_video_audio(t_audio_clear, t_audio)
    # Распознаём видео-файл
    if t_video != "" and not (t_video is None):
        return get_text_from_video_audio(t_audio_clear, t_video)
    
    return "Укажите параметры работы с видео материалом."

# Работа с аудио/видео на локальном диске
def get_text_from_video_audio(t_audio_clear, fileName):
    folder_work = get_folder_result()
    audio_file = mp.AudioFileClip(fileName)
    fileName_input = folder_work + '/vrem.wav'
    fileName_postobr = fileName_input
    audio_file.write_audiofile(fileName_input)
    if t_audio_clear == 'Да':
        fileName_postobr = clear_audio(fileName_input)
    segments, info = whisper_model.transcribe(fileName_postobr, 
        task='transcribe',  
        language="ru",
        beam_size=5,        # Настройка декодера (выражает количество гипотез для поиска)
        best_of=5           # Оставляем лучшую гипотезу из 5 вариантов)
       )
    sText = ''
    for segment in segments:
        sText += segment.text
    
    os.remove(fileName)
    os.remove(fileName_input)
    if fileName_postobr != fileName_input:
        os.remove(fileName_postobr)
    os.rmdir(folder_work)
    
    return sText

# подавление шумов
def clear_audio (input_file):
    
    # Загрузка и конвертация аудиофайла
    wav = None
    #try:
    f = AudioFile(input_file)
    wav = f.read(streams=0)
    wav = convert_audio(wav, demucs_model.samplerate, demucs_model.samplerate, demucs_model.audio_channels)
    
    # Применение модели для разделения звуковых источников
    with torch.no_grad():
        sources = apply_model(demucs_model, wav[None], device=device)
        sources = sources[0]  # Убираем измерение batch
    
    # Сохранение всех дорожек (вокал, бас, ударные, и т.д.)
    # Нам интересен только вокал
    for source, name in zip(sources, demucs_model.sources):
        if name == 'vocals':
            output_file = os.path.dirname(input_file) + '/' + f"{Path(input_file).stem}_{name}.wav"
            torchaudio.save(str(output_file), source.cpu(), demucs_model.samplerate)
            return output_file
            #print(f"Сохранено: {output_file}")

# gradio может быть одновремено запущен несколько раз
# для обеспечения уникальности будем сохранять результаты в отдельных каталогах (просто под номерами)
def get_folder_result():
    sf = 'tmp'
    if not os.path.exists(sf):
        os.mkdir(sf)
        
    i = 1
    while os.path.exists(sf + '/' + str(i)):
        i += 1
    sf = sf + '/' + str(i)
    os.mkdir(sf)
    
    return sf
    
    
    