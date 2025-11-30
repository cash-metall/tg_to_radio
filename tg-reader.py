import os
import subprocess
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
import time
import json
import threading
import pyaudio
import numpy as np
import wave
from datetime import datetime
import asyncio
from contextlib import contextmanager
from asyncio import run_coroutine_threadsafe

if os.name == 'nt':
    # Настройка корректной работы asyncio и кодировки консоли в Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def fix_encoding(text: str) -> str:
    """Преобразует строки, некорректно отображающиеся в консоли Windows."""
    if not isinstance(text, str):
        text = str(text)
    try:
        return text.encode('cp1251').decode('utf-8')
    except UnicodeError:
        pass
    try:
        return text.encode('latin1').decode('utf-8')
    except UnicodeError:
        pass
    return text

TOKEN = "TOKEN"

# Настройки
VOLUME = 2.0         # усиление голосовых сообщений
BEEP_VOLUME = 5.0    # громкость писка
BEEP_FREQ = 1000     # частота (Гц)
BEEP_DURATION = 1.5  # длительность (сек)

# Morse настройки
MORSE_MESSAGE = "UB9HEQ"
MORSE_UNIT = 0.1                # длительность точки
MORSE_FREQ = 800                # частота сигнала

# VOX настройки
VOX_THRESHOLD = 500      # порог активации VOX (уровень сигнала)
VOX_SILENCE_TIME = 2.0   # время тишины для остановки записи (сек)
AUDIO_CHUNK = 1024       # размер буфера аудио
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1       # моно
AUDIO_RATE = 44100       # частота дискретизации
INPUT_DEVICE_INDEX = 0  # None = использовать устройство по умолчанию
MIN_RECORDING_DURATION = 5.0  # минимальная длительность записи для отправки (сек)

# Глобальные переменные для VOX и передачи
vox_active = False
recording = False
audio_frames = []
chat_ids = set()  # множество chat_id для отправки сообщений
app_instance = None
transmitting_event = threading.Event()
transmission_lock = threading.Lock()
main_loop = None 

def load_config():
    global TOKEN
    with open('config.json') as f:
        data = json.load(f)
    TOKEN = data['token']

    
load_config()

def play_tone(frequency: float, duration: float, volume: float = BEEP_VOLUME):
    subprocess.run([
        "ffplay", "-nodisp", "-autoexit",
        "-f", "lavfi",
        "-i", f"sine=frequency={frequency}:duration={duration}",
        "-af", f"volume={volume}"
    ], capture_output=True)


def beep(duration):
    play_tone(BEEP_FREQ, duration, BEEP_VOLUME)


def is_radio_receiving() -> bool:
    """Проверяет, идет ли сейчас прием сигнала."""
    return vox_active or recording


def wait_for_reception_end(poll_interval: float = 0.1):
    """Ждет окончания приема, чтобы не мешать передаче."""
    while is_radio_receiving():
        time.sleep(poll_interval)


@contextmanager
def radio_transmission():
    """Гарантирует, что передача начнется только после окончания приема."""
    with transmission_lock:
        wait_for_reception_end()
        transmitting_event.set()
        try:
            yield
        finally:
            transmitting_event.clear()


MORSE_CODE = {
    'A': '.-',    'B': '-...',  'C': '-.-.',  'D': '-..',
    'E': '.',     'F': '..-.',  'G': '--.',   'H': '....',
    'I': '..',    'J': '.---',  'K': '-.-',   'L': '.-..',
    'M': '--',    'N': '-.',    'O': '---',   'P': '.--.',
    'Q': '--.-',  'R': '.-.',   'S': '...',   'T': '-',
    'U': '..-',   'V': '...-',  'W': '.--',   'X': '-..-',
    'Y': '-.--',  'Z': '--..',
    '1': '.----', '2': '..---', '3': '...--', '4': '....-',
    '5': '.....', '6': '-....', '7': '--...', '8': '---..',
    '9': '----.', '0': '-----',
    '(': '-.--.', ')': '-.--.-', '&': '.-...',
    '?': '..--..', '/': '-..-.', '=': '-...-', '+': '.-.-.',
    '-': '-....-', '"': '.-..-.', '@': '.--.-.'
}


def play_morse_message(message: str):
    """Передает сообщение в азбуке Морзе."""
    unit = MORSE_UNIT
    dot = unit
    dash = unit * 3
    intra_char_gap = unit
    letter_gap = unit * 3
    word_gap = unit * 7

    beep(BEEP_DURATION)
    words = message.upper().split(' ')
    for word_index, word in enumerate(words):
        for letter_index, letter in enumerate(word):
            code = MORSE_CODE.get(letter)
            if not code:
                continue
            for symbol_index, symbol in enumerate(code):
                duration = dot if symbol == '.' else dash
                play_tone(MORSE_FREQ, duration, BEEP_VOLUME)
                if symbol_index < len(code) - 1:
                    time.sleep(intra_char_gap)
            if letter_index < len(word) - 1:
                time.sleep(letter_gap)
        if word_index < len(words) - 1:
            time.sleep(word_gap)


async def morse_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /morse - передает callsign азбукой Морзе."""
    await update.message.reply_text(
        f"Передача callsign '{MORSE_MESSAGE}' азбукой Морзе..."
    )
    
    # Запускаем передачу в отдельном потоке, чтобы не блокировать async event loop
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        lambda: play_morse_with_transmission()
    )
    
    await update.message.reply_text("Передача завершена.")


def play_morse_with_transmission():
    """Вспомогательная функция для передачи Морзе с блокировкой передачи."""
    with radio_transmission():
        play_morse_message(MORSE_MESSAGE)


async def start_listen_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start_listen - добавляет chat_id в список для получения сообщений с радиостанции"""
    chat_id = update.message.chat_id
    chat_ids.add(chat_id)
    await update.message.reply_text(
        "Вы подписались на получение сообщений с радиостанции. "
        "Теперь вы будете получать все записи с радиостанции."
    )
    print(f"Chat ID {chat_id} добавлен в список получателей")


async def stop_listen_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /stop_listen - удаляет chat_id из списка"""
    chat_id = update.message.chat_id
    if chat_id in chat_ids:
        chat_ids.remove(chat_id)
        await update.message.reply_text(
            "Вы отписались от получения сообщений с радиостанции."
        )
        print(f"Chat ID {chat_id} удален из списка получателей")
    else:
        await update.message.reply_text(
            "Вы не были подписаны на получение сообщений."
        )


async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await update.message.voice.get_file()
    file_path = "voice.ogg"
    await file.download_to_drive(file_path)

    # Сначала писк и голосовое – только когда нет приема
    with radio_transmission():
        beep(BEEP_DURATION)
        
        # Потом голосовое
        subprocess.run([
            "ffplay", "-nodisp", "-autoexit",
            "-af", f"volume={VOLUME}",
            file_path
        ], capture_output=True)

    os.remove(file_path)


def calculate_rms(data):
    """Вычисляет RMS (Root Mean Square) уровень аудио сигнала"""
    if not data or len(data) == 0:
        return 0.0
    
    audio_data = np.frombuffer(data, dtype=np.int16)
    
    if len(audio_data) == 0:
        return 0.0
    
    # Вычисляем RMS, избегая проблем с NaN и отрицательными значениями
    mean_squared = np.mean(audio_data.astype(np.float64)**2)
    
    # Проверяем на валидность перед sqrt
    if np.isnan(mean_squared) or mean_squared < 0:
        return 0.0
    
    return np.sqrt(mean_squared)


def get_recording_duration_seconds() -> float:
    """Возвращает длительность текущей записи в секундах."""
    if not audio_frames or AUDIO_RATE <= 0:
        return 0.0
    bytes_per_sample = 2  # paInt16
    total_samples = sum(len(frame) // bytes_per_sample for frame in audio_frames)
    return total_samples / AUDIO_RATE


def record_audio():
    """Записывает аудио с микрофона"""
    global recording, audio_frames
    
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_RATE,
            input=True,
            input_device_index=INPUT_DEVICE_INDEX,
            frames_per_buffer=AUDIO_CHUNK
        )
        
        recording = True
        audio_frames = []
        
        print("Начало записи...")
        
        while recording:
            data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            audio_frames.append(data)
        
        stream.stop_stream()
        stream.close()
        print("Запись завершена")
        
    except Exception as e:
        print(f"Ошибка при записи: {e}")
    finally:
        p.terminate()


def save_and_convert_audio():
    """Сохраняет записанное аудио в WAV, затем конвертирует в OGG"""
    if not audio_frames:
        return None
    
    # Сохраняем во временный WAV файл
    wav_filename = f"recorded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    
    try:
        wf = wave.open(wav_filename, 'wb')
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(2) # 16-bit
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(b''.join(audio_frames))
        wf.close()
        
        # Конвертируем в OGG для Telegram
        ogg_filename = wav_filename.replace('.wav', '.ogg')
        subprocess.run([
            "ffmpeg", "-y", "-i", wav_filename,
            "-c:a", "libopus",
            "-b:a", "32k",
            ogg_filename
        ], capture_output=True)
        
        # Удаляем временный WAV файл
        os.remove(wav_filename)
        
        return ogg_filename
    except Exception as e:
        print(f"Ошибка при сохранении аудио: {e}")
        if os.path.exists(wav_filename):
            os.remove(wav_filename)
        return None


async def send_voice_message(ogg_file):
    """Отправляет голосовое сообщение во все сохраненные чаты"""
    if not ogg_file:
        return
    
    try:
        if not os.path.exists(ogg_file):
            return
        
        if not chat_ids:
            print("Нет сохраненных chat_id для отправки сообщения")
            return
        
        for chat_id in chat_ids:
            with open(ogg_file, 'rb') as voice_file:
                await app_instance.bot.send_voice(
                    chat_id=chat_id,
                    voice=voice_file,
                    caption="Сообщение с радиостанции"
                )
        print(f"Сообщение отправлено в {len(chat_ids)} чат(ов)")
    except Exception as e:
        print(f"Ошибка при отправке сообщения: {e}")
    finally:
        # Удаляем временный файл
        if ogg_file and os.path.exists(ogg_file):
            os.remove(ogg_file)


def vox_monitor():
    """Мониторинг VOX - обнаружение сигнала на микрофоне"""
    global vox_active, recording, audio_frames
    
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_RATE,
            input=True,
            input_device_index=INPUT_DEVICE_INDEX,
            frames_per_buffer=AUDIO_CHUNK
        )
        
        silence_start = None
        recording_thread = None
        
        print("VOX мониторинг запущен...")
        
        while True:
            data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            
            if transmitting_event.is_set():
                silence_start = None
                continue
            
            rms = calculate_rms(data)
            
            if rms > VOX_THRESHOLD:
                # Сигнал обнаружен
                if not vox_active:
                    vox_active = True
                    silence_start = None
                    print(f"VOX активирован (уровень: {rms:.0f})")
                    
                    # Запускаем запись в отдельном потоке
                    if not recording:
                        recording_thread = threading.Thread(target=record_audio)
                        recording_thread.daemon = True
                        recording_thread.start()
                
            else:
                # Сигнал отсутствует
                if vox_active:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= VOX_SILENCE_TIME:
                        # Тишина длится достаточно долго - останавливаем запись
                        vox_active = False
                        recording = False
                        print("VOX деактивирован - обработка записи...")
                        
                        # Ждем завершения записи
                        if recording_thread and recording_thread.is_alive():
                            recording_thread.join(timeout=2.0)
                        
                        duration = get_recording_duration_seconds()
                        if duration < MIN_RECORDING_DURATION:
                            print(
                                f"Запись отклонена: длительность {duration:.1f} с "
                                f"< минимальных {MIN_RECORDING_DURATION:.1f} с"
                            )
                            audio_frames.clear()
                            silence_start = None
                            continue

                        # Сохраняем и конвертируем аудио
                        ogg_file = save_and_convert_audio()
                        
                        # Отправляем в Telegram
                        if ogg_file and app_instance and main_loop:
                            try:
                                fut = run_coroutine_threadsafe(send_voice_message(ogg_file), main_loop)
                                fut.result()  # по желанию: дождаться или можно не вызывать
                            except Exception as e:
                                print(f"Ошибка при отправке сообщения: {e}")
                        silence_start = None
                else:
                    silence_start = None
        
    except KeyboardInterrupt:
        print("VOX мониторинг остановлен")
    except Exception as e:
        print(f"Ошибка в VOX мониторинге: {e}")
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        p.terminate()


def list_audio_devices():
    """Выводит список доступных аудио устройств"""
    p = pyaudio.PyAudio()
    print("\nДоступные аудио устройства:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            device_name = fix_encoding(info['name'])
            print(f"  [{i}] {device_name} (входов: {info['maxInputChannels']})")
    p.terminate()


async def telegram_main():
    global app_instance, main_loop
    
    # Показываем список устройств (можно закомментировать после настройки)
    list_audio_devices()
    
    # Создаем приложение Telegram
    app_instance = Application.builder().token(TOKEN).build()
    app_instance.add_handler(CommandHandler("start_listen", start_listen_handler))
    app_instance.add_handler(CommandHandler("stop_listen", stop_listen_handler))
    app_instance.add_handler(CommandHandler("morse", morse_handler))
    app_instance.add_handler(MessageHandler(filters.VOICE, voice_handler))
    
    main_loop = asyncio.get_running_loop()

    # Запускаем VOX мониторинг в отдельном потоке
    vox_thread = threading.Thread(target=vox_monitor, daemon=True)
    vox_thread.start()
    
    print("Бот запущен. Ожидание сообщений и мониторинг VOX...")
    
    # Запускаем Telegram бота
    async with app_instance:
        await app_instance.start()
        await app_instance.updater.start_polling()
        # вечное ожидание (Ctrl+C прервёт)
        await asyncio.Event().wait()

def main():
    asyncio.run(telegram_main())

if __name__ == "__main__":
    main()