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
import logging

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
VOX_THRESHOLD_ON = 500      # порог активации VOX (уровень сигнала)
VOX_THRESHOLD_OFF = 150      # порог деактивации VOX (уровень сигнала)
VOX_SILENCE_TIME = 2.0   # время тишины для остановки записи (сек)
AUDIO_CHUNK = 1024       # размер буфера аудио
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1       # моно
AUDIO_RATE = 44100       # частота дискретизации
INPUT_DEVICE_INDEX = 0  # None = использовать устройство по умолчанию
MIN_RECORDING_DURATION = 5.0  # минимальная длительность записи для отправки (сек)

# Глобальные переменные для VOX и передачи
vox_active = False
audio_frames = []
chat_ids = set()  # множество chat_id для отправки сообщений
app_instance = None
transmitting_event = threading.Event()
transmission_lock = threading.Lock()
main_loop = None 
audio_lock = threading.Lock()

def load_config():
    """Load configuration values from `config.json`, falling back to defaults."""
    global TOKEN, VOLUME, BEEP_VOLUME, BEEP_FREQ, BEEP_DURATION
    global MORSE_MESSAGE, MORSE_UNIT, MORSE_FREQ
    global VOX_THRESHOLD_ON, VOX_THRESHOLD_OFF, VOX_SILENCE_TIME
    global AUDIO_CHUNK, AUDIO_CHANNELS, AUDIO_RATE, INPUT_DEVICE_INDEX
    global MIN_RECORDING_DURATION

    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print('config.json not found; using built-in defaults')
        return
    except Exception as e:
        print(f'Failed to load config.json: {e}; using defaults')
        return

    TOKEN = data.get('token', TOKEN)
    VOLUME = float(data.get('volume', VOLUME))
    BEEP_VOLUME = float(data.get('beep_volume', BEEP_VOLUME))
    BEEP_FREQ = int(data.get('beep_freq', BEEP_FREQ))
    BEEP_DURATION = float(data.get('beep_duration', BEEP_DURATION))

    MORSE_MESSAGE = data.get('morse_message', MORSE_MESSAGE)
    MORSE_UNIT = float(data.get('morse_unit', MORSE_UNIT))
    MORSE_FREQ = int(data.get('morse_freq', MORSE_FREQ))

    VOX_THRESHOLD_ON = float(data.get('vox_threshold_on', VOX_THRESHOLD_ON))
    VOX_THRESHOLD_OFF = float(data.get('vox_threshold_off', VOX_THRESHOLD_OFF))
    VOX_SILENCE_TIME = float(data.get('vox_silence_time', VOX_SILENCE_TIME))

    AUDIO_CHUNK = int(data.get('audio_chunk', AUDIO_CHUNK))
    AUDIO_CHANNELS = int(data.get('audio_channels', AUDIO_CHANNELS))
    AUDIO_RATE = int(data.get('audio_rate', AUDIO_RATE))

    # Allow explicit null in config to mean default device
    if 'input_device_index' in data:
        INPUT_DEVICE_INDEX = data['input_device_index']

    MIN_RECORDING_DURATION = float(data.get('min_recording_duration', MIN_RECORDING_DURATION))


load_config()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

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
    # Read shared state under lock to avoid race.
    with audio_lock:
        return bool(vox_active)


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
        "Вы подписались на получение сообщений с радиостанции. Теперь вы будете получать все записи с радиостанции."
    )
    logging.info("Chat ID %s добавлен в список получателей", chat_id)


async def stop_listen_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /stop_listen - удаляет chat_id из списка"""
    chat_id = update.message.chat_id
    if chat_id in chat_ids:
        chat_ids.remove(chat_id)
        await update.message.reply_text("Вы отписались от получения сообщений с радиостанции.")
        logging.info("Chat ID %s удален из списка получателей", chat_id)
    else:
        await update.message.reply_text("Вы не были подписаны на получение сообщений.")


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
    logging.info("Воспроизведено входящее голосовое сообщение и временный файл удалён")


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


def get_recording_duration_seconds(frames=None) -> float:
    """Возвращает длительность записи в секундах для переданных фреймов.

    Если `frames` не переданы, используется глобальный `audio_frames`.
    """
    if frames is None:
        frames = audio_frames
    if not frames or AUDIO_RATE <= 0:
        return 0.0
    bytes_per_sample = 2  # paInt16
    total_samples = sum(len(frame) // bytes_per_sample for frame in frames)
    return total_samples / AUDIO_RATE


# Recording is now handled inside `vox_monitor` to avoid multiple threads
# opening the audio device concurrently and to centralize frame capture.


def save_and_convert_audio(frames):
    """Сохраняет переданные фреймы в WAV, затем конвертирует в OGG.

    Эта функция не изменяет глобальный `audio_frames` — вызывающий код
    должен передать копию массива фреймов, полученную под блокировкой.
    """
    if not frames:
        return None

    wav_filename = f"recorded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    try:
        wf = wave.open(wav_filename, 'wb')
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(2) # 16-bit
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        ogg_filename = wav_filename.replace('.wav', '.ogg')
        subprocess.run([
            "ffmpeg", "-y", "-i", wav_filename,
            "-c:a", "libopus",
            "-b:a", "32k",
            ogg_filename
        ], capture_output=True)

        os.remove(wav_filename)
        logging.info("Converted and removed temporary wav: %s", wav_filename)
        return ogg_filename
    except Exception as e:
        logging.exception("Ошибка при сохранении аудио: %s", e)
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
            logging.info("Нет сохраненных chat_id для отправки сообщения")
            return
        for chat_id in chat_ids:
            with open(ogg_file, 'rb') as voice_file:
                await app_instance.bot.send_voice(chat_id=chat_id, voice=voice_file, caption="Сообщение с радиостанции")
        logging.info("Сообщение отправлено в %d чат(ов)", len(chat_ids))
    except Exception as e:
        logging.exception("Ошибка при отправке сообщения: %s", e)
    finally:
        if ogg_file and os.path.exists(ogg_file):
            os.remove(ogg_file)
            logging.info("Временный ogg-файл удалён: %s", ogg_file)


def vox_monitor():
    """Мониторинг VOX - обнаружение сигнала на микрофоне"""
    global vox_active, recording, audio_frames
    
    p = pyaudio.PyAudio()
    stream = None
    
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
        logging.info("VOX мониторинг запущен")

        while True:
            data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)

            if transmitting_event.is_set():
                silence_start = None
                continue

            rms = calculate_rms(data)

            if rms > VOX_THRESHOLD_ON:
                # Сигнал обнаружен
                silence_start = None
                if not vox_active:
                    vox_active = True
                    logging.info("VOX активирован (уровень: %.0f)", rms)

                # Собираем фреймы под блокировкой
                with audio_lock:
                    audio_frames.append(data)

            elif vox_active and rms > VOX_THRESHOLD_OFF:
                # Низкий, но всё ещё выше порога отключения — продолжаем сбор
                silence_start = None
                with audio_lock:
                    audio_frames.append(data)

            else:
                # Сигнал отсутствует
                if vox_active:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= VOX_SILENCE_TIME:
                        # Тишина длится достаточно долго - останавливаем запись
                        vox_active = False
                        logging.info("VOX деактивирован - обработка записи")

                        # Копируем фреймы под блокировкой и очищаем буфер
                        with audio_lock:
                            frames_copy = list(audio_frames)
                            audio_frames.clear()

                        duration = get_recording_duration_seconds(frames_copy)
                        if duration < MIN_RECORDING_DURATION:
                            print(
                                f"Запись отклонена: длительность {duration:.1f} с "
                                f"< минимальных {MIN_RECORDING_DURATION:.1f} с"
                            )
                            silence_start = None
                            continue

                        # Сохраняем и конвертируем аудио из копии
                        ogg_file = save_and_convert_audio(frames_copy)

                        # Отправляем в Telegram
                        if ogg_file and app_instance and main_loop:
                            try:
                                fut = run_coroutine_threadsafe(send_voice_message(ogg_file), main_loop)
                                fut.result()
                            except Exception as e:
                                logging.exception("Ошибка при отправке сообщения: %s", e)
                        silence_start = None
                else:
                    silence_start = None
        
    except KeyboardInterrupt:
        logging.info("VOX мониторинг остановлен")
    except Exception as e:
        logging.exception("Ошибка в VOX мониторинге: %s", e)
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
    
    logging.info("Бот запущен. Ожидание сообщений и мониторинг VOX...")
    
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