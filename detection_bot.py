import telebot
import os
import cv2
import requests
import json
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from imageai.Detection import ObjectDetection
import matplotlib

# Настройка шрифтов для русского языка
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# Импортируем токен бота
from config import API_TOKEN

bot = telebot.TeleBot(API_TOKEN)

def download_model(model_url, model_path="models/yolov3.pt"):
    """
    Функция для скачивания модели YOLOv3
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if not os.path.exists(model_path):
        print(f"SafetyAI: Скачиваем модель...")
        try:
            response = requests.get(model_url, stream=True)
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("SafetyAI: Модель успешно скачана!")
        except Exception as e:
            print(f"SafetyAI: Ошибка при скачивании модели: {e}")
            return None
    else:
        print("SafetyAI: Модель уже существует")

    return model_path

def detect_objects_on_image(input_image_path, min_probability=40):
    """
    Детектирует все объекты на изображении
    """
    print("SafetyAI: Инициализация детектора...")

    # Скачиваем модель
    model_url = "https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/yolov3.pt"
    model_path = download_model(model_url)

    if not model_path:
        return []

    # Создаем и настраиваем детектор
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path)
    detector.loadModel()

    print("SafetyAI: Выполняется детекция объектов...")

    # Создаем временный файл для выходного изображения
    output_image_path = "temp_detection.jpg"

    # Детектируем объекты
    detections = detector.detectObjectsFromImage(
        input_image=input_image_path,
        output_image_path=output_image_path,
        minimum_percentage_probability=min_probability
    )

    print(f"SafetyAI: Обнаружено {len(detections)} объектов")
    return detections, output_image_path

def analyze_objects(detections):
    """
    Анализирует и фильтрует объекты
    """
    # Расширенный список категорий
    object_categories = {
        'person': {'russian': 'человек', 'danger_level': 'высокий'},
        'bicycle': {'russian': 'велосипед', 'danger_level': 'средний'},
        'car': {'russian': 'автомобиль', 'danger_level': 'средний'},
        'motorcycle': {'russian': 'мотоцикл', 'danger_level': 'высокий'},
        'bus': {'russian': 'автобус', 'danger_level': 'средний'},
        'truck': {'russian': 'грузовик', 'danger_level': 'высокий'},
        'train': {'russian': 'поезд', 'danger_level': 'очень высокий'},
        'traffic light': {'russian': 'светофор', 'danger_level': 'информационный'},
        'stop sign': {'russian': 'знак стоп', 'danger_level': 'высокий'},
        'cat': {'russian': 'кошка', 'danger_level': 'низкий'},
        'dog': {'russian': 'собака', 'danger_level': 'низкий'},
        'bird': {'russian': 'птица', 'danger_level': 'низкий'},
        'chair': {'russian': 'стул', 'danger_level': 'низкий'},
        'table': {'russian': 'стол', 'danger_level': 'низкий'}
    }

    filtered_objects = []
    danger_count = 0

    for detection in detections:
        object_name = detection['name'].lower()
        if object_name in object_categories:
            # Добавляем расширенную информацию
            detection['russian_name'] = object_categories[object_name]['russian']
            detection['danger_level'] = object_categories[object_name]['danger_level']
            detection['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if object_categories[object_name]['danger_level'] in ['высокий', 'очень высокий']:
                danger_count += 1

            filtered_objects.append(detection)
        else:
            # Для неизвестных объектов используем базовую информацию
            detection['russian_name'] = object_name
            detection['danger_level'] = 'неизвестно'
            detection['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            filtered_objects.append(detection)

    return filtered_objects, danger_count

def calculate_distance(box_points, image_width):
    """
    Оценочный расчет расстояния до объекта на основе размера bounding box
    """
    x1, y1, x2, y2 = box_points
    object_width = x2 - x1
    object_height = y2 - y1

    # Простая эвристика: чем больше объект относительно изображения, тем он ближе
    size_ratio = (object_width * object_height) / (image_width * image_width)

    if size_ratio > 0.3:
        return "очень близко (<10м)"
    elif size_ratio > 0.15:
        return "близко (10-25м)"
    elif size_ratio > 0.05:
        return "средняя дистанция (25-50м)"
    else:
        return "далеко (>50м)"

def draw_detection_results(image_path, detections, output_path="detected_result.jpg"):
    """
    Рисует улучшенную визуализацию с bounding boxes и информацией
    """
    # Загружаем изображение с помощью PIL для лучшей поддержки текста
    pil_image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(pil_image)

    # Пытаемся загрузить шрифт с поддержкой кириллицы
    try:
        font_paths = [
            'C:/Windows/Fonts/arial.ttf',
            'C:/Windows/Fonts/tahoma.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/System/Library/Fonts/Arial.ttf'
        ]

        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, 20)
                    break
                except:
                    continue

        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    image_width, image_height = pil_image.size

    for detection in detections:
        box_points = detection["box_points"]
        x1, y1, x2, y2 = box_points

        # Выбираем цвет в зависимости от уровня опасности
        if detection.get('danger_level', 'неизвестно') == 'очень высокий':
            color = (0, 0, 0)  # Красный
        elif detection.get('danger_level', 'неизвестно') == 'высокий':
            color = (0, 0, 0)  # Оранжевый
        elif detection.get('danger_level', 'неизвестно') == 'средний':
            color = (0, 0, 0)  # Желтый
        else:
            color = (0, 0, 0)  # Зеленый

        # Рисуем bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Подготавливаем текст
        russian_name = detection.get('russian_name', detection['name'])
        label = f"{russian_name} {detection['percentage_probability']:.1f}%"

        # Рисуем фон для текста
        bbox = draw.textbbox((x1, y1 - 35), label, font=font)
        draw.rectangle(bbox, fill=color)

        # Добавляем текст
        draw.text((x1 + 5, y1 - 35), label, font=font, fill=(255, 255, 255))

        # Добавляем информацию о расстоянии
        distance = calculate_distance(box_points, image_width)
        distance_text = f"Расстояние: {distance}"
        draw.text((x1, y2 + 5), distance_text, font=font, fill=color)

    # Добавляем временную метку
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_text = f"Анализ: {timestamp}"
    draw.text((10, 10), timestamp_text, font=font, fill=(255, 255, 255))

    # Сохраняем изображение
    pil_image.save(output_path)
    return output_path

def generate_short_caption(detections, danger_count):
    """
    Генерирует короткую подпись для фото (максимум 1024 символа)
    """
    caption = f"🔍 Обнаружено: {len(detections)} объектов"
    if danger_count > 0:
        caption += f" ⚠️ Опасных: {danger_count}"
    
    # Добавляем топ-3 самых уверенных объектов
    if detections:
        top_objects = sorted(detections, key=lambda x: x['percentage_probability'], reverse=True)[:3]
        objects_list = ", ".join([f"{obj.get('russian_name', obj['name'])}" for obj in top_objects])
        caption += f"\n📋 Топ-3: {objects_list}"
    
    return caption

def generate_detailed_report(detections, danger_count):
    """
    Генерирует подробный отчет отдельным сообщением
    """
    if not detections:
        return "❌ На изображении не обнаружено объектов."
    
    report = f"📊 **ДЕТАЛЬНЫЙ ОТЧЕТ**\n\n"
    report += f"• Всего объектов: **{len(detections)}**\n"
    report += f"• Потенциально опасных: **{danger_count}**\n\n"
    
    # Группируем объекты по типам для компактности
    object_counts = {}
    for obj in detections:
        obj_name = obj.get('russian_name', obj['name'])
        if obj_name not in object_counts:
            object_counts[obj_name] = {
                'count': 0,
                'max_confidence': 0,
                'danger_level': obj.get('danger_level', 'неизвестно')
            }
        object_counts[obj_name]['count'] += 1
        if obj['percentage_probability'] > object_counts[obj_name]['max_confidence']:
            object_counts[obj_name]['max_confidence'] = obj['percentage_probability']
    
    report += "**Обнаруженные объекты:**\n"
    for obj_name, info in object_counts.items():
        danger_icon = "⚠️" if info['danger_level'] in ['высокий', 'очень высокий'] else "✅"
        report += f"{danger_icon} **{obj_name}**: {info['count']} шт. (макс. уверенность: {info['max_confidence']:.1f}%)\n"
    
    # Добавляем информацию о самых уверенных обнаружениях
    if detections:
        most_confident = max(detections, key=lambda x: x['percentage_probability'])
        report += f"\n🎯 **Самый уверенный объект**: {most_confident.get('russian_name', most_confident['name'])} "
        report += f"({most_confident['percentage_probability']:.1f}%)"
    
    report += f"\n\n⏰ Время анализа: {datetime.now().strftime('%H:%M:%S')}"
    
    # Обрезаем отчет если слишком длинный (максимум 4096 символов для Telegram)
    if len(report) > 4000:
        report = report[:4000] + "\n\n... (отчет сокращен)"
    
    return report

def split_message(text, max_length=4096):
    """
    Разделяет длинное сообщение на части
    """
    if len(text) <= max_length:
        return [text]
    
    parts = []
    while text:
        if len(text) <= max_length:
            parts.append(text)
            break
        else:
            # Находим последний перенос строки в пределах лимита
            split_pos = text.rfind('\n', 0, max_length)
            if split_pos == -1:
                # Если нет переносов, разбиваем по точкам
                split_pos = text.rfind('.', 0, max_length)
            if split_pos == -1:
                # Если нет точек, разбиваем по пробелам
                split_pos = text.rfind(' ', 0, max_length)
            if split_pos == -1:
                # Если нет пробелов, принудительно обрезаем
                split_pos = max_length
            
            parts.append(text[:split_pos])
            text = text[split_pos:].lstrip()
    
    return parts

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """
    Обработчик команд /start и /help
    """
    welcome_text = """
🚀 **SafetyAI Bot - Детекция объектов на изображениях**

Я могу анализировать ваши фотографии и находить на них различные объекты.

**Как использовать:**
1. Просто отправьте мне фотографию
2. Я проанализирую её и найду все объекты
3. Вы получите обработанное изображение с выделенными объектами
4. И подробный отчет о том, что было обнаружено

**Команды:**
/start - показать это сообщение
/help - помощь
/status - статус бота

Отправьте мне фотографию для анализа! 📸
    """
    bot.reply_to(message, welcome_text, parse_mode='Markdown')

@bot.message_handler(commands=['status'])
def send_status(message):
    """
    Обработчик команды /status
    """
    status_text = "✅ Бот активен и готов к работе!\nМодель детекции объектов загружена."
    bot.reply_to(message, status_text)

@bot.message_handler(content_types=['photo'])
def handle_photos(message):
    """
    Обработчик фотографий - основная функция детекции
    """
    try:
        # Отправляем сообщение о начале обработки
        processing_msg = bot.reply_to(message, "🔄 Обрабатываю изображение... Пожалуйста, подождите.")
        
        # Скачиваем фото
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Сохраняем временный файл
        input_image_path = f"temp_input_{message.chat.id}.jpg"
        with open(input_image_path, 'wb') as new_file:
            new_file.write(downloaded_file)
        
        # Выполняем детекцию объектов
        detections, temp_output_path = detect_objects_on_image(input_image_path)
        
        if not detections:
            bot.edit_message_text("❌ На изображении не обнаружено объектов.", 
                                message.chat.id, processing_msg.message_id)
            # Удаляем временные файлы
            os.remove(input_image_path)
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            return
        
        # Анализируем объекты
        filtered_objects, danger_count = analyze_objects(detections)
        
        # Создаем визуализацию
        output_image_path = draw_detection_results(input_image_path, filtered_objects, 
                                                 f"result_{message.chat.id}.jpg")
        
        # Генерируем короткую подпись для фото
        photo_caption = generate_short_caption(filtered_objects, danger_count)
        
        # Генерируем подробный отчет
        detailed_report = generate_detailed_report(filtered_objects, danger_count)
        
        # Отправляем обработанное изображение с короткой подписью
        with open(output_image_path, 'rb') as photo:
            bot.send_photo(message.chat.id, photo, caption=photo_caption)
        
        # Отправляем подробный отчет отдельным сообщением
        report_parts = split_message(detailed_report)
        for part in report_parts:
            bot.send_message(message.chat.id, part, parse_mode='Markdown')
        
        # Удаляем сообщение о обработке
        bot.delete_message(message.chat.id, processing_msg.message_id)
        
        # Удаляем временные файлы
        os.remove(input_image_path)
        os.remove(output_image_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
            
    except Exception as e:
        error_msg = f"❌ Произошла ошибка при обработке изображения: {str(e)}"
        try:
            bot.edit_message_text(error_msg, message.chat.id, processing_msg.message_id)
        except:
            bot.reply_to(message, error_msg)
        
        # Удаляем временные файлы в случае ошибки
        if 'input_image_path' in locals() and os.path.exists(input_image_path):
            os.remove(input_image_path)
        if 'output_image_path' in locals() and os.path.exists(output_image_path):
            os.remove(output_image_path)
        if 'temp_output_path' in locals() and os.path.exists(temp_output_path):
            os.remove(temp_output_path)

@bot.message_handler(func=lambda message: True)
def handle_text_messages(message):
    """
    Обработчик текстовых сообщений
    """
    if message.text:
        bot.reply_to(message, "📸 Отправьте мне фотографию для анализа объектов!")

def main():
    """
    Основная функция запуска бота
    """
    print("🚀 SafetyAI Bot запущен!")
    print("🤖 Бот готов к работе. Отправьте боту фотографию для анализа.")
    print("⏹️ Для остановки бота нажмите Ctrl+C")
    
    try:
        bot.infinity_polling()
    except KeyboardInterrupt:
        print("\n🛑 Бот остановлен.")
    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")

if __name__ == "__main__":
    # Создаем необходимые папки
    os.makedirs("models", exist_ok=True)
    
    # Запускаем бота
    main()