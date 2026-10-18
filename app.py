import os
import cv2
import face_recognition
import numpy as np
from collections import Counter  # Добавили для подсчета голосов

def load_known_faces_multi(faces_dir="faces"):
    """
    Сканирует подпапки в faces/ и загружает несколько образцов для каждого человека.
    Структура: faces/имя_человека/фото1.jpg, фото2.jpg
    """
    known_encodings = []
    known_names = []
    
    if not os.path.exists(faces_dir):
        print(f"[-] Папка '{faces_dir}' не найдена. Создаю.")
        os.makedirs(faces_dir)
        return known_encodings, known_names

    print("[*] Индексация базы данных лиц (мульти-шаблон)...")
    
    for person_name in os.listdir(faces_dir):
        person_dir = os.path.join(faces_dir, person_name)
        
        if os.path.isdir(person_dir):
            print(f"[>] Сканирую образцы для пользователя: {person_name}")
            
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(person_dir, filename)
                    
                    try:
                        image = face_recognition.load_image_file(path)
                        encodings = face_recognition.face_encodings(image)
                        
                        if len(encodings) > 0:
                            known_encodings.append(encodings[0])
                            known_names.append(person_name)
                            print(f"  └─ [+] Загружен образец: {filename}")
                        else:
                            print(f"  └─ [!] Лицо на {filename} не обнаружено.")
                    except Exception as e:
                        print(f"  └─ [!] Ошибка файла {filename}: {e}")
                        
    return known_encodings, known_names

def main():
    known_encodings, known_names = load_known_faces_multi()
    
    if not known_encodings:
        print("[!] База данных лиц пуста. Система будет работать в режиме детекции Unknown.")

    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    process_this_frame = True

    # Заранее объявляем переменные, чтобы не было NameError на кадрах без обработки
    face_locations = []
    face_names = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Сжатие fx=0.5 и fy=0.5 для баланса скорости и точности
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                face_names = []
                for face_encoding in face_encodings:
                    name = "Unknown"

                    if known_encodings:
                        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                        
                        # Больше шаблонов - мягче порог
                        TOLERANCE = 0.60 
                        valid_matches = [i for i, dist in enumerate(face_distances) if dist < TOLERANCE]

                        if valid_matches:
                            matched_names = [known_names[i] for i in valid_matches]
                            
                            # достаем имя победителя из Counter
                            most_common = Counter(matched_names).most_common(1)
                            winner_name = most_common[0][0]  #
                        
                            # Фильтруем дистанции ТОЛЬКО для этого пользователя
                            user_distances = [face_distances[i] for i in valid_matches if known_names[i] == winner_name]
                            
                            if user_distances:
                                # Для вывода процентов берем НАИЛУЧШУЮ (минимальную) дистанцию среди совпадений
                                best_distance = min(user_distances)
                            
                            # # Нелинейная калибровка расстояния в проценты сходства
                                # if best_distance > 0.5:
                                    # linear_percent = ((1.0 - best_distance) / (1.0 - 0.5)) * 0.5
                                # else:
                                    # linear_percent = 0.5 + ((0.5 - best_distance) / 0.5) * 0.5
                                    
                                # accuracy = int(linear_percent * 100)
                                
                                # # Ограничиваем сверху, чтобы не было более 99% без абсолютного совпадения
                                # accuracy = min(accuracy, 98) 
                                # name = f"{winner_name} ({accuracy}%)"
                                if user_distances:
                                    best_distance = min(user_distances)
                            
                            # НОВАЯ АДАПТИВНАЯ КАЛИБРОВКА (Сдвигаем веса в сторону человеческого восприятия)
                                    if best_distance >= 0.6:
                                        accuracy = 0
                                    elif best_distance > 0.45:
                                        # Зона сомнения (от 0.45 до 0.60) -> покажет 50% - 75%
                                        accuracy = int(((0.60 - best_distance) / 0.15) * 25 + 50)
                                    else:
                                        # Зона высокой уверенности (меньше 0.45) -> растягиваем от 75% до 98%
                                        accuracy = int(75 + ((0.45 - best_distance) / 0.45) * 23)
                                
                            # Ограничиваем пиковое значение, чтобы не мелькало 100% при шумах камеры
                                    accuracy = min(max(accuracy, 0), 98)
                                    
                                    name = f"{winner_name} ({accuracy}%)"

                    face_names.append(name)


                face_names.append(name)

        process_this_frame = not process_this_frame

        # Отрисовка (Коэффициент * 2, так как сжатие fx=0.5)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 2; right *= 2; bottom *= 2; left *= 2 
            
            color = (0, 255, 0) if "Unknown" not in name else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow('Multi-Template Face ID (KNN Voting)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()