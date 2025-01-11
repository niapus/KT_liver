# Используем официальный Python образ в качестве базового
FROM python:3.10-slim

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Копируем файл с зависимостями
COPY requirements.txt .

# Создаем виртуальное окружение
RUN python -m venv /app/venv

# Активируем виртуальное окружение и устанавливаем зависимости
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Копируем весь проект в контейнер
COPY . .

# Открываем порт
EXPOSE 8000

# Указываем, что используется только CPU
ENV TF_FORCE_CPU_ONLY=true

# Команда для запуска приложения
CMD ["/app/venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]