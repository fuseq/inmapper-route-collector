FROM python:3.12-slim

WORKDIR /app

# Sistem bağımlılıkları (xml parsing için)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyaları
COPY helpers/ helpers/
COPY files/ files/
COPY viewer_app.py .
COPY wsgi.py .
COPY route_viewer_dynamic.html .

# Port (CapRover bunu 80 olarak ayarlar)
ENV PORT=80
ENV PYTHONUNBUFFERED=1
ENV VENUE=zorlu

EXPOSE 80

# Gunicorn ile çalıştır (preload ile venue verileri bir kez yüklenir)
CMD gunicorn wsgi:app \
    --bind 0.0.0.0:${PORT} \
    --workers 2 \
    --timeout 120 \
    --preload



