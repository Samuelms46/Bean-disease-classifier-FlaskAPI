FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=app1.py
ENV FLASK_RUN_HOST=0.0.0.0

# Expose the port (Render will set the PORT environment variable)
EXPOSE $PORT

# Run the app with Gunicorn (reduced workers for memory efficiency)
CMD ["sh", "-c", "gunicorn --timeout 300 --workers 1 --max-requests 100 --max-requests-jitter 10 -b 0.0.0.0:${PORT:-8080} app1:app"]