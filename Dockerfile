FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=app1.py
ENV FLASK_RUN_HOST=0.0.0.0

# Expose the port (Render will set the PORT environment variable)
EXPOSE $PORT

# Run the app with Gunicorn (4 workers, using PORT from environment)
CMD ["sh", "-c", "gunicorn -w 4 -b 0.0.0.0:${PORT:-8080} app1:app"]