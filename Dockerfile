FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=app1.py
ENV FLASK_RUN_HOST=0.0.0.0

# Expose the port
EXPOSE 8080

# Run the app with Gunicorn (4 workers, port 8080)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app1:app"]