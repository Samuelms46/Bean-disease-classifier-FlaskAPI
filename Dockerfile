FROM python:3.12-slim

WORKDIR /app

COPY requirements-minimal.txt .

RUN pip install --no-cache-dir -r requirements-minimal.txt

# Copy application files and directories explicitly
COPY app1.py .
COPY templates/ templates/
COPY static/ static/
COPY models/ models/

ENV FLASK_APP=app1.py
ENV FLASK_RUN_HOST=0.0.0.0

# Expose the port (Render sets the PORT environment variable)
EXPOSE $PORT

# Run the app with Gunicorn (minimal configuration for free tier)
CMD ["sh", "-c", "gunicorn --timeout 300 --workers 1 --threads 1 --worker-class sync --max-requests 10 --preload -b 0.0.0.0:${PORT:-8080} app1:app"]