FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py webui.py manage_users.py ./
COPY static ./static
# model cache, recordings and data (users, settings, secret) are mounted as volumes
EXPOSE 8765
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8765"]
