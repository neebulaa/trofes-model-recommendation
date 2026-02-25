# Gunakan Python 3.10 (Sama dengan env laptop kamu)
FROM python:3.10-slim

WORKDIR /code

# Copy requirements bersih tadi
COPY ./requirements.txt /code/requirements.txt

# Install library
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy file kode & model
COPY . .

# Jalankan server
CMD ["uvicorn", "API.app:app", "--host", "0.0.0.0", "--port", "7860"]