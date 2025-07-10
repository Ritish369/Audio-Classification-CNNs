FROM python:3.10.11-slim
WORKDIR /usr/local/app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy requirements and install dependencies
COPY requirements.txt ./
COPY vastai/working/models ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import matplotlib.pyplot as plt"

# Copy the entire project (app.py and src folder)
COPY . .

# Create app user and set permissions
RUN useradd -m app && chown -R app /usr/local/app
USER app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]