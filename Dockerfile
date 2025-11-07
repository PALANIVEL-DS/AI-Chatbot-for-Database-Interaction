# ==========================================
# 🧠 AI Chatbot for Database Interactions
# Streamlit + LangChain + HuggingFace
# ==========================================

# ---- 1️⃣ Base image ----
FROM python:3.10.19-slim

# ---- 2️⃣ Prevent Python from buffering output ----
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# ---- 3️⃣ Set working directory ----
WORKDIR /app

# ---- 4️⃣ Install system dependencies ----
RUN apt-get update && apt-get install -y \
    build-essential \
    unixodbc-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- 5️⃣ Copy requirements and install ----
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ---- 6️⃣ Copy the entire project ----
COPY . .

# ---- 7️⃣ Expose Streamlit’s default port ----
EXPOSE 8501

# ---- 8️⃣ Run Streamlit app ----
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
