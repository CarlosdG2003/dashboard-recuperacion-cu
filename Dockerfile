# Imagen base con Python 3.13.7
FROM python:3.13.7-slim

# Establecer directorio de trabajo
WORKDIR /app

# Evitar .pyc y forzar salida sin buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto
COPY . .

# Exponer el puerto de Streamlit
EXPOSE 8501

# Ejecutar la creación/actualización de DuckDB y luego Streamlit
CMD python create_duckdb.py && streamlit run src/app.py --server.address=0.0.0.0
