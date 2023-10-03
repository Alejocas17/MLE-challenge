# syntax=docker/dockerfile:1.2
FROM python:3.9
# put you docker configuration here# Usa una imagen base de Python


# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia los archivos del proyecto al contenedor
COPY ./challenge /app
COPY requirements.txt requirements-deploy.txt
# Instala las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Expone el puerto 8000, que es el puerto en el que FastAPI se ejecuta de forma predeterminada
EXPOSE 8000

# Comando para ejecutar la aplicación cuando se inicie el contenedor
# CMD ["python","api.py"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]