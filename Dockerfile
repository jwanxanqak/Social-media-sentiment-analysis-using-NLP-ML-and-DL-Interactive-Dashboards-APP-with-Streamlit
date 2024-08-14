# Usar una imagen base
FROM python:3.11

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo requirements.txt al directorio de trabajo en el contenedor
COPY ./requirements.txt /app/requirements.txt

# Instalar las dependencias
RUN pip install -r requirements.txt

# Copiar el resto de los archivos necesarios para la aplicación
COPY . /app

# Informar a Docker que el contenedor escuchará en el puerto 8501 en tiempo de ejecución
EXPOSE 8501

# Configurar el contenedor para que se ejecute como una aplicación Streamlit
ENTRYPOINT ["streamlit", "run"]

# Comando para ejecutar la aplicación
CMD ["app.py", "--server.port=8501", "--server.address=0.0.0.0"]