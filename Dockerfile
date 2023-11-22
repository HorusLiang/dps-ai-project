FROM python:3.9.16
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
CMD ["flask", "run", "--host=0.0.0.0"]
