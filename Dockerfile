FROM python:3.8

WORKDIR /app

ADD requirements.txt /app/

ADD . /app/ 
RUN pip install -r requirements.txt
ENV MODEL_PATH="/app/assets/Logistic Regression_v1.0.0.joblib"
ENV VECTORIZER_PATH="/app/assets/vectorizer.pickle"
ENV UNCERTAINTY_THRESHOLD=0.6

CMD [ "python", "app.py"]