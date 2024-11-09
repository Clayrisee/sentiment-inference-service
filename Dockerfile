FROM python:3.8

WORKDIR /app

COPY requirements.txt /app/
COPY assets/ /app/assets/
COPY nltkdata/ /app/nltkdata
COPY src /app/src/

RUN pip install -r requirements.txt

ENV AWS_ACCESS_KEY_ID=None
ENV AWS_SECRET_ACCESS_KEY=None
ENV AWS_DEFAULT_REGION="ap-southeast-1"

ENV MODEL_PATH="/app/assets/Logistic Regression_v1.0.0.joblib"
ENV VECTORIZER_PATH="/app/assets/vectorizer.pickle"
ENV UNCERTAINTY_THRESHOLD=0.6

COPY app.py /app/

CMD ["python", "app.py"]