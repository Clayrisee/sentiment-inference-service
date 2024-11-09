FROM public.ecr.aws/lambda/python:3.8

# Install the function's dependencies using file requirements.txt
# from your project folder.
COPY assets ./assets/
COPY nltkdata ./nltkdata/
COPY src ./src/


COPY requirements_aws.txt  .
RUN  pip install -r requirements_aws.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]