# Use an official Python runtime as a parent image
FROM python:3-slim

# Set the working directory to /app
WORKDIR /vgg16_base

#Copy the current directory contents into the container at /app
ADD . /vgg16_base

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
RUN apt-get -qq update \
  && apt-get -qq -y install libglib2.0-0 \
  && apt-get install -y libsm6 libxext6 \
  && apt-get install -y libgtk2.0-dev \
  && apt-get install -y tk 

CMD ["python3", "check_gpu.py"]
