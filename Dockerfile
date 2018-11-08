FROM python:3.6-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc

# Install pytorch, fastai and Flask
RUN pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
RUN pip install fastai
RUN pip install Flask

ADD models models
ADD predict.py predict.py
ADD mushrooms.csv mushrooms.csv
ADD static static
ADD templates templates

# Run it once to trigger resnet download
RUN python predict.py

EXPOSE 8008

# Start the server
CMD ["python", "predict.py", "serve"]

# export FLASK_APP=predict.py
# flask run --host=0.0.0.0