FROM python:3.6-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc

# Install pytorch, fastai and Flask
RUN pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
RUN pip install fastai
RUN pip install Flask

ADD models/3_resnet34_defaults.pth models/3_resnet34_defaults.pth
ADD predict.py predict.py
ADD static static
ADD templates templates

# Run it once to trigger resnet download
RUN python predict.py

EXPOSE 8008

# Start the server
CMD ["python", "predict.py"]

# export FLASK_APP=predict.py
# flask run --host=0.0.0.0
