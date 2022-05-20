FROM jupyter/scipy-notebook
RUN pip install -U imbalanced-learn
RUN pip install joblib

COPY . /model
WORKDIR /model
CMD python3 predict.py
