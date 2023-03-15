FROM pytorch/pytorch:latest

WORKDIR /

RUN pip install runpod protobuf transformers icetk cpm_kernels

ADD model/ .

ADD handler.py .

CMD [ "python", "-u", "/handler.py" ]