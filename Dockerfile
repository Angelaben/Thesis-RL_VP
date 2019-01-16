# Base image
FROM tensorflow/tensorflow:latest-gpu


# Install some package
RUN apt-get -qq -y update && apt-get -y install python3 \
                                                zlib1g-dev \
                                                cmake \
                                                libgtk2.0-0 \
                                                libsm6 \
                                                libxext6 \
                                                wget \
                                                curl \
                                                vim \
                                                git \
                                                make \
                                                tar \
                                                nano \
                                                gcc \
                                                telnet \
                                                redis-server \
                                                python3-dev \
                                                python3-pip \
                                                python-pip \
                                                screen

# Copy folder inside the container
COPY . /code
# change active folder
WORKDIR /code

RUN pip install --upgrade pip
RUN pip install -r Requirements.txt
RUN python3 -m pip install ipykernel
RUN python3 -m ipykernel install --user
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install tensorflow numpy matplotlib tqdm seaborn gym tqdm pandas sklearn numba keras torch

# Launch file during the process
# ENTRYPOINT ["python"]
# CMD ["app.py"]
