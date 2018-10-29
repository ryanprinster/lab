FROM ubuntu:18.04

# Install dependencies.
# g++ (v. 5.4) does not work: https://github.com/tensorflow/tensorflow/issues/13308
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    zip \
    unzip \
    software-properties-common \
    pkg-config \
    g++-4.8 \
    zlib1g-dev \
    python \
    lua5.1 \
    liblua5.1-0-dev \
    libffi-dev \
    gettext \
    freeglut3 \
    libsdl2-dev \
    libosmesa6-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    python-dev \
    build-essential \
    git \
    python-setuptools \
    python-pip \
    libjpeg-dev \
    python-numpy-dev


# Install bazel
    # The Following Breaks for bazel 0.17.1
    #
    # RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | \
    # tee /etc/apt/sources.list.d/bazel.list && \
    # curl https://bazel.build/bazel-release.pub.gpg | \
    # apt-key add - && \
    # apt-get update && apt-get install -y bazel
    #
    #
RUN export BAZEL_VERSION=0.16.1 && \
    wget https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    sed -i 's@sudo@, @g' bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    chmod +x bazel-$BAZEL_VERSION-installer-linux-x86_64.sh &&\
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh


# Install TensorFlow and other dependencies
RUN pip install tensorflow==1.9.0 dm-sonnet

# Install Tensorforce (cpu)
# RUN pip install tensorforce[tf]

# Build and install DeepMind Lab pip package.
# We explicitly set the Numpy path as shown here:
# https://github.com/deepmind/lab/blob/master/docs/users/build.md
RUN NP_INC="$(python -c 'import numpy as np; print(np.get_include())[5:]')" && \
    git clone https://github.com/deepmind/lab.git && \
    cd lab && \
    sed -i 's@hdrs = glob(\[@hdrs = glob(["'"$NP_INC"'/\*\*/*.h", @g' python.BUILD && \
    sed -i 's@includes = \[@includes = ["'"$NP_INC"'", @g' python.BUILD && \
    bazel build -c opt python/pip_package:build_pip_package && \
    pip install wheel && \
    ./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg && \
    pip install /tmp/dmlab_pkg/DeepMind_Lab-1.0-py2-none-any.whl --force-reinstall

# install tensorforce manually (to support latest dev branch)
RUN git clone https://github.com/ducandu/tensorforce
WORKDIR tensorforce
RUN git checkout master
RUN git pull
RUN python setup.py install
WORKDIR ../

# install tqdm
RUN pip install tqdm


# RUN apt-get update && apt-get install -y \
#	dirmngr \
#	gnupg \
#	--no-install-recommends \
#	&& apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 0AB215679C571D1C8325275B9BDB3D89CE49EC21 \
#	&& echo "deb http://ppa.launchpad.net/mozillateam/firefox-next/ubuntu xenial main" >> /etc/apt/sources.list.d/firefox.list \
#	&& apt-get update && apt-get install -y \
#	apulse \
#	ca-certificates \
#	firefox \
#	hicolor-icon-theme \
#	libasound2 \
#	libpulse0 \
#	--no-install-recommends \
#	&& rm -rf /var/lib/apt/lists/*

ENV LANG en-US

COPY local.conf /etc/fonts/local.conf

# RUN echo 'pref("browser.tabs.remote.autostart", false);' >> /etc/firefox/syspref.js

# COPY entrypoint.sh /usr/bin/startfirefox

ENTRYPOINT bash

