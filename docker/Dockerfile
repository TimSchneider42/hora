FROM ros:noetic-ros-base

RUN apt-get update && apt-get -y install python3-venv wget

RUN python3 -m venv /venv
RUN PATH="/venv/bin:$PATH" pip install hydra-core termcolor omegaconf gym tensorboard tensorboardx torch torchvision

RUN mkdir /tmp/isaac/ && cd /tmp/isaac/ && \
  wget "https://drive.usercontent.google.com/download?id=1StaRl_hzYFYbJegQcyT7-yjgutc6C7F9&export=download&confirm=t" -O isaac4.tar.gz && \
  tar -xzf isaac4.tar.gz && \
  PATH="/venv/bin:$PATH" pip install -e ./isaacgym/python && \
  rm -r /tmp/isaac/

# Convenience
RUN mkdir retain && \
    touch /retain/.bash_history && ln -s /retain/.bash_history /root/.bash_history
RUN echo "PS1='\[\033[01;95m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> /etc/skel/.bashrc
RUN cp /etc/skel/.bash* /root/

ADD entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

