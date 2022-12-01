FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ADD setup.sh /
RUN ls -la /
RUN chmod +x /setup.sh
RUN bash /setup.sh

RUN rm -rf /root/.cache/pip

CMD [ "/bin/bash" ]
