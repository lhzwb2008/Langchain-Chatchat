FROM cnstark/pytorch:2.0.1-py3.10.11-cuda11.8.0-ubuntu22.04

ENV PYTHONUNBUFFERED=1 



RUN mkdir /code
WORKDIR /code
COPY . /code

# 创建 pip 配置文件，并设置清华大学 TUNA 协会的镜像源
RUN mkdir -p ~/.config/pip \
    && echo "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple" > ~/.config/pip/pip.conf

RUN pip install --no-cache-dir --upgrade torch~=2.1.2
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements_api.txt
RUN pip install --no-cache-dir --upgrade -r requirements_webui.txt

RUN python startup.py -a


