FROM python:3.10
ENV PYTHONUNBUFFERED 1

RUN echo \
    deb https://mirrors.tuna.tsinghua.edu.cn/debian/ buster main contrib non-free\
    deb https://mirrors.tuna.tsinghua.edu.cn/debian/ buster-updates main contrib non-free\
    deb https://mirrors.tuna.tsinghua.edu.cn/debian/ buster-backports main contrib non-free\
    deb https://mirrors.tuna.tsinghua.edu.cn/debian-security buster/updates main contrib non-free



RUN apt-get clean && apt-get update && \
    apt-get -y install build-essential python3-dev default-libmysqlclient-dev


RUN mkdir /code
WORKDIR /code

RUN pip install pip -U -i https://pypi.tuna.tsinghua.edu.cn/simple
ADD requirements.txt /code/

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码
ADD . /code/

EXPOSE 8000

CMD python manage.py migrate && python manage.py runserver 0.0.0.0:8000