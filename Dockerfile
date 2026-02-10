# 基于官方 Python 镜像的 Dockerfile
# 适用于 shaper 项目，默认启动 server.py

FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY win/requirements.txt ./

# 安装依赖 (排除桌面端库 pywebview 以减小体积)
RUN grep -v "pywebview" requirements.txt > requirements.tmp && \
    pip install --no-cache-dir -r requirements.tmp && \
    rm requirements.txt requirements.tmp

# 复制项目文件
COPY . .

# 暴露端口（如 server.py 默认 5555）
EXPOSE 5555

# 启动服务
CMD ["python", "server.py"]
