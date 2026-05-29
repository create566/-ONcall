FROM python:3.11-slim

WORKDIR /app

# 安装 uv
RUN pip install uv

# 先复制依赖文件
COPY pyproject.toml uv.lock ./

# 预安装依赖（利用缓存）
RUN uv sync --frozen --no-install-project

# 复制剩余代码
COPY . .

# 暴露端口
EXPOSE 9900

# 启动命令
CMD [".venv/bin/python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9900"]