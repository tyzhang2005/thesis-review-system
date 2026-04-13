import atexit
import logging
import signal
import subprocess
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# vLLM服务器配置
VLLM_SERVER_HOST = "http://114.212.84.105"
VLLM_SERVER_PORT = 8003
VLLM_SERVER_URL = f"{VLLM_SERVER_HOST}:{VLLM_SERVER_PORT}"

# vLLM服务器进程
vllm_process = None


def start_vllm_server():
    """启动vLLM服务器"""
    global vllm_process
    vllm_process = subprocess.Popen(
        [
            "vllm",
            "serve",
            "/DATA/zhangtianyue_231300023/models/TreeDy2023/DeepSeek-R1-Distill-Qwen-32B-GGUF/DeepSeek-R1-Distill-Qwen-32B-Q4_K_L.gguf",
            "--tokenizer",
            "/DATA/zhangtianyue_231300023/models/TreeDy2023/DeepSeek-R1-Distill-Qwen-32B-GGUF",
            "--quantization",
            "gguf",
            "--tensor-parallel-size",
            "4",
            "--dtype",
            "float16",
            "--gpu-memory-utilization",
            "0.6",
            "--block-size",
            "8",
            "--max-model-len",
            "12000",
            "--max-num-seqs",
            "16",
            "--max-num-batched-tokens",
            "20000",
            "--swap-space",
            "16",
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_SERVER_PORT),
        ]
    )
    logger.info(f"vLLM服务器已启动，监听端口: {VLLM_SERVER_PORT}")
    return vllm_process


def stop_vllm_server():
    """停止vLLM服务器"""
    global vllm_process
    if vllm_process is not None:
        try:
            vllm_process.terminate()
            vllm_process.wait()
            logger.info("vLLM服务器已停止")
        except Exception as e:
            logger.error(f"停止vLLM服务器失败: {str(e)}")


if __name__ == "__main__":
    import atexit

    # 启动vLLM服务器
    start_vllm_server()

    # 注册退出处理函数，确保服务器在应用退出时停止
    atexit.register(stop_vllm_server)

    def handle_exit(signum, frame):
        logger.info("接收到终止信号，正在停止服务器...")
        exit(0)

    signal.signal(signal.SIGINT, handle_exit)  # 处理Ctrl+C
    signal.signal(signal.SIGTERM, handle_exit)  # 处理kill命令

    logger.info("服务器运行中，按 Ctrl+C 停止...")

    # 保持主线程运行
    while True:
        time.sleep(3600)
