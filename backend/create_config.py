# create_config.py (改进版)
from config_manager import config_manager

if __name__ == "__main__":
    print("=== Create Encrypted Config ===")
    print("Leave API Key empty to skip")
    print("//请在这里输入你的阿里云api")
    config_data = {}

    # 可以添加更多配置项
    # config_data["OTHER_SETTING"] = input("Other setting: ")

    config_manager.create_encrypted_config(config_data)
