import base64
import getpass
import json
import os
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class ConfigManager:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path.cwd() / ".config" / "config.enc"
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(exist_ok=True, parents=True)
        self._config_cache = None

    def create_encrypted_config(self, config_data=None):
        """创建加密配置文件"""
        if config_data is None:
            config_data = {}

        # 交互式输入API密钥
        if "CLOUD_API_KEY" not in config_data:
            api_key = input("Enter your API Key: ").strip()
            if api_key:
                config_data["CLOUD_API_KEY"] = api_key

        # 设置和确认密码
        while True:
            password = getpass.getpass("Set encryption password: ")
            confirm = getpass.getpass("Confirm password: ")
            if password == confirm:
                if password:  # 确保密码非空
                    break
                print("Password cannot be empty")
            else:
                print("Passwords do not match, try again.")

        # 生成盐和密钥
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        fernet = Fernet(key)

        # 使用JSON序列化配置
        config_str = json.dumps(config_data, indent=2)
        encrypted_data = fernet.encrypt(config_str.encode())

        # 写入文件
        with open(self.config_path, "wb") as f:
            f.write(salt + encrypted_data)

        self.config_path.chmod(0o600)
        print(f"✓ Encrypted config file created at: {self.config_path}")

    def load_config(self, force_reload=False) -> dict:
        """加载并解密配置"""
        if self._config_cache and not force_reload:
            return self._config_cache

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found at {self.config_path}\n"
                f"Run 'python create_config.py' to create one."
            )

        password = getpass.getpass("Enter decryption password: ")

        with open(self.config_path, "rb") as f:
            data = f.read()

        salt = data[:16]
        encrypted_data = data[16:]

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        fernet = Fernet(key)

        try:
            decrypted_data = fernet.decrypt(encrypted_data).decode()
            config_dict = json.loads(decrypted_data)
            self._config_cache = config_dict
            return config_dict
        except Exception as e:
            raise ValueError("Failed to decrypt config. Wrong password?") from e

    def get_api_key(self):
        """获取API密钥的便捷方法"""
        config = self.load_config()
        return config.get("CLOUD_API_KEY")


config_manager = ConfigManager()
