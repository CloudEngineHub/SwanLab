import os
import sys
from datetime import datetime

from swankit.env import create_time
from swankit.log import FONT

from swanlab.swanlab_settings import get_settings


class SwanWriterProxy:
    """
    标准输出流拦截代理
    """

    def __init__(self):
        self.epoch = 0
        self.write_callback = None
        self.file = None
        """
        当前正在写入的文件句柄
        """
        self.write_handler = None
        """
        标准输出流原始句柄
        """
        self.__buffer = ""
        """
        上传到云端的缓冲区
        """
        self.console_folder = None
        """
        保存控制台输出文件夹路径
        """
        self.now = None
        """
        当前文件名称（不包含后缀）
        """
        self.max_upload_len = None

    @property
    def can_callback(self) -> bool:
        return self.write_callback is not None

    def set_write_callback(self, func):
        # 封装一层func，加入epoch处理逻辑
        def _func(message):
            self.epoch += 1
            func({"message": message, "create_time": create_time(), "epoch": self.epoch})

        # 封装第二层，加入message处理逻辑以及是否调用逻辑
        def _(message):
            if self.can_callback:
                # 上传到云端
                messages = message.split("\n")
                # 如果长度为1，说明没有换行符
                if len(messages) == 1:
                    self.__buffer = self.__buffer + messages[0]
                # 如果长度大于2，说明其中包含多个换行符
                elif len(messages) > 1:
                    _func(self.__buffer + messages[0])
                    self.__buffer = messages[-1]
                    for m in messages[1:-1]:
                        _func(m)

        self.write_callback = _

    def init(self, path):
        if path:
            self.console_folder = path
            # path 是否存在
            if not os.path.exists(path):
                os.makedirs(path)
            # 日志文件路径
            self.now = datetime.now().strftime("%Y-%m-%d")
            console_path = os.path.join(path, f"{self.now}.log")
            # 拿到日志文件句柄
            self.file = open(console_path, "a", encoding="utf-8")
        # 封装sys.stdout
        self.write_handler = sys.stdout.write
        self.stderr_write_handler = sys.stderr.write  # 新增

        def handle_message(message, is_stderr=False):
            try:
                # 如果是 stderr，也写入原始 stderr
                if is_stderr:
                    self.stderr_write_handler and self.stderr_write_handler(message)
                else:
                    self.write_handler and self.write_handler(message)
            except (UnicodeEncodeError, ValueError) as e:
                if isinstance(e, ValueError) and "I/O operation on closed file" not in str(e):
                    raise

            # 限制上传长度
            message = FONT.clear(message)[: self.max_upload_len]
            self.write_callback and self.write_callback(message)

            if path:
                # 检查文件分片
                now = datetime.now().strftime("%Y-%m-%d")
                if now != self.now:
                    self.now = now
                    if hasattr(self, "console") and not self.file.closed:
                        self.file.close()
                    self.file = open(os.path.join(self.console_folder, self.now + ".log"), "a", encoding="utf-8")

                self.file.write(message)
                self.file.flush()

        # 重定向 stdout.write
        def stdout_wrapper(message):
            handle_message(message, is_stderr=False)

        sys.stdout.write = stdout_wrapper

        # 新增：重定向 stderr.write
        def stderr_wrapper(message):
            handle_message(message, is_stderr=True)

        sys.stderr.write = stderr_wrapper
        # 设置最大上传长度
        self.max_upload_len = get_settings().max_log_length

    def reset(self):
        sys.stdout.write = self.write_handler
        sys.stderr.write = self.stderr_write_handler  # 新增
        self.file and self.file.close()
        self.file = None
        self.max_upload_len = None
        self.write_callback = None


class SwanConsoler:
    def __init__(self):
        """
        控制台输出重定向器
        """
        self.writer = SwanWriterProxy()
        self.__installed = False

    @property
    def installed(self):
        return self.__installed

    def uninstall(self):
        """重置输出为原本的样子"""
        self.writer.reset()
        self.__installed = False

    def install(self, console_dir):
        """"""
        self.writer.init(console_dir)
        self.__installed = True

    @property
    def write_callback(self):
        return self.writer.write_callback

    def set_write_callback(self, func):
        self.writer.set_write_callback(func)
