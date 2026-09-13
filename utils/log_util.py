import logging
import os

def setup_logging():
    """配置日志系统，只应调用一次"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    print("日志系统已初始化")

def get_logger(name=None):
    """获取logger实例"""
    if not logging.getLogger().handlers:
        setup_logging()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger

# 当直接运行此文件时测试日志
if __name__ == '__main__':
    setup_logging()
    logger = get_logger(__name__)
    logger.info("日志工具测试成功")