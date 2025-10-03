import logging
import os

def get_logger(logger_name, log_file_name):
    """
    创建并返回一个配置好的 logger
    
    Args:
        logger_name (str): logger 的名称（通常使用 __name__）
        log_file_name (str): 日志文件名（如 'code1.txt'）
    
    Returns:
        logging.Logger: 配置好的 logger 实例
    """
    # 获取项目根目录（假设 log.py 在 project 目录下）
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 创建 logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 创建日志文件的完整路径
        log_file_path = os.path.join(project_root, log_file_name)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(
            log_file_path, 
            mode='a', 
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # 添加处理器到 logger
        logger.addHandler(file_handler)
    
    return logger