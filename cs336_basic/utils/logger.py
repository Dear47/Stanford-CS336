import logging
import os
import inspect

class ClassNameFormatter(logging.Formatter):
    """自定义 Formatter，在日志中自动添加 [ClassName] 前缀"""
    
    def format(self, record):
        # 获取调用栈信息
        try:
            # 获取调用日志的帧（通常是上上层，因为 logging 内部有一层）
            frame = inspect.currentframe()
            for _ in range(3):  # 跳过 logging 内部的几层
                if frame.f_back:
                    frame = frame.f_back
                else:
                    break
            
            # 尝试从 self 参数推断类名（如果是方法调用）
            args = frame.f_locals
            class_name = None
            if 'self' in args:
                class_name = args['self'].__class__.__name__
            elif 'cls' in args:
                class_name = args['cls'].__name__
            else:
                # 如果不是类方法，尝试从函数名或模块推测（可选）
                func_name = record.funcName
                # 可以保留原样，或标记为函数
                class_name = f"func:{func_name}"
            
            if class_name:
                record.msg = f"[{class_name}] {record.msg}"
        except Exception:
            # 如果提取失败，不中断日志，保持原样
            pass
        
        return super().format(record)


def get_logger(logger_name, log_file_name):
    """
    创建并返回一个配置好的 logger，自动在日志消息前添加 [ClassName]
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        log_file_path = os.path.join(project_root, log_file_name)
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 使用自定义 formatter
        formatter = ClassNameFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger