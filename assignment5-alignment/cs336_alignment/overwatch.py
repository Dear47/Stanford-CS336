import logging
import logging.config
import os
import json
import sys
from typing import Any, Dict, Optional, Union
from cs336_alignment.paths import log_path

try:
    from accelerate import PartialState
    _HAS_ACCELERATE = True
except ImportError:
    _HAS_ACCELERATE = False

# Overwatch Format
RICH_FORMATTER = "| >> %(message)s"
FILE_FORMATTER = "%(asctime)s - [%(filename)s:%(lineno)d] - %(funcName)s - %(levelname)s - %(message)s"
DATEFMT = "%m/%d [%H:%M:%S]"

def _get_rank():
    """Get current process's Rank"""
    if _HAS_ACCELERATE:
        return PartialState().process_index
    # Fallback: try to load the environmet variable (Torchrun default configuration)
    return int(os.environ.get("RANK", 0))

def setup_global_logging(log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """
    Configure and return a standard global Root Logger object.
    
    Args:
        name: The name of Logger
        log_file: The save path of log file (optional)
        level: Log level (INFO, DEBUG, etc.)
    """
    rank = _get_rank()
    is_main_process = (rank == 0)
    current_level = level if is_main_process else "ERROR"    

    # Configure basic logger dict
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "rich_console": {
                "format": RICH_FORMATTER, 
                "datefmt": DATEFMT
            },
            "standard_file": {
                "format": FILE_FORMATTER
            },
        },
        "handlers": {
            "console": {
                "class": "rich.logging.RichHandler",
                "formatter": "rich_console",
                "markup": True,
                "rich_tracebacks": True,
                "show_level": True,
                "show_path": False,
                "show_time": True,
                "level": current_level,
            }
        },
        "root": {
            "level": current_level, 
            "handlers": ["console"]
        },
    }

    # if specify the file path, add FileHandler
    if log_file and is_main_process:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "filename": log_file,
            "mode": "a",
            "formatter": "standard_file",
            "level": level,
            "encoding": "utf-8"
        }
        config["root"]["handlers"].append("file")

    logging.config.dictConfig(config)


class ExperimentLogger:
    def __init__(self, name="SFT-Experiment", log_file: str = "experiment.log"):
        """
        An encapsulated experimental log class,
        which can automatically handles main process log recording in a distributed environment.
        """
        setup_global_logging(log_file)

        self.logger = logging.getLogger(name)
        self.rank = _get_rank()

    def info(self, msg: str, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self.logger.error(f"[Rank {self.rank}] {msg}", *args, **kwargs)

    def log_metrics(self, step: int, metrics: Dict[str, Any], prefix: str = "Train"):
        """
        Specifically designed for recording SFT or Eval metrics.
        """
        if self.rank == 0:
            # console message
            metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
            console_msg = f"[bold cyan]{prefix}[/bold cyan] Step {step} | {metrics_str}"
            self.logger.info(console_msg, extra={"markup": True})

    def section(self, title: str):
        """
        Used to distinguish different stages of the experiment,
        print the dividing lines.
        """
        self.logger.info(f"", extra={"markup": True})
        self.logger.info(f"[bold yellow]{'='*10} {title} {'='*10}[/bold yellow]", extra={"markup": True})
        self.logger.info(f"", extra={"markup": True})

# === 使用示例 ===
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 初始化
    exp_logger = ExperimentLogger(name="Example-log", log_file=str(log_path("example.log")))
    
    # 记录普通信息
    exp_logger.info(f"""{"="*40}
⚙️ 开始加载模型 Qwen2.5-Math-1.5B ...
first new line
second new line""")
    
    # 打印带分割线的阶段标题
    exp_logger.section("xxx Training Phase")
    
    # 模拟训练循环
    for step in range(1, 4):
        loss = 0.5 / step
        # 记录指标
        exp_logger.log_metrics(step, {"loss": loss, "lr": 1e-5}, prefix="Training")
        
    exp_logger.section("Evaluation Phase")
    exp_logger.info("Evaluation accuracy: [green]85.4%[/green]", extra={"markup": True})
