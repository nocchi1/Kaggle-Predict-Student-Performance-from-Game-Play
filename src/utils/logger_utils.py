from pathlib import Path
from datetime import datetime
import time
import inspect
from typing import Union, Any, Type


class Logger:
    
    def __init__(self, log_dir: Path, exp_name: str, params: Union[dict, Type[Any]] = None):
        exist_file_num = len(list(log_dir.glob('log_*.txt')))
        file_number = exist_file_num + 1
        self.log_path = log_dir / f'log_{file_number}.txt'
        self.log_path.touch()
        self._init_logger(exp_name, params)

    def _init_logger(self, exp_name: str, params: dict):
        date_time = str(datetime.fromtimestamp(time.time()))
        init_messeage = f'[{date_time}] EXP NAME: {exp_name}'
        self._log(init_messeage)
        self._log(f'\n' + '-' * 100 + '\n')

        if params:
            if inspect.isclass(params):
                params = {name: getattr(params, name) for name in dir(params) if not name.startswith('__')}
            self._log(f'\n' + '[HyperParameters]' + '\n' * 2)
            for name, val in params.items():
                self._log(f'{name}: {val} \n')
            self._log(f'\n' + '-' * 100 + '\n')
            
    def _log(self, messeage):
        with open(self.log_path, 'a') as f:
            f.write(messeage)
        
    def log(self, name: str = '', content: str = ''):
        date_time = str(datetime.fromtimestamp(time.time()))
        messeage = f'\n[{date_time}][{name}] -->> {content}'
        self._log(messeage)