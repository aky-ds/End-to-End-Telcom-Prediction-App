import os
from pathlib import Path
import sys

class CustomException(Exception):
    def __int__(self,error_message,error_Detail:sys):
        self.error_message=error_message
        _,_,exct_b=self.error_Detail.exc_info()
        self.line_no=exct_b.tb_lineno
        self.file_name=exct_b.tb_frame.f_code.co_filename
    
    
    def __str__(self):
        return f'The error is {self.error_message}, in the file {self.file_name}, in the line {self.line_no}'
        