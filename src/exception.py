import sys
from logger import logging

def error_message_detail(error, error_details:sys):
    _ ,_ ,ext_tb = error_details.exc_info()
    file_name = ext_tb.tb_frame.f_code.co_filename
    line_number = ext_tb.tb_lineno
    error_message = f"Error occured in python script name [{file_name}] line number [{line_number}] error message [{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_details=error_details)
    def __str__(self):
        return self.error_message
        
if __name__=='__main__':
    try:
        a = 1/0
    except Exception as e:
        logging.info(CustomException(e, sys))
        raise CustomException(e, sys)
        
    
  