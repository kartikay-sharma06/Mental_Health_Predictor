import sys 


def error_msg_info(err,err_info:sys):
    _,_,exc_tb=err_info

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    err_msg=(f"Error occurred in Python script name [{file_name}] "
        f"line number [{line_number}] "
        f"error message [{str(err)}]"
    )

    return err_msg



class CustomException(Exception):
    def __init__(self,err_msg,err_info:sys):
        super().__init__(err_msg)


        self.error_message = error_msg_info(
            err_msg,
            err_info
        )

    def __str__(self):
        return self.error_message