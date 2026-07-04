import sys
import logging

logger = logging.getLogger(__name__)
def log_exception(error):
    exc_type, exc_obj,exc_tb = sys.exc_info()
    logger.error(
        f"""
            Exception Type: {exc_type.__name__}\n
            Error Message: {str(error)}\n
            File name : {exc_tb.tb_frame.f_code.co_filename}
            Line Number : {exc_tb.tb_lineno}
            """
    )