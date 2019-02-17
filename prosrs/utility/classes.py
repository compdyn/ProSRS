"""
Copyright (C) 2016-2019 Chenchao Shou

Licensed under Illinois Open Source License (see the file LICENSE). For more information
about the license, see http://otm.illinois.edu/disclose-protect/illinois-open-source-license.

Define utility classes.
"""
import sys


class std_out_logger(object):
    """
    A class that handles logging standard outputs to a file.
    
    Usage:
        
        sys.stdout = std_out_logger(log_file) # redefine behavior of ``sys.stdout``
    """
    def __init__(self, log_file):
        """
        Args:
            
            log_file (str): File that standard outputs will be written to.
        """
        self.terminal = sys.stdout
        self.log = open(log_file, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self): # needed for Python 3 compatibility.
        pass


class std_err_logger(object):
    """
    A class that handles logging standard errors to a file.
    
    Usage:
        
        sys.stderr = std_err_logger(log_file) # redefine behavior of ``sys.stderr``
    """
    def __init__(self, log_file):
        """
        Args:
            
            log_file (str): File that standard errors will be written to.
        """
        self.terminal = sys.stderr
        self.log = open(log_file, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self): # needed for Python 3 compatibility.
        pass