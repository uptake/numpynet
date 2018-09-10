""""
This module makes logging easy and fun!  If no arguments are given to
startLog it will create a log in the working directory ('.') and
name it using the main script name ('mainScript.log')

This has been updated to an object for multiprocessing support
loggerLegacy still exists for legacy support

Usage:
to start your log:  from logger import log # makes the logs global
                    log.setLevel('INFO') # Set the logging level
use a log in a sub: from logger import log
                    log.out.info('Starting a log for Kenny Loggins.')
to release logs:    log.stopLog()
to make another global log, such as for a data output:
                    dataLog = logger.logObject(fileName = "data_density.csv",
                                   header = "messageOnly", onConsole = False)

@author: Brad Beechler (brad.beechler@uptake.com)
# Last Modification: 09/15/2017 (Brad Beechler)
"""
import logging
import os
import sys
import errno
from shutil import copyfile
import __main__


class LogObject:
    out = None
    filePath = None
    fileName = None
    fileFullName = None
    header = None
    onConsole = None
    onFile = None


    def __init__(self, filePath=None, fileName=None, header=None, onConsole=True, onFile=True):
        """
        Initialize log object.
        """
        self.out = None
        self.fileFullName = None
        self.filePath = filePath
        self.fileName = fileName
        self.header = header
        self.onConsole = onConsole
        self.onFile = onFile
        self.startLog(filePath=self.filePath, fileName=self.fileName, header=self.header,
                      onConsole=self.onConsole, onFile=self.onFile)

    def startLog(self, filePath=None, fileName=None, header=None, onConsole=True, onFile=True):
        """
        Initialize log file with optional path and optional file name.
        Parameters:
        filePath - where you want the log file
        fileName - what you want it called
        header - to specify the header style options are:
                "" - <time><level><functionName><message>
                "simple" - <level><message>
                "message_only" - just the message
                "no_time" - remove time
                "<custom header>" - enter your own format
        onConsole - set true for console output
        """
        self.header = header
        self.onConsole = onConsole
        self.onFile = onFile

        # Create Global Logger
        try:
            self.out = logging.getLogger(str(__main__.__file__)+'_logger')

        except AttributeError:
            self.out = logging.getLogger(str(__name__+'_logger'))

        # Clear old handlers
        self.out.handlers = []
        # Set Logger Level
        self.out.setLevel(logging.DEBUG)

        # create formatter and add to handlers
        if self.header is None:
            formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(funcName)20s();%(message)s")
        elif self.header == "message_only":
            formatter = logging.Formatter("%(message)s")
        elif self.header == "simple":
            formatter = logging.Formatter("%(levelname)s;%(message)s")
        elif self.header == "no_time":
            formatter = logging.Formatter("%(levelname)s;%(funcName)20s();%(message)s")
        else:
            formatter = self.header

        if self.onFile:
            if filePath is None:
                self.filePath = os.getcwd()
            else:
                self.filePath = filePath

            try:
                main_filename = os.path.splitext(os.path.basename(__main__.__file__))[0]
            except AttributeError:
                main_filename = os.path.splitext(os.path.basename(__name__))[0]

            if fileName is None:
                # If not provided, the fileName is set to the top level file calling the global logger.
                self.fileName = main_filename + '.log'
            else:
                self.fileName = fileName

            if os.access(self.filePath, os.W_OK):
                self.fileFullName = os.path.join(self.filePath, self.fileName)
                flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
                try:  # checks if a file is there, if not initialize it (from StackO)
                    os.open(self.fileFullName, flags)
                except OSError as e:
                    if e.errno == errno.EEXIST:  # Failed as the file already exists.
                        pass
                    else:  # Something unexpected went wrong so reraise the exception.
                        raise
                else:  # No exception, so the file must have been created successfully.
                    with open(self.fileFullName, 'w') as file_obj:
                        file_obj.write(main_filename + ' Log File\n')
            else:
                sys.exit('ERROR! Path specified not able to be written to!')

            # create console and file handler and set level to debug
            file_handle = logging.FileHandler(self.fileFullName)
            file_handle.setLevel(logging.DEBUG)
            file_handle.setFormatter(formatter)
            self.out.addHandler(file_handle)

        if self.onConsole:
            # Make a handler to deal with warnings and above --> stderr
            error_handle = logging.StreamHandler(sys.stderr)
            error_handle.setLevel(logging.WARNING)
            error_handle.setFormatter(formatter)
            self.out.addHandler(error_handle)
            # Make a handler to deal with all --> stdout
            standard_handle = logging.StreamHandler(sys.stdout)
            standard_handle.setLevel(logging.DEBUG)
            standard_handle.setFormatter(formatter)
            self.out.addHandler(standard_handle)


    def stopLog(self):
        """
        Release logging handlers.
        """
        self.out.handlers = []


    def setLevel(self,level_tag):
        """
        Sets the logging level
        levelStr (str) - a string describing the desired logging level
                        'INFO', 'DEBUG', 'WARNING', also 'NOTSET'
        """
        self.out.setLevel(logging.getLevelName(level_tag))


    def changeFileName(self, new_name, header=None, onConsole=True, onFile=True):
        """
        Change the name of the log
        new_name
        header (str) - A string describing the type of header information you
                       want with the logs passed to startLog: 'simple',
                       'message_only', 'no_time', <custom header>
                       (see startLog for more details on formatting)
        onConsole (bool) - True if you want logs written to the console
        onFile (bool)    - True if you want the log written to the file
        """
        # Get rid of tildas
        new_name = os.path.expanduser(new_name)
        # Check if name is okay
        # Copy old log to new name
        new_path, new_filename = os.path.split(new_name)
        if new_path == '':
            new_path = self.filePath
        new_full_filename = os.path.join(new_path, new_filename)
        if os.access(new_path, os.W_OK):
            self.out.handlers = []  # clear old handlers
            if self.onFile:
                copyfile(self.fileFullName, new_full_filename)
                os.remove(self.fileFullName)
            self.startLog(filePath=new_path, fileName=new_filename, header=header,
                          onConsole=onConsole, onFile=onFile)
        else:
            log.out.warning("No permissions to write new log name")


# Make a global log object to share
log = LogObject()
