import datetime
import numpy


class Logger:

    @staticmethod
    def divider(title):
        print(
            "\n============================================{}=============================================\n".format(
                title),flush=True)

    @staticmethod
    def info(message):
        time = datetime.datetime.now()
        timeStr = time.strftime("[%Y%m%d-%H:%M:%S]")
        print(timeStr + '=>[info]: {}'.format(message),flush=True)

    @staticmethod
    def getTimeStr(time):
        timeStr = time.strftime("[%m%d-%H:%M:%S]")
        return timeStr

