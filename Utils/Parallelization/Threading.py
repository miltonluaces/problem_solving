import threading
import time
import logging

# First multithreading snippet
def MThHelloWorld():

    def worker(num):
        print('Worker: ' , num)

    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

# Current thread
def DetermineCurrentThread():
    def worker():
        print(threading.currentThread().getName(), 'Starting')
        time.sleep(2)
        print(threading.currentThread().getName(), 'Exiting')

    def service():
        print(threading.currentThread().getName(), 'Starting')
        time.sleep(3)
        print(threading.currentThread().getName(), 'Exiting')

    serv = threading.Thread(name='service', target=service)
    work1 = threading.Thread(name='worker 1', target=worker)
    work2 = threading.Thread(name='worker 2', target=worker)

    work1.start()
    work2.start()
    serv.start()

# Logging instead of print to the output (thread-safe)
def Logging():
    logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] (%(threadName)-10s) %(message)s',)

    def worker():
        logging.debug('Starting')
        time.sleep(2)
        logging.debug('Exiting')

    def service():
        logging.debug('Starting')
        time.sleep(3)
        logging.debug('Exiting')

    serv = threading.Thread(name='service', target=service)
    work1 = threading.Thread(name='worker 1', target=worker)
    work2 = threading.Thread(name='worker 2', target=worker)

    work1.start()
    work2.start()
    serv.start()


# Thread subclasses
def SubClassing():
    logging.basicConfig(level=logging.DEBUG, format='(%(threadName)-10s) %(message)s',)

    class Worker(threading.Thread):
        def run(self):
            logging.debug('running')

    for i in range(5):
        t = Worker()
        t.start()

# Thread subclasses with parameters
def SubClassingWithArgs():
    logging.basicConfig(level=logging.DEBUG, format='(%(threadName)-10s) %(message)s',)

    class Worker(threading.Thread):
        def __init__(self, group=None, target=None, name=None, args=(), kwargs=None):
            threading.Thread.__init__(self, group=group, target=target, name=name)
            self.args = args
            self.kwargs = kwargs
        
        def run(self):
            logging.debug('running with %s and %s', self.args, self.kwargs)

    for i in range(5):
        t = Worker(args=(i,), kwargs={'a':'A', 'b':'B'})
        t.start()

#MThHelloWorld()
#DetermineCurrentThread()
#Logging()
#SubClassing()
SubClassingWithArgs()