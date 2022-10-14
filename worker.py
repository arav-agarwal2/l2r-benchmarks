import socket
from distrib_l2r.asynchron.worker import AsnycWorker
#from src.utils.envwrapper_aicrowd import EnvContainer

learner_ip = socket.gethostbyname('learner-service')
learner_address = (learner_ip, 4444)



if __name__ == '__main__':
    worker = AsnycWorker(learner_address=learner_address)
    print("Worker inited!!!")
    worker.work()
