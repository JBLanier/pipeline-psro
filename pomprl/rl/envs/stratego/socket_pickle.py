import socket
import pickle

"""
Send and receive arbitrary python objects with sockets without having to worry about transport protocol

https://pythonprogramming.net/pickle-objects-sockets-tutorial-python-3/
"""

HEADER_SIZE = 10
import time

def listen_and_accept_connection(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.listen(5)
    clientsocket, address = s.accept()
    return clientsocket


def get_connection(host_addr, port, retries=5):
    s = None
    while retries > 0:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host_addr, port))
            break
        except ConnectionRefusedError:
            retries -= 1
            time.sleep(0.1)

    if s is None:
        raise ConnectionRefusedError

    return s


def pickle_recv(s):
    full_msg = b''
    new_msg = True
    while True:
        msg = s.recv(16)
        if new_msg:
            # print("new msg len:", msg[:HEADERSIZE])
            msglen = int(msg[:HEADER_SIZE])
            new_msg = False

        # print(f"full message length: {msglen}")

        full_msg += msg

        # print(len(full_msg))

        if len(full_msg) - HEADER_SIZE == msglen:
            break
    # print("full msg recvd")
    # print(full_msg[HEADERSIZE:])
    # print(pickle.loads(full_msg[HEADERSIZE:]))
    return pickle.loads(full_msg[HEADER_SIZE:])


def pickle_send(s, data):
    msg = pickle.dumps(data)
    msg = bytes(f"{len(msg):<{HEADER_SIZE}}", 'utf-8') + msg
    s.send(msg)


if __name__ == '__main__':
    import numpy as np

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((socket.gethostname(), 1243))
    s.listen(5)


    sc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sc.connect((socket.gethostname(), 1243))

    clientsocket, address = s.accept()

    pickle_send(clientsocket, {'hey': np.random.uniform(size=(10,10,2))})
    print(pickle_recv(sc))