import sim

# funciones del coppelia
def coppelia_connect(port):
    sim.simxFinish(-1) # just in case, close all opened connections
    clientID=sim.simxStart('127.0.0.1',port,True,True,2000,5) # Connect
    if clientID == -1: raise Exception("Error: could not connect to coppelia.")
    else: print("connected to", port)
    # if clientID == 0: print("connected to", port)
    # else: print("could not connect")
    return clientID