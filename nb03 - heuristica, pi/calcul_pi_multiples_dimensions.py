from multiprocessing import Pool
from multiprocessing import Process
import multiprocessing as mp
import numpy as np
import os

def myfunc(name):
    print('module %s, ppid %s, Hello %s from pid %s  ' %(__name__, os.getppid(), name, os.getpid()))

def ex_1():
    p = Process(target = myfunc, args = ('CEIABD',))
    p.start()
    p.join()

def ex_2():
    mypool = Pool(processes = 8)
    mypool.map(myfunc, ['Gaurav', 'Abel', 'Grigor', 'Jaume', 'Oriol', 'Aniol', 'Miquel', 'Joan'])
    mypool.close()
    mypool.terminate()

def pi2d_(params):
    n_in = 0
    iters, seed = params
    np.random.seed(seed)
    for _ in range(iters):
        if np.sqrt(np.sum(np.random.rand(2)**2)) <= 1:
            n_in += 1

    return (4 *n_in/iters)

def pi3d_(params):
    n_in = 0
    iters, seed = params
    np.random.seed(seed)

    for _ in range(iters):
        if np.sqrt(np.sum(np.random.rand(3)**2)) <= 1:
            n_in += 1
    return (6 *n_in/iters)

def pi4d_(params):
    n_in = 0
    iters, seed = params
    np.random.seed(seed)

    for _ in range(iters):
        if np.sqrt(np.sum(np.random.rand(4)**2)) <= 1:
            n_in += 1
    return (np.sqrt(32 *n_in/iters))

def pi5d_(params):
    n_in = 0
    iters, seed = params
    np.random.seed(seed)

    for _ in range(iters):
        if np.sqrt(np.sum(np.random.rand(5)**2)) <= 1:
            n_in += 1
    return ((60 *n_in/iters)**(1/2))

def pi6d_(params):
    n_in = 0
    iters, seed = params
    np.random.seed(seed)

    for _ in range(iters):
        if np.sqrt(np.sum(np.random.rand(6)**2)) <= 1:
            n_in += 1
    return ((384 *n_in/iters)**(1/3))

def pi10d_(params):
    n_in = 0
    iters, seed = params
    np.random.seed(seed)

    for _ in range(iters):
        if np.sqrt(np.sum(np.random.rand(10)**2)) <= 1:
            n_in += 1
    return (4*((120*(n_in/iters))**(1/5)))

def pi15d_(params):
    n_in = 0
    iters, seed = params
    np.random.seed(seed)

    for _ in range(iters):
        if np.sqrt(np.sum(np.random.rand(15)**2)) <= 1:
            n_in += 1
    return (2*(2027025*(n_in/iters))**(1/7))

def calc_pi(dardos, threads, dimension, function): 
    with Pool(processes = mp.cpu_count()) as pool:
        outQ = pool.map(function, [(dardos, r) for r in np.random.randint(1, 1000, threads)])
        # Mitjana
        print(f"mean {dimension}: ", np.mean(outQ))
        # Variança
        print(f"var {dimension}: ", np.var(outQ))

if __name__ == '__main__':
    dardos = 2000
    threads = 100

    print("2D:")
    calc_pi(dardos, threads, "2D", pi2d_)

    print("")
    print("3D:")
    calc_pi(dardos, threads, "3D", pi3d_)

    print("")
    print("4D:")
    calc_pi(dardos, threads, "4D", pi4d_)

    print("")
    print("5D:")
    calc_pi(dardos, threads, "5D", pi5d_)

    print("")
    print("6D:")
    calc_pi(dardos, threads, "6D", pi6d_)

    print("")
    print("10D:")
    calc_pi(dardos, threads, "10D", pi10d_)

    print("")
    print("15D:")
    calc_pi(dardos, threads, "15D", pi15d_)