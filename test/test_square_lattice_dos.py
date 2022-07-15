import numpy as np
import sys
import unittest
import matplotlib.pyplot as plt
sys.path.append("..")
from twodos.twodos_api import TwoDos

a = 2.46
bvec = np.array([[(2 * np.pi) / a, 0], [0, (2 * np.pi) / a]])
avec = np.array([[a, 0], [0, a]])
t = 1
num_k = 300
num_e = 300
E_low = -2.1 * 2 * t
E_high = 2.1 * 2 * t
step = 1.0 / (num_k - 1)
Edisp = np.zeros((num_k, num_k))
kvec = np.zeros((num_k, num_k, 2))
kmesh = np.array([
    i * step * bvec[0] + j * step * bvec[1] for i in range(num_k)
    for j in range(num_k)
])
kmesh = kmesh.reshape((num_k, num_k, 2))
for ik in range(num_k):
    for jk in range(num_k):
        b1 = ik * step * bvec[0] + jk * step * bvec[1]
        kvec[ik][jk] = b1
        x = np.dot(avec[0], kvec[ik][jk])
        y = np.dot(avec[1], kvec[ik][jk])
        Edisp[ik][jk] = 2 * t * (np.cos(x) + np.cos(y))

Edisp = Edisp.reshape((num_k, num_k, 1))


class square_lattice_test(unittest.TestCase):
    def test_dos_result(self):
        myslover = TwoDos(num_k, bvec, num_e)
        myslover.set_emesh(Edisp)
        myslover.set_kmesh(kvec)
        myslover.solver()
        dos_integral = myslover.dos_check()
        myslover.plot()
        plt.savefig("./square_lattice_dos.png")
        plt.close()
        self.assertTrue(0.95 < dos_integral < 1.05)
