import numpy as np
import matplotlib.pyplot as plt

from itertools import product
import sys
import unittest

sys.path.append("..")
from twodos.twodos_api import TwoDos

a = 1
t = 3
avecs = np.array([[np.sqrt(3) * a / 2, -a / 2], [np.sqrt(3) * a / 2, a / 2]])
bvecs = 2 * np.pi / a * np.array([[np.sqrt(3) / 3, -1], [np.sqrt(3) / 3, 1]])

k1 = -1 / 3 * bvecs[0] + 1 / 3 * bvecs[1]
k2 = 1 / 3 * bvecs[0] - 1 / 3 * bvecs[1]
m = bvecs[1] / 2
gamma = np.array([0, 0])

graphene_high_symm_ks = {'k1': k1, 'gamma': gamma, 'k1': k1, 'm': m}

print(
    np.dot(avecs[0], bvecs[0]) / (2 * np.pi),
    np.dot(avecs[1], bvecs[1]) / (2 * np.pi))
print(np.dot(avecs[0], bvecs[1]), np.dot(avecs[1], bvecs[0]))

delta1 = a * np.array([1 / np.sqrt(3), 0])
delta2 = a * np.array([-1 / (2 * np.sqrt(3)), 1 / 2])
delta3 = a * np.array([-1 / (2 * np.sqrt(3)), -1 / 2])
delta_list = [delta1, delta2, delta3]
print(np.linalg.norm(delta1), np.linalg.norm(delta2), np.linalg.norm(delta3),
      a / np.sqrt(3))


def set_tb_disp_kmesh(n_k: int, high_symm_pnts: dict) -> tuple:
    """setup kpath along high symmetry points in moire B.Z.
    Note that, the length of each path is not equal, so it is not a 
    normal sampling. The Kpath is K1 - Gamma - M - K2.
    Args:
        n_k (int): number of kpts on one path
        high_symm_pnts (dict): hcoordinates of high symmetry points
    Returns:
        tuple: (kline, kmesh)
    """

    num_sec = 4
    num_kpt = n_k * (num_sec - 1)
    length = 0

    klen = np.zeros((num_sec), float)
    ksec = np.zeros((num_sec, 2), float)
    kline = np.zeros((num_kpt + 1), float)
    kmesh = np.zeros((num_kpt + 1, 2), float)

    # set k path (K1 - Gamma - M - K2)
    ksec[0] = high_symm_pnts['gamma']
    ksec[1] = high_symm_pnts['m']
    ksec[2] = high_symm_pnts['k1']
    ksec[3] = high_symm_pnts['gamma']

    for i in range(num_sec - 1):
        vec = ksec[i + 1] - ksec[i]
        length = np.sqrt(np.dot(vec, vec))
        klen[i + 1] = klen[i] + length

        for ikpt in range(n_k):
            kline[ikpt + i * n_k] = klen[i] + ikpt * length / n_k
            kmesh[ikpt + i * n_k] = ksec[i] + ikpt * vec / n_k
    kline[num_kpt] = kline[2 * n_k] + length
    kmesh[num_kpt] = ksec[3]

    return (kline, kmesh)


def set_kmesh_dos(n_k: int, bvecs: np.ndarray) -> np.ndarray:

    m_g_unitvec_1 = bvecs[0]
    m_g_unitvec_2 = bvecs[1]
    k_step = 1 / (n_k - 1)
    kmesh = [
        i * k_step * m_g_unitvec_1 + j * k_step * m_g_unitvec_2
        for (i, j) in product(range(n_k), range(n_k))
    ]

    return np.array(kmesh)


def make_hamk(kvec: np.ndarray):

    hamk = np.zeros((2, 2), dtype='complex')
    kdelta = np.array([np.dot(kvec, delta) for delta in delta_list])
    #print(type(kdelta), kdelta)
    hamk[0, 1] = t * np.sum(np.exp(1j * kdelta))
    hamk[1, 0] = t * np.sum(np.exp(-1j * kdelta))

    #print(hamk, kvec)
    return hamk



class graphene_test(unittest.TestCase):

    def test_graphene_band(self):
        kline, kmesh = set_tb_disp_kmesh(30, graphene_high_symm_ks)
        emesh = []
        for kvec in kmesh:
            hamk = make_hamk(kvec)
            v, w = np.linalg.eigh(hamk)
            emesh.append(v)
        emesh = np.array(emesh)
        plt.plot(kline, emesh[:,0])
        plt.plot(kline, emesh[:,1])
        plt.savefig("graphene_band.png")
        plt.close()

    def test_dos_result(self):
        n_k = 100
        num_e = 300
        kmesh = set_kmesh_dos(n_k, bvecs)
        emesh = []
        for kvec in kmesh:
            hamk = make_hamk(kvec)
            v, _ = np.linalg.eigh(hamk)
            emesh.append(v)

        kmesh = kmesh.reshape((n_k, n_k, 2))
        emesh = np.array(emesh).reshape((n_k, n_k, 2))
        graphene_dos = TwoDos(n_k, bvecs, num_e)
        graphene_dos.set_kmesh(kmesh)
        graphene_dos.set_emesh(emesh)
        graphene_dos.solver()
        graphene_dos.dos_smear_gaussian_solver()
        dos_integral = graphene_dos.dos_check()
        graphene_dos.plot()
        plt.savefig("graphene_dos.png")
        self.assertTrue(1.95 < dos_integral < 2.05)
        plt.close()
