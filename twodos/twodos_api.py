import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from itertools import product


@jit(nopython=True)
def _triangle_integral_dos(tri_kpnts: np.ndarray,
                           tri_edisp: np.ndarray) -> float:
    """calculate the density of states analytically for a small triangle

    Args:
        tri_kpnts (np.ndarray): three kpoints, shape (3, 2)
        tri_edisp (np.ndarray): three edispertion, shape (3, )
    
    Return:
        (float): dos caculated in a small triangle
    """

    # sgns for energies
    sgn = [1 if tri_edisp[i] > 0 else -1 for i in range(3)]
    # if all sgns equal, contribute 0 to dos
    if sgn[0] == sgn[1] == sgn[2]:
        return 0

    # find the different sgn, and order the kpoints
    diff_sgn = sgn[0] * sgn[1] * sgn[2]
    for i in range(3):
        if sgn[i] == diff_sgn:
            p1 = i
            p2 = (i + 1) % 3
            p3 = (i + 2) % 3

    ratio_12 = tri_edisp[p1] / (tri_edisp[p1] - tri_edisp[p2])
    ratio_13 = tri_edisp[p1] / (tri_edisp[p1] - tri_edisp[p3])

    projected_k12 = tri_kpnts[p1] + ratio_12 * (tri_kpnts[p2] - tri_kpnts[p1])
    projected_k13 = tri_kpnts[p1] + ratio_13 * (tri_kpnts[p3] - tri_kpnts[p1])

    projected_k23 = projected_k12 - projected_k13
    klength = np.sqrt(projected_k23[0]**2 + projected_k23[1]**2)

    # make 3D points
    kp1 = np.array([tri_kpnts[p1][0], tri_kpnts[p1][1], tri_edisp[p1]])
    kp2 = np.array([tri_kpnts[p2][0], tri_kpnts[p2][1], tri_edisp[p2]])
    kp3 = np.array([tri_kpnts[p3][0], tri_kpnts[p3][1], tri_edisp[p3]])

    vec_12 = kp2 - kp1
    vec_13 = kp3 - kp1
    # calculate |dE/dk|
    cross_vec = np.cross(vec_12, vec_13)
    jacobian = np.sqrt(cross_vec[0]**2 + cross_vec[1]**2) / np.abs(
        cross_vec[2])

    res = klength / jacobian
    #print('klength', klength)
    return res


@jit(nopython=True, parallel=True)
def _triangle_dos_solver(num_k: int, karea: float, eline: np.ndarray,
                         kmesh: np.ndarray, emesh: np.ndarray) -> np.ndarray:
    """calculate dos v.s. eline

    Args:
        num_k (int): num of kpoints along one kvector
        karea (float): k space area
        eline (np.ndarray): energy line
        kmesh (np.ndarray): kemsh, shape(num_k, num_k, 3)
        emesh (np.ndarray): emehs for one band, shape(num_k, num_k)

    Returns:
        np.ndarray: edos
    """

    tri_kpnts = np.zeros((3, 2))
    tri_edisp = np.zeros((3))
    edos = []
    count = 1
    count_total = eline.shape[0]

    for enow in eline:
        res = 0
        print(">>> count", count, "of total:", count_total)
        count += 1
        for ik in range(num_k - 1):
            for jk in range(num_k - 1):
                tri_kpnts[0] = kmesh[ik][jk]
                tri_kpnts[1] = kmesh[ik + 1][jk]
                tri_kpnts[2] = kmesh[ik][jk + 1]
                tri_edisp[0] = emesh[ik][jk] - enow
                tri_edisp[1] = emesh[ik + 1][jk] - enow
                tri_edisp[2] = emesh[ik][jk + 1] - enow

                res += _triangle_integral_dos(tri_kpnts, tri_edisp)

                tri_kpnts[0] = kmesh[ik + 1][jk + 1]
                tri_kpnts[1] = kmesh[ik + 1][jk]
                tri_kpnts[2] = kmesh[ik][jk + 1]
                tri_edisp[0] = emesh[ik + 1][jk + 1] - enow
                tri_edisp[1] = emesh[ik + 1][jk] - enow
                tri_edisp[2] = emesh[ik][jk + 1] - enow

                res += _triangle_integral_dos(tri_kpnts, tri_edisp)
        #print('res', res)
        edos.append(res / karea)

    return np.array(edos)


class TwoDos():
    def __init__(self, num_k: int, bvecs: np.ndarray, num_e: int):

        self.num_k = num_k
        self.num_e = num_e
        self.b_vec_1 = bvecs[0]
        self.b_vec_2 = bvecs[1]
        self.karea = self._set_karea()

        self.kmesh = None
        self.emesh = None
        self.edos = None
        self.eline = None
        self.dos_integral = None

    def _set_karea(self):

        karea = np.linalg.norm(np.cross(self.b_vec_1, self.b_vec_2))

        return karea

    def set_kmesh(self, kmesh: np.ndarray):

        kstep = 1 / (self.num_k - 1)
        _kmesh = np.array([
            i * kstep * self.b_vec_1 + j * kstep * self.b_vec_2
            for (i, j) in product(range(self.num_k), range(self.num_k))
        ])
        _kmesh = _kmesh.reshape((self.num_k, self.num_k, 2))

        assert np.allclose(_kmesh, kmesh)
        self.kmesh = kmesh

    def set_emesh(self, emesh: np.ndarray):

        self.emesh = emesh
        assert self.emesh.shape[0] == self.emesh.shape[1] == self.num_k

        self.elow = np.min(emesh) - 1
        self.ehigh = np.max(emesh) + 1
        self.delta_e = (self.ehigh - self.elow) / self.num_e
        self.eline = np.array(
            [self.elow + i * self.delta_e for i in range(self.num_e)])
        self.num_bands = emesh.shape[2]

    def solver(self):

        if self.kmesh is None or self.emesh is None:
            raise Exception("Set kmesh and emesh first.")

        self.edos = np.zeros(self.eline.shape)

        for band in range(self.num_bands):
            print(">>> slove for band [%i]..." % (band + 1))
            self.edos += _triangle_dos_solver(self.num_k, self.karea,
                                              self.eline, self.kmesh,
                                              self.emesh[:, :, band])

        print(">>> DOS Integral Finished.")

    def plot(self):

        if self.edos is None or self.eline is None:
            raise Exception("Set kmesh and emesh and solve dos first.")

        plt.plot(self.eline, self.edos)
        plt.grid()

    def dos_check(self):

        self.dos_integral = np.sum(self.edos * self.delta_e)

        return self.dos_integral

    def print_info(self):

        print(">>> elow  setted as:", self.elow)
        print(">>> ehigh setted as:", self.ehigh)
        print(">>> del e setted as:", self.delta_e)
        print(">>> num of bands   :", self.num_bands)
