import random

import numpy as np
import math
import cv2
from tqdm import trange

from blur_the_image import gen_blur_matrix, image_shape
import matplotlib.pyplot as plt

def read_blurred_img(imgname):
    img = cv2.imread("data/blurred_{}.png".format(imgname), 2)
    return img


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def nabla_f(x_t, X, Y):
    return X @ (x_t.T @ X - Y).T


def PG_Solver_for_LASSO(x_t, nabla_f, L, X, Y, reg_lambda):
    '''
    Solving the minimizer for PG when $g(x) = \|x\|_1$ is 1-norm, i.e.:
    F(x) = f(x) + g(x)
         = 1/2 * \|x_t^\Top @ X - Y\|_2^2 + reg_lambda * \|x\|_1

    :param nabla_f: function, nabla_f(x) is the gradient of f(x) w.r.t. x
    :param L: float, L-smooth
    :param X, Y: parameters in F(x)
    :param reg_lambda: regularizer weight of g(x)
    :return: optimize one_step for x_{t+1}
    '''
    # x_t (25600, 1)
    # TODO: Finish the code here
    x_tmp = x_t - 1 / L * nabla_f(x_t, X, Y)
    x_new = np.sign(x_tmp) * (np.abs(x_tmp) - reg_lambda / L)
    return x_new # (25600, 1)


class ISTA:
    def __init__(self, PG_Solver, X, Y):
        self.name = 'ISTA/PG Algorithm'
        self.PG_Solver = PG_Solver

        self.X = X # 模糊矩阵
        self.Y = Y # 原始图片

        self.x = self.Y.T  # initialize x_0 # 原始图片
        # self.x = np.random.randn(256, 1)  # initialize x_0
        self.err_list = []

    def one_step(self, L=1024.0):
        # ISTA Update:
        self.x = self.PG_Solver(x_t=self.x, nabla_f=nabla_f, L=L, X=self.X, Y=self.Y, reg_lambda=2e-5)

        # record the error at this iteration
        self.err_list.append(np.linalg.norm(self.x.T @ self.X - self.Y))

        # return result
        return self.x
        # raise NotImplementedError


class FISTA:
    def __init__(self, PG_Solver, X, Y):
        self.name = 'FISTA/APG Algorithm'
        self.PG_Solver = PG_Solver

        self.X = X
        self.Y = Y

        self.x = self.Y.T  # initialize x_0
        # self.x = np.random.randn(256, 1)   # initialize x_0
        self.y = self.x
        self.alpha = None
        self.lambda_ = 1
        self.err_list = []

    def one_step(self, L=1024.0):
        # FISTA Update:
        # %%
        # TODO: Finish the code here
        self.new_lambda_ = (1 + np.sqrt(1 + 4 * (self.lambda_ ** 2))) / 2
        self.alpha = (1 - self.lambda_) / self.new_lambda_
        self.lambda_ = self.new_lambda_

        self.y_new = self.PG_Solver(x_t=self.x, nabla_f=nabla_f, L=L, X=self.X, Y=self.Y, reg_lambda=2e-5)
        self.x = (1 - self.alpha) * self.y_new + self.alpha * self.y
        self.y = self.y_new
        # %%

        # record the error at this iteration
        self.err_list.append(np.linalg.norm(self.x.T @ self.X - self.Y))

        # return result
        return self.x


def save_img(img, imgname, name, idx):
    img = img.reshape((image_shape, image_shape))
    img = img / np.max(img.flatten())
    img = img * 256.0
    cv2.imwrite('output_imgs/{}_{}_{}.png'.format(imgname, name, idx), img)
    return 0


if __name__ == '__main__':

    R = gen_blur_matrix(image_shape ** 2, radius=3)
    set_seed(2)

    imgname = 'eason'
    # imgname = 'cameraman'

    img = read_blurred_img(imgname)
    img = img.flatten()
    img = img.reshape((1, image_shape ** 2))
    img = img / np.max(img.flatten())

    # %%
    ################################################################
    # Example Code for using ISTA for image deblur
    '''
    :param X: X = R, where R is the matrix representing the blur operator.
    :param Y: the (vectorized) observed blurred image
    '''

    X = R # 模糊矩阵
    Y = img # 模糊图片
    # print(R.shape, img.shape) # (25600, 25600) (1, 25600)

    ista = ISTA(PG_Solver_for_LASSO, X, Y)
    fista = FISTA(PG_Solver_for_LASSO, X, Y)
    T = 21

    for i in trange(T):
        x_fista = fista.one_step(L=image_shape * 2)  # We observe that there the F-norm of (X^\Top @ X) is roughly (2 * image_shape)
        x_ista = ista.one_step(L=image_shape * 2)
        if i % 1 == 0:
            save_img(x_fista, imgname, 'fista', i)
            save_img(x_ista, imgname, 'ista', i)
        print(i, '\tISTA:', ista.err_list[-1], '\tFISTA:', fista.err_list[-1])

    re_ista = np.array(ista.err_list)
    re_fista = np.array(fista.err_list)
    # print('ISTA', re_ista)
    # print('FISTA', re_fista)
    # re_diff = re_ista - re_fista
    # print('diff:', re_diff)
    print('Finished')

    # %%
    # TODO: Compare convergence rate between ISTA and FISTA (Plot)
    plt.figure()
    plt.plot(re_ista, label="ista")
    plt.plot(re_fista, label="fista")
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.legend()
    plt.title('the convergence rate between ISTA and FISTA')
    plt.savefig('conv-rate.pdf')
    # %%
