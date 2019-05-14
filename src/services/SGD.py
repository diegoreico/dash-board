
import numpy as np


class SGD:

    def __init__(self, data: np.ndarray, n_factors: int, alpha: np.double, n_epochs:int) -> None:
        super().__init__()

        self.data = data
        self.n_factors = n_factors
        self.alpha = alpha
        self.n_epochs = n_epochs

        self._u: np.ndarray = None
        self._v: np.ndarray = None

        self.is_training = False
        self.is_train = False

    def train(self):
        """Learn the vectors p_u and q_i with SGD.
           data is the user-item matrix
           n_factor is the number of latent factors to use
           alppha is the learning rate of the SGD
           n_epochs is the number of iterations to run the algorithm
        """
        self.is_training = True

        shape = np.shape(self.data)
        n_users = shape[0]
        n_items = shape[1]

        # Randomly initialize the user and item factors.
        p = np.random.normal(0, .1, (n_users, self.n_factors))
        q = np.random.normal(0, .1, (n_items, self.n_factors))

        # Optimization procedure
        for x in range(self.n_epochs):
            print("Current SVD epoch {}".format(x))
            for (u, i), r_ui in np.ndenumerate(self.data):
                if r_ui > 0:
                    err = r_ui - np.dot(p[u], q[i])
                    # Update vectors p_u and q_i
                    p[u] += self.alpha * err * q[i]
                    q[i] += self.alpha * err * p[u]

        self._u = p
        self._v = q

        self.is_training = False
        self.is_train = True

        return p, q

    def rmse(self, u: np.ndarray, v: np.ndarray) -> int:
        errors = u - v
        return np.sqrt(np.sum(errors * errors) / errors.size)

