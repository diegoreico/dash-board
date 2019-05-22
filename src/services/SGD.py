import numpy as np


class SGD:

    def __init__(self, data: np.ndarray, n_factors: int, alpha: np.double, n_epochs: int) -> None:
        super().__init__()

        self.current_epoch = 0

        self.data_min = np.min(data)
        self.data_max = np.max(data)
        self.data = (data - self.data_min)/(self.data_max - self.data_min)

        self.n_factors = n_factors
        self.alpha = alpha
        self.n_epochs = n_epochs

        self._u: np.ndarray = None
        self._v: np.ndarray = None

        self.epoch_errors = []

        self.is_training = False
        self.is_train = False

    def train(self, n_factors: int, alpha: np.double, n_epochs: int):
        """Learn the vectors p_u and q_i with SGD.
           data is the user-item matrix
           n_factor is the number of latent factors to use
           alppha is the learning rate of the SGD
           n_epochs is the number of iterations to run the algorithm
        """

        self.n_factors = n_factors
        self.alpha = alpha
        self.n_epochs = n_epochs

        self.is_training = True

        shape = np.shape(self.data)
        n_users = shape[0]
        n_items = shape[1]

        # Randomly initialize the user and item factors.
        p = np.random.normal(0, 1, (n_users, self.n_factors)).astype(np.double)
        q = np.random.normal(0, 1, (n_items, self.n_factors)).astype(np.double)

        # Optimization procedure
        for x in range(self.n_epochs):
            self.current_epoch = x
            print('Current epoch: {}'.format(x))
            for (u, i), r_ui in np.ndenumerate(self.data):
                if r_ui > 0:
                    err = r_ui - np.dot(p[u], q[i])

                    # Update vectors p_u and q_i
                    p[u] += self.alpha * err * q[i]
                    q[i] += self.alpha * err * p[u]

            # updates matrices at the end of an epoch
            self._u = p
            self._v = q

            # obtain current error
            reconstructed_matrix = (p.dot(q.T) * (self.data_max - self.data_min)) + self.data_min
            error = self.rmse(self.data, reconstructed_matrix)
            self.epoch_errors.append(error)

        self.is_training = False
        self.is_train = True

        return p, q

    def rmse(self, u: np.ndarray, v: np.ndarray):
        errors = u - v
        return np.sqrt(np.sum(errors * errors) / errors.size)

    def predict(self, input) -> np.ndarray:
        users = self._u[input]
        predictions = users.dot(self._v.T)

        return predictions

    def obtain_group_recommendations(self, input: np.ndarray) -> (np.ndarray, np.ndarray):
        group_individual_recommendations = self.predict(input)

        least_misery_recommendations = np.amin(group_individual_recommendations, axis=0)
        least_misery_indexes = np.argsort(-least_misery_recommendations)

        return least_misery_recommendations, least_misery_indexes
