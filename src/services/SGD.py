import numpy as np

np.random.seed(1234)

class SGD:

    def __init__(self, data: np.ndarray, n_factors: int, learning_rate: np.double, n_epochs: int, l2reg: float = 0.2) -> None:
        super().__init__()

        self.current_epoch = 0

        self.data_min = np.min(data)
        self.data_max = np.max(data)
        self.data = (data - self.data_min) / (self.data_max - self.data_min)
        # self.data = data

        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self._l2_reg = l2reg

        shape = np.shape(self.data)
        self._n_users = shape[0]
        self._n_items = shape[1]

        # Randomly initialize the user and item factors
        self._p = np.random.normal(0, 1. / self.n_factors, (self._n_users, self.n_factors))
        self._q = np.random.normal(0, 1. / self.n_factors, (self._n_items, self.n_factors))

        self._p_bias = np.zeros(self._n_users)
        self._q_bias = np.zeros(self._n_items)
        self._global_bias = np.mean(self.data[np.where(self.data != 0)])

        self.epoch_errors = []

        self.is_training = False
        self.is_train = False

    def train(self, n_factors: int, learning_rate: np.double, n_epochs: int, bias_reg: float, l2_reg: float):

        np.seterr(all='raise')
        self.epoch_errors = []

        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self._l2_reg = l2_reg

        self.is_training = True

        user_bias_reg = bias_reg
        item_bias_reg = bias_reg

        # Optimization procedure
        for x in range(self.n_epochs):
            self.current_epoch = x
            print('Current epoch: {}'.format(x))
            for (u, i), r_ui in np.ndenumerate(self.data):
                if r_ui > 0:
                    # obtain current error
                    prediction = self.predict(u, i)
                    error = (r_ui - prediction)

                    # update bias
                    self._p_bias[u] += self.learning_rate * (error - user_bias_reg * self._p_bias[u])
                    self._q_bias[i] += self.learning_rate * (error - item_bias_reg * self._q_bias[i])

                    # Update latent factors
                    self._p[u] += self.learning_rate * (error * self._q[i] - self._l2_reg * self._p[u])
                    self._q[i] += self.learning_rate * (error * self._p[u] - self._l2_reg * self._q[i])

            # obtain current error
            reconstructed_matrix = self.predict_all(list(range(self._n_users)))
            scaled_matrix = reconstructed_matrix * (self.data_max - self.data_min) + self.data_min
            error = self.rmse(self.data, scaled_matrix)
            # error = self.rmse(self.data, reconstructed_matrix)
            self.epoch_errors.append(error)

        self.is_training = False
        self.is_train = True

    def rmse(self, u: np.ndarray, v: np.ndarray):
        errors = u - v
        return np.sqrt(np.sum(errors * errors) / errors.size)

    def predict(self, u, i) -> np.ndarray:
        prediction = self._global_bias + self._p_bias[u] + self._q_bias[i]
        prediction += self._p[u].dot(self._q[i].T)

        # scales back the prediction
        prediction = prediction * (self.data_max - self.data_min) + self.data_min

        return prediction

    def predict_all(self, rows):
        predictions = np.zeros((len(rows), self._n_items))
        for u in range(len(rows)):
            for i in range(self._n_items):
                predictions[u, i] = self.predict(rows[u], i)

        return predictions

    def obtain_group_recommendations(self, input: np.ndarray) -> (np.ndarray, np.ndarray):
        group_individual_recommendations = self.predict_all(input)

        least_misery_recommendations = np.amin(group_individual_recommendations, axis=0)
        least_misery_indexes = np.argsort(-least_misery_recommendations)

        return least_misery_recommendations, least_misery_indexes
