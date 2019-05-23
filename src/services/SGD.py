import numpy as np

np.random.seed(1234)


class SGD:

    def __init__(self, data: np.ndarray, n_factors: int, learning_rate: np.double, n_epochs: int,
                 l2reg: float = 0.2) -> None:
        super().__init__()

        self.current_epoch = 0

        self.data = data
        shape = np.shape(self.data)
        self._n_users = shape[0]
        self._n_items = shape[1]
        self.data_min = np.min(data)
        self.data_max = np.max(data)

        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self._l2_reg = l2reg

        # hides some values in matrix to validate recommendation method
        ###
        positive_values = np.where(self.data > 0)
        test_split_size = round(0.2 * len(positive_values[0]))
        test_split_positive_values_idx = np.random.randint(0, len(positive_values[0]), test_split_size)
        test_split_selected_values = list(zip(
            positive_values[0][test_split_positive_values_idx],
            positive_values[1][test_split_positive_values_idx]
        ))

        self.test_data = []

        for x, y in test_split_selected_values:
            self.test_data.append((x, y, self.data[x][y]))
            self.data[x][y] = 0
        ###

        # normalizes matrix values between 0 and 1
        self.data = (data - self.data_min) / (self.data_max - self.data_min)

        # Randomly initialize the user and item factors
        self._p = np.random.normal(0, 1. / self.n_factors, (self._n_users, self.n_factors))
        self._q = np.random.normal(0, 1. / self.n_factors, (self._n_items, self.n_factors))

        self._p_bias = np.zeros(self._n_users)
        self._q_bias = np.zeros(self._n_items)
        self._global_bias = np.mean(self.data[np.where(self.data != 0)])

        self.epoch_errors = []
        self.epoch_test_errors = []

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

            # obtain current test error
            test_error = 0.0
            for x, y, vxy in self.test_data:
                partial_test_error = scaled_matrix[x, y] - vxy
                test_error += partial_test_error**2

            self.epoch_test_errors.append(test_error/len(self.test_data))

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
