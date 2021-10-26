

class Adam:

    def calculate_gradients(self, input_dz, previous_weights):

        weights_arr = np.array(self.weights)

        if previous_weights is not None:
            dz = previous_weights.T.dot(input_dz) * self.activation.backward()
        else:
            dz = input_dz

        dw = 1 / self.n_inputs * dz.dot(weights_arr.T)
        db = 1 / self.n_inputs * np.sum(dz)

        return {
            'dw': dw,
            'db': db
        }
