class Neuron:
    def __init__(self):
        self.w = 1
        self.b = 1
    
    def forpass(self, x):
        y_hat = x * self.w + self.b
        return y_hat

    def backprop(self, x, err):
        w_grad = x * arr
        b_grad = 1 * arr
        return w_grad, b_grad

    def fit(self, x, y, epochs=100):
        for i in range(epochs):
            for x_i, y_i in zip(x, y):
                y_hat = self.forpass(x_i)
                err = -(y_i - y_hat)
                w_grad, b_grad = self.backprop(x_i, err)
                self.w -= w_grad
                self.b -= b_grad
