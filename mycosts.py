from tvopt import costs, sets
import numpy as np

class TwoExpCost(costs.Cost):
    def __init__(self, a, b, c, d):
        super().__init__(sets.R())
        self.smooth = 2
        self.a, self.b, self.c, self.d = np.array(a).item(), np.array(b).item(), np.array(c).item(), np.array(d).item()

    def function(self, x):
        return np.array(self.c * np.exp(self.a * x) + self.d * np.exp(-self.b * x)).item()

    def gradient(self, x):
        return self.c * self.a * np.exp(self.a * x) - self.d * self.b* np.exp(-self.b * x)

    def hessian(self, x):
        return self.c * self.a**2 * np.exp(self.a * x) + self.d * self.b**2 * np.exp(self.b * x)
