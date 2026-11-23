"""Two probability models implemented with the Python standard library."""

import math
from statistics import fmean, pvariance


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1 + exponent)


class LogisticRegression:
    def fit(self, customers, steps=350, rate=0.35, penalty=0.01):
        columns = list(zip(*(row.features for row in customers)))
        self.means = [fmean(column) for column in columns]
        self.scales = [math.sqrt(pvariance(column)) or 1.0 for column in columns]
        standardized = [(self.transform(row.features), row.churned) for row in customers]
        self.weights = [0.0] * len(columns)
        self.intercept = 0.0
        for _ in range(steps):
            gradients = [0.0] * len(columns)
            intercept_gradient = 0.0
            for values, target in standardized:
                residual = sigmoid(self.linear(values)) - target
                intercept_gradient += residual
                for i, value in enumerate(values):
                    gradients[i] += residual * value
            self.intercept -= rate * intercept_gradient / len(standardized)
            self.weights = [weight - rate * (gradient / len(standardized) + penalty * weight)
                            for weight, gradient in zip(self.weights, gradients)]
        return self

    def transform(self, values):
        return [(value - mean) / scale for value, mean, scale in zip(values, self.means, self.scales)]

    def linear(self, standardized):
        return self.intercept + sum(weight * value for weight, value in zip(self.weights, standardized))

    def predict(self, values):
        return sigmoid(self.linear(self.transform(values)))


class NaiveBayes:
    """Use Gaussian likelihoods for numeric inputs and Bernoulli for flags."""

    def __init__(self, binary_indices):
        self.binary_indices = frozenset(binary_indices)

    def fit(self, customers):
        self.groups = []
        for label in (0, 1):
            group = [row.features for row in customers if row.churned == label]
            likelihoods = []
            for i, values in enumerate(zip(*group)):
                if i in self.binary_indices:
                    likelihoods.append(((sum(values) + 1) / (len(values) + 2), 0.0))
                else:
                    likelihoods.append((fmean(values), max(pvariance(values), 1e-6)))
            self.groups.append((math.log(len(group) / len(customers)), likelihoods))
        return self

    def predict(self, values):
        scores = []
        for log_prior, likelihoods in self.groups:
            log_score = log_prior
            for i, (value, (mean, variance)) in enumerate(zip(values, likelihoods)):
                if i in self.binary_indices:
                    log_score += math.log(mean if value else 1 - mean)
                else:
                    log_score -= 0.5 * (math.log(2 * math.pi * variance) + (value - mean) ** 2 / variance)
            scores.append(log_score)
        return sigmoid(scores[1] - scores[0])
