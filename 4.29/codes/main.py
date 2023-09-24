import scipy.stats as stats
import numpy as np

# Parameters
n = 5  # Total number of persons
p_not_swimmer = 0.3  # Probability that a person is not a swimmer
p_swimmer = 1 - p_not_swimmer  # Probability that a person is a swimmer
k = 4  # Number of swimmers we want

# Calculate the expected value (mean) and standard deviation
mu = n * p_swimmer
sigma = np.sqrt(n * p_swimmer * (1 - p_swimmer))

# Calculate the probability using the Gaussian approximation
probability = stats.norm.cdf(k + 0.5, loc=mu, scale=sigma) - stats.norm.cdf(k - 0.5, loc=mu, scale=sigma)
print(f"The probability that 4 out of 5 persons are swimmers (Gaussian approximation) is {probability:.4f}")
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 5  # Total number of persons
p_not_swimmer = 0.3  # Probability that a person is not a swimmer
p_swimmer = 1 - p_not_swimmer  # Probability that a person is a swimmer
k = 4  # Number of swimmers we want

# Calculate the expected value (mean) and standard deviation
mu = n * p_swimmer
sigma = np.sqrt(n * p_swimmer * (1 - p_swimmer))

# Create an array of x values for the PDF graph
x = np.linspace(0, n, 1000)

# Calculate the PDF values for the x values
pdf_values = stats.norm.pdf(x, loc=mu, scale=sigma)

# Plot the PDF graph
plt.figure(figsize=(8, 4))
plt.plot(x, pdf_values, label='PDF')
plt.fill_between(x, 0, pdf_values, where=(x >= k-0.5) & (x <= k+0.5), alpha=0.3, label='Probability')
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import math

# Probability that a person is not a swimmer
p_not_swimmer = 0.3

# Probability that a person is a swimmer
p_swimmer = 1 - p_not_swimmer

# Number of persons
n = 5

# Create lists to store values of k and their corresponding probabilities
k_values = list(range(n + 1))
probabilities = []

# Calculate probabilities for all possible values of k
for k in k_values:
    binomial_coefficient = math.comb(n, k)
    probability = binomial_coefficient * (p_swimmer ** k) * (p_not_swimmer ** (n - k))
    probabilities.append(probability)

# Create the stem plot
plt.stem(k_values, probabilities, basefmt=' ')
plt.xlabel('Number of Swimmers (k)')
plt.ylabel('Probability')
plt.title('Stem Plot of Probability')

# Highlight k=4 with a different color
plt.stem([4], [probabilities[4]], linefmt='r-', markerfmt='ro', basefmt=' ')

plt.show()



