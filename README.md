# Hyperparameter-Optimization-Tree-Structured-Parzen-Estimator

## Steps of Implementation : 

1. Define the optimization problem
The initial step is to define the optimization problem, this is the objective function to be optimized and the hyperparameters to be fine-tuned. In this step, the objective function is usually a performance evaluation metric such as accuracy for classification problems or mean absolute error for regression problems. The aim of the optimization is to find the optimal hyperparameter values that maximize or minimize the objective function.

2. Define the hyperparameter search space
The search space refers to the possible range of values that each defined hyperparameter can take at any given time. These hyperparameters either have integer values, float values, boolean values, or string values. It is essential to select a search space that is broad enough to incorporate potentially good combinations capable of producing the best performance but not so broad that the search becomes inefficient.

Some examples of the hyperparameters used for the KNN algorithm available on the scikit-learn library (KNeighborsClassifier()) are stated below:

n_neighbors: This is the number of neighbors to use.

weights: This is the weight function used in prediction.

algorithm: This is the algorithm used to compute the nearest neighbors.

leaf_size: This is the leaf size passed to BallTree or KDTree.

3. Build the surrogate models
TPE maintains two surrogate models, one for good search space and one for bad search space. These surrogate models are often represented as kernel density estimators, which are statistical tools used to estimate the probability density function of a random variable.

First surrogate model:

TPE builds the first surrogate model to predict the probability distribution of hyperparameters being in the good space, given the data observed so far.

Second surrogate model: 

TPE builds another surrogate model to predict the probability distribution of hyperparameters being in the bad space, given the data observed so far.

4. Apply the acquisition function
TPE uses these surrogate models to sample new hyperparameters for evaluation. It utilizes an acquisition function, also known as a selection function, which guides the search to select the next combination of hyperparameters that are more likely to result in better performance.

This helps in balancing between exploration and exploitation by evaluating the ratio of the first surrogate model’s (good space) probability and the second surrogate model’s (bad space) probability.

5. Select the next combination to evaluate
The acquisition function is used to select the next combination of hyperparameters to evaluate. This selection is based on the probabilities of both surrogate models. This combination is typically chosen to maximize the performance over the current best combination of hyperparameters.

6. Evaluate the objective function
The selected combination of hyperparameters is used to train an ML model (KNN) on a subset of the data or to cross-validate data. Then, the objective function value is computed for the ML model’s performance on the validation set. This value represents how well the ML model is doing with the current set of hyperparameters.

7. Update the surrogate models
If the objective function value is better than the current best result, TPE updates the best result and stores the new hyperparameters as a candidate. Then TPE classifies the sampled hyperparameters as either belonging to the good space or the bad space based on a predefined threshold or criteria.
Finally, both the surrogate models are updated based on the classified hyperparameters. This step fine-tunes the surrogate models and focuses the search on more promising regions.

8. Iteratate the steps
TPE iteratively repeats Steps 4 to 7 for a fixed number of iterations or until a convergence criterion is met. TPE adaptively allocates resources to explore promising hyperparameter configurations. In each iteration, it improves its understanding of the search space of hyperparameters and the objective function.

9. Terminate the process 
The process continues until the desired number of iterations is completed. TPE returns the combination of hyperparameters associated with the best-performing ML model found during the optimization process.

Overall, the TPE method is an efficient optimization algorithm that balances exploration and exploitation by modeling the search space with probabilistic models. Many ML practitioners have successfully applied this method to a wide range of different ML problems to get the best performance.


## Implementation Example : Binary classification problem is solved where the hyperparameters of Histogram based Gradient Boosting algorithm and KNN algorithms are optimized using Tree structured Parzen Estimator
