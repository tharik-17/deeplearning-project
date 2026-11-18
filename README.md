# Linear Regression: Food Truck Profit Prediction

A from-scratch implementation of linear regression with one variable, using batch gradient descent in NumPy. The model predicts the profit of a food truck from the population of the city it operates in.

Based on the first programming exercise of Andrew Ng's Coursera Machine Learning course.

## Overview

The notebook (`Linear_Regression.ipynb`) walks through:

1. Loading and exploring the dataset
2. Visualizing the data with a scatter plot
3. Implementing the cost function (mean squared error)
4. Implementing batch gradient descent
5. Training the model and plotting the fitted line
6. Plotting the cost against training iterations to confirm convergence

## Dataset

`data/ex1data1.txt` is a comma-separated file with no header and 97 rows:

| Column     | Description                                   |
|------------|-----------------------------------------------|
| Population | City population (in units of 10,000s)         |
| Profit     | Food truck profit (in units of $10,000s)      |

## Method

**Hypothesis:** `h(x) = theta0 + theta1 * x`

**Cost function:**

```
J(theta) = (1 / 2m) * sum((X * theta^T - y)^2)
```

**Gradient descent update (applied simultaneously to all parameters):**

```
theta_j := theta_j - (alpha / m) * sum((X * theta^T - y) * x_j)
```

A column of ones is prepended to `X` so the intercept `theta0` is handled as part of the matrix product.

## Hyperparameters

| Parameter     | Value |
|---------------|-------|
| Learning rate (`alpha`) | 0.01 |
| Iterations    | 1000  |
| Initial theta | `[0, 0]` |

## Results

| Metric                      | Value        |
|-----------------------------|--------------|
| Initial cost (theta = 0)    | ~32.07       |
| Final cost                  | ~4.516       |
| Learned theta0 (intercept)  | ~-3.2414     |
| Learned theta1 (slope)      | ~1.1273      |

The cost decreases steadily over the iterations and flattens out, which indicates the algorithm has converged. The fitted line follows the trend of the training data: larger city populations tend to yield higher profits.

## Requirements

- Python 3.5+
- NumPy
- pandas
- Matplotlib
- Jupyter Notebook

Install the dependencies with:

```bash
pip install numpy pandas matplotlib notebook
```

## Usage

1. Place the dataset at `data/ex1data1.txt`.
2. Launch Jupyter:

   ```bash
   jupyter notebook Linear_Regression.ipynb
   ```

3. Run all cells in order.

> **Note:** The notebook reads the data with a Windows-style path (`data\ex1data1.txt`). On macOS or Linux, change it to `data/ex1data1.txt`, or use `os.path.join('data', 'ex1data1.txt')`.

## Project Structure

```
.
├── Linear_Regression.ipynb
├── data/
│   └── ex1data1.txt
└── README.md
```

## Possible Improvements

- Vectorize the parameter update loop in `gradientDescent` instead of looping over parameters
- Replace `np.matrix` with `np.ndarray`, since `np.matrix` is deprecated
- Extend to multivariate regression with feature normalization
- Compare against the closed-form normal equation solution or scikit-learn's `LinearRegression`

## Reference

- Andrew Ng, *Machine Learning*, Coursera (Exercise 1: Linear Regression)
