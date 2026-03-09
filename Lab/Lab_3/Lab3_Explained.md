# Lab 3 — Complete Explanation: Linear Regression, Logistic Regression & Regularization

---

## Table of Contents

1. [Part 1: What is Linear Regression?](#part-1)
2. [Part 2: Loading Dependencies](#part-2)
3. [Part 3: Dataset 1 — Simple Linear Regression](#part-3)
4. [Part 4: Dataset 2 — Handling Non-Linear Data with Feature Engineering](#part-4)
5. [Part 5: Boston Housing Dataset — Multiple Linear Regression](#part-5)
6. [Part 6: Normalization (MinMax Scaling)](#part-6)
7. [Part 7: What is Logistic Regression?](#part-7)
8. [Part 8: Wisconsin Breast Cancer — Logistic Regression in Practice](#part-8)
9. [Part 9: Why Normalization Matters for Gradient-Based Solvers](#part-9)
10. [Part 10: Regularization — L1 and L2](#part-10)
11. [Part 11: Ridge Regression on Boston Dataset with Validation](#part-11)
12. [Key Takeaways](#key-takeaways)

---

<a name="part-1"></a>
## Part 1: What is Linear Regression?

### The Core Idea

Linear regression is the simplest machine learning model for **regression** (predicting a continuous number). It assumes that the relationship between input features and the output is a **straight line** (or a flat hyperplane in higher dimensions).

### The Equation

```
y = β₀ + β₁·x₁ + β₂·x₂ + ... + βₙ·xₙ + ε
```

Breaking this down piece by piece:

| Symbol | Name | What it means |
|--------|------|---------------|
| `y` | Target variable | The value we want to predict (e.g., house price) |
| `x₁, x₂, ..., xₙ` | Features | The input variables we use to make the prediction |
| `β₀` | Intercept (bias) | The baseline value of y when all features are 0 |
| `β₁, β₂, ..., βₙ` | Coefficients (weights) | How much each feature contributes to the prediction. These are **learned** during training |
| `ε` | Residual (error) | The part of reality the model can't capture — noise, missing variables, etc. |

### How Does It Learn?

The model finds the best `β` values by **minimizing the Sum of Squared Residuals (SSR)**:

```
SSR = Σ (yᵢ - ŷᵢ)²
```

where `ŷᵢ` is the model's prediction and `yᵢ` is the actual value. This method is called **Ordinary Least Squares (OLS)**. Scikit-Learn's `LinearRegression` uses OLS — it solves the problem using a direct matrix calculation (not gradient descent), which means you always get the exact same answer.

---

<a name="part-2"></a>
## Part 2: Loading Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets, preprocessing, linear_model
from sklearn.model_selection import train_test_split
```

Line-by-line:

- **`import numpy as np`** — NumPy is the core numerical library. It gives us fast arrays and math operations (like `np.mean`, `np.linspace`). The alias `np` is a universal convention.

- **`import pandas as pd`** — Pandas provides DataFrames, which are like Excel spreadsheets in Python. You use them to load, inspect, and manipulate tabular data. The alias `pd` is standard.

- **`import matplotlib.pyplot as plt`** — Matplotlib is the foundational plotting library. `pyplot` is its MATLAB-like interface for creating graphs. The alias `plt` is standard.

- **`from sklearn import datasets, preprocessing, linear_model`** — From Scikit-Learn, we import three sub-modules:
  - `datasets`: Pre-built datasets (like breast cancer data)
  - `preprocessing`: Tools for normalizing/scaling data
  - `linear_model`: Contains LinearRegression, LogisticRegression, Ridge, etc.

- **`from sklearn.model_selection import train_test_split`** — A utility function that randomly splits your data into training and testing sets.

---

<a name="part-3"></a>
## Part 3: Dataset 1 — Simple Linear Regression (One Feature)

### Loading and Inspecting the Data

```python
df = pd.read_csv('dataset1.csv')
print(df.shape)
```
**Output:** `(100, 2)`

- `pd.read_csv('dataset1.csv')` reads the CSV file into a DataFrame called `df`.
- `df.shape` returns a tuple: `(rows, columns)`. So we have **100 data points** and **2 columns** (one feature `x1` and one target `y`).

```python
df.head(10)
```

This displays the first 10 rows of the DataFrame. The data looks like:
```
         x1          y
0   9.363503  37.889247
1  23.767858  93.576394
...
```

As `x1` increases, `y` increases roughly proportionally — this hints at a **linear relationship**.

### Extracting Features and Target

```python
X = df[['x1']].values
print(X.shape)  # (100, 1)

y = df['y'].values
print(y.shape)  # (100,)
```

**Critical distinction:**
- `df[['x1']]` uses **double brackets**, returning a DataFrame (2D). `.values` converts it to a 2D NumPy array of shape `(100, 1)`. Scikit-Learn expects features as a **2D array**: `(n_samples, n_features)`.
- `df['y']` uses **single brackets**, returning a Series (1D). `.values` converts it to a 1D NumPy array of shape `(100,)`. The target can be 1D.

### Visualizing the Data

```python
plt.scatter(X, y)
plt.grid('on')
```

This creates a **scatter plot** with `x1` on the horizontal axis and `y` on the vertical axis. The grid helps you visually judge the trend. Since the points form a roughly straight line going upward, linear regression is appropriate.

### Splitting into Train and Test Sets

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1867500)
print("X_train shape:", X_train.shape)  # (80, 1)
print("X_test shape:", X_test.shape)    # (20, 1)
```

- `test_size=0.20` means **20% of data goes to testing** (20 samples), **80% to training** (80 samples).
- `random_state=1867500` is a seed for the random number generator. Using the same seed guarantees the **same split every time** you run the code. This is crucial for reproducibility.
- The function returns 4 arrays: training features, testing features, training targets, testing targets.

**Why split?** If you train and evaluate on the same data, you can't tell whether the model truly learned the pattern or just memorized the data. The test set acts as "unseen data" to evaluate real-world performance.

### Visualizing the Split

```python
plt.plot(X_train, y_train, "ko")
plt.plot(X_test, y_test, 'r*')
plt.axis([min(X)-1, max(X)+1, min(y)-1, max(y)+1])
```

- `"ko"` means **black circles** (k=black, o=circle) — these are training points.
- `'r*'` means **red stars** — these are test points.
- `plt.axis([...])` manually sets the x and y axis limits so all points are visible with a small margin.

### Training the Model

```python
model = linear_model.LinearRegression()
model = model.fit(X_train, y_train)

theta0 = model.intercept_
theta1 = model.coef_
print(theta0)   # 1.1505712126468026
print(theta1)   # [3.88214221]
```

Line-by-line:
- `linear_model.LinearRegression()` creates a new, untrained linear regression model object.
- `model.fit(X_train, y_train)` **trains the model** — it finds the best `β₀` and `β₁` values that minimize the SSR on the training data.
- `model.intercept_` is `β₀` (the y-intercept) = **1.15**
- `model.coef_` is the array of feature coefficients. Here it's `[3.88]`, meaning `β₁ = 3.88`.

So the learned equation is: **y ≈ 1.15 + 3.88 · x₁**

This means: for every 1-unit increase in `x1`, `y` increases by about 3.88 units.

**Key note from the lab:** Running `.fit()` multiple times gives the **same result** because Scikit-Learn uses a direct matrix solver (not random initialization + gradient descent).

### Plotting the Regression Line

```python
xx = np.linspace(min(X), max(X), 100)
yy = theta0 + theta1 * xx
plt.plot(xx, yy, lw=4)
plt.plot(X_train, y_train, 'ko')
plt.grid('on')
plt.axis([min(X)-1, max(X)+1, min(y)-1, max(y)+1])
plt.show()
```

- `np.linspace(min(X), max(X), 100)` creates 100 evenly spaced points between the minimum and maximum x values. This gives us a smooth line.
- `yy = theta0 + theta1 * xx` computes the predicted y for each x using the learned equation.
- `lw=4` sets the line width to 4 pixels (makes it bold and visible).
- The result is the **best-fit line** plotted over the training data.

A second plot shows the same line against the **test data** (red stars) to visually check how well the model generalizes.

### Evaluating Performance with MSE

```python
mse_train = np.mean((y_train - model.predict(X_train))**2)
print("Training loss:", mse_train)    # 18.03

y_predicted = model.predict(X_test)
mse_test = np.mean((y_test - y_predicted)**2)
print("Testing loss:", mse_test)      # 29.16
```

**Mean Squared Error (MSE)** measures how far off predictions are, on average:

```
MSE = (1/n) × Σ (yᵢ - ŷᵢ)²
```

Step-by-step for `mse_train`:
1. `model.predict(X_train)` — get predictions for every training sample
2. `y_train - model.predict(X_train)` — compute the error (residual) for each sample
3. `(...)**2` — square each error (so negatives don't cancel positives)
4. `np.mean(...)` — take the average

**Results:**
- Training MSE = 18.03 (how well the model fits data it's seen)
- Testing MSE = 29.16 (how well it generalizes to new data)

The test error is higher than training error — this is **normal and expected**. The model was optimized on training data, so it naturally does slightly worse on unseen data. If the gap were huge, it might indicate **overfitting**.

---

<a name="part-4"></a>
## Part 4: Dataset 2 — Handling Non-Linear Data with Feature Engineering

### The Problem

```python
df = pd.read_csv("dataset2.csv")
X = df[['x']].values
y = df['y'].values
plt.plot(X, y, 'ko')
plt.grid('on')
```

When you plot this data, the relationship is **parabolic** (U-shaped), not a straight line. The pattern looks like: `y ≈ m·x² + c`.

A straight line cannot fit a parabola. So can we still use linear regression?

### The Solution: Feature Engineering

```python
df['x^2'] = df['x']**2
df.head()
```

**Output:**
```
           x           y         x^2
0 -15.000000  122.434283  225.000000
1 -14.595960  103.755732  213.042037
...
```

- `df['x']**2` squares every value in the `x` column.
- We store the result as a **new column** called `x^2`.

**The key insight:** Even though the relationship between `y` and `x` is non-linear, the relationship between `y` and `x²` is **linear**! Linear regression is "linear in the parameters" (β values), not necessarily in the features.

```python
plt.scatter(df['x^2'].values, df['y'].values, c='black')
```

If you plot `y` vs `x²`, you'll see a **linear trend** — confirming our approach works.

### Training with Both Features

```python
X = df[['x', 'x^2']].values
y = df['y'].values
print("Shape of X:", X.shape)  # (100, 2)
```

Now each sample has **two features**: the original `x` and the engineered `x²`. The feature matrix is `(100, 2)`.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)
```

Same 80/20 split as before.

```python
model = linear_model.LinearRegression()
model = model.fit(X_train, y_train)
y_predict = model.predict(X_test)

theta0 = model.intercept_    # -1.70
theta1 = model.coef_         # [-0.0072, 0.5064]
```

- `theta1[0]` = -0.0072 (coefficient for `x`) — essentially **zero**, which makes sense because `x` alone doesn't explain the parabola.
- `theta1[1]` = 0.5064 (coefficient for `x²`) — this is the **significant** coefficient.

So the learned equation is approximately: **y ≈ -1.70 + 0·x + 0.51·x²**, which simplifies to **y ≈ 0.51·x² - 1.70** — exactly the parabolic shape we expected!

### Plotting the Parabola

```python
xx = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
yy = theta0 + theta1[0]*xx + theta1[1]*(xx**2)
plt.plot(xx, yy, lw=4)
plt.plot(X_train[:,0], y_train, 'ko')
```

- `X[:,0]` extracts the first column (original `x` values) from all rows.
- The equation uses both `xx` and `xx**2` to draw the parabola.
- `X_train[:,0]` plots training points using the original x coordinate.

### Performance

```python
mse_train = np.mean((y_train - model.predict(X_train))**2)
print("Training loss:", mse_train)  # 355.13
```

The MSE is higher here because the data has more noise (scatter around the parabola) compared to Dataset 1.

---

<a name="part-5"></a>
## Part 5: Boston Housing Dataset — Multiple Linear Regression

### What is the Boston Housing Dataset?

This is a classic ML dataset with **506 samples** and **13 features** describing neighborhoods in Boston. The goal is to predict **MEDV** (Median value of homes in $1000s).

**Note:** Scikit-Learn removed this dataset due to ethical concerns about one of its features. The lab loads it from the raw source at Carnegie Mellon University.

### Loading the Raw Data

```python
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
```

- `sep="\\s+"` tells pandas to use **any whitespace** (one or more spaces/tabs) as the column separator. This is a regular expression: `\s` means whitespace, `+` means one or more.
- `skiprows=22` skips the first 22 lines of the file (which contain a text description, not data).
- `header=None` means the file has no header row, so pandas assigns numeric column names (0, 1, 2, ...).

### Why the Data Needs Reshaping

The raw file stores each sample across **two lines**: the first line has 11 values, the second has 3 values. So `raw_df` has twice as many rows as actual data points, and many NaN values.

```python
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :3]])
```

This is a powerful one-liner. Let's break it down:

- `raw_df.values` converts the DataFrame to a NumPy array.
- `[::2, :]` means "every other row starting from 0" (rows 0, 2, 4, ...) and **all columns**. These are the "first lines" with 11 features.
- `[1::2, :3]` means "every other row starting from 1" (rows 1, 3, 5, ...) and **only the first 3 columns**. These are the "second lines" with the remaining 3 features.
- `np.hstack(...)` **horizontally stacks** these two arrays side by side, giving us 11 + 3 = 14 columns per sample.

```python
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
           'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.DataFrame(data, columns=columns)
```

Assigns human-readable column names and wraps the array back into a DataFrame.

### Extracting Features and Target

```python
X = data[columns[:-1]].to_numpy()    # All columns except MEDV
y = data[columns[-1]].to_numpy()     # Only MEDV
```

- `columns[:-1]` is list slicing: everything except the last element (the 13 features).
- `columns[-1]` is the last element: `'MEDV'` (the target).
- `.to_numpy()` converts DataFrame columns to a NumPy array.

**Output:** `X.shape = (506, 13)`, `y.shape = (506,)`

### Looking at Feature Ranges

```python
for i in range(X.shape[-1]):
    print(min(X[:,i]),'---',max(X[:,i]))
```

**Output:**
```
0.00632 --- 88.9762
0.0 --- 100.0
0.46 --- 27.74
...
```

- `X.shape[-1]` is the last dimension = 13 (number of features).
- `X[:,i]` selects all rows for feature `i`.
- This shows that features have **wildly different scales**: one ranges 0–1, another 0–100, another 187–711. This is why normalization matters.

---

<a name="part-6"></a>
## Part 6: Normalization (MinMax Scaling)

### What is MinMax Scaling?

```python
X_norm = preprocessing.minmax_scale(X)
```

MinMax scaling transforms each feature to the range [0, 1] using:

```
x_scaled = (x - x_min) / (x_max - x_min)
```

After scaling, every feature has a minimum of 0 and maximum of 1 (or very close to 1 due to floating-point arithmetic).

### Why Normalize?

The lab explains three reasons:

1. **For gradient-based learning (CRITICAL):** Gradient descent updates weights proportionally to feature values. If one feature is 100x larger than another, its gradient will dominate and training becomes unstable or slow.

2. **For explainability:** After normalization, all coefficients are on the same scale, so you can directly compare which feature has the most influence.

3. **For Scikit-Learn's OLS (not critical):** Since OLS uses a matrix solver (not gradients), normalization isn't strictly needed for getting good predictions — but it helps with numerical stability and interpretability.

**Warning:** MinMax scaling is sensitive to **outliers**. If one value is 1000 and the rest are 0–10, everything except the outlier gets squished near 0.

### With Normalization — Training and Evaluation

```python
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.20, random_state=100)

model = linear_model.LinearRegression()
model = model.fit(X_train, y_train)
y_predict = model.predict(X_test)

mse_train = np.mean((y_train - model.predict(X_train))**2)
print("Training loss:", mse_train)    # 21.65
mse_test = np.mean((y_test - y_predict)**2)
print("Testing loss:", mse_test)      # 23.62
```

### Interpreting Coefficients (Normalized)

```python
theta0 = model.intercept_    # 27.00
theta1 = model.coef_
for i in range(len(theta1)):
    print(f'{columns[i]} coefficient = {theta1[i]}')
```

Because features are normalized, the coefficients are **directly comparable**:

| Feature | Coefficient | Meaning |
|---------|-------------|---------|
| CRIM (crime rate) | -7.25 | Higher crime → lower home value |
| RM (rooms per dwelling) | +19.17 | **Strongest feature** — more rooms → much higher value |
| NOX (pollution) | -7.84 | More pollution → lower value |
| CHAS (near river) | +3.06 | River access → higher value |

### Without Normalization — Same Performance, Unreadable Coefficients

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)
# ... same training code ...
```

**Output:** MSE is virtually identical (21.65 train, 23.62 test), but the coefficients are now on their original scales (e.g., RM coefficient = 3.67 instead of 19.17) and you **cannot compare them** to each other because each feature has a different range.

---

<a name="part-7"></a>
## Part 7: What is Logistic Regression?

### The Problem with Linear Regression for Classification

For classification tasks (e.g., "is this tumor malignant or benign?"), the output should be either 0 or 1 (or a probability between 0 and 1). A straight line from linear regression outputs values from -∞ to +∞, which doesn't make sense for classification.

### The Sigmoid Function

The sigmoid function squashes any real number into the range [0, 1]:

```
sigmoid(x) = 1 / (1 + e^(-x))
```

- When x is very negative → sigmoid ≈ 0
- When x = 0 → sigmoid = 0.5
- When x is very positive → sigmoid ≈ 1

### Combining Linear Regression + Sigmoid = Logistic Regression

The key idea: take the linear regression output and **pass it through the sigmoid**:

```
f(X) = sigmoid(β₀ + β₁·x₁ + β₂·x₂ + ... + βₙ·xₙ)
     = 1 / (1 + e^(-(β₀ + β₁·x₁ + ... + βₙ·xₙ)))
```

Now the output is always between 0 and 1, which we interpret as a **probability**. If f(X) > 0.5, we predict class 1; otherwise class 0.

### Visualization in the Lab

```python
x1 = 5 + 2*np.random.randn(100)       # 100 points centered at 5
x2 = -5 + 1.75*np.random.randn(100)   # 100 points centered at -5
y1 = np.ones_like(x1)                  # Label = 1 for group 1
y2 = np.zeros_like(x2)                 # Label = 0 for group 2
```

- `np.random.randn(100)` generates 100 random numbers from a standard normal distribution (mean=0, std=1).
- `5 + 2*np.random.randn(100)` shifts the mean to 5 and scales the spread to std=2.
- `np.ones_like(x1)` creates an array of 1s with the same shape as x1.

The plot shows two clusters: red dots at y=1, blue dots at y=0. The black line (linear regression) misses the data entirely, while the green S-curve (logistic regression) captures the binary nature of the data perfectly.

---

<a name="part-8"></a>
## Part 8: Wisconsin Breast Cancer — Logistic Regression in Practice

### Loading the Dataset

```python
data = datasets.load_breast_cancer()
X = data['data']
y = data['target']

print("shape of X =", X.shape)    # (569, 30)
print("shape of y =", y.shape)    # (569,)
print("feature names:", data["feature_names"])
```

- `datasets.load_breast_cancer()` loads a built-in Scikit-Learn dataset with 569 tumor samples, each described by 30 features (radius, texture, perimeter, area, smoothness, etc.).
- `data['target']` is 0 (malignant) or 1 (benign) for each sample.

### Normalization and Splitting

```python
X_norm = preprocessing.minmax_scale(X)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.20, random_state=100)
```

Same as before — normalize all 30 features to [0,1] range, then do an 80/20 split.

### Training Logistic Regression

```python
model = linear_model.LogisticRegression(C=100, fit_intercept=True, solver='lbfgs', max_iter=100)
model = model.fit(X_train, y_train)
y_predict = model.predict(X_test)
```

Key parameters:
- **`C=100`**: The **inverse regularization strength**. Higher C = less regularization = more freedom for the model. `C=100` means very light regularization. (Regularization is explained in Part 10.)
- **`fit_intercept=True`**: Include the β₀ bias term (default is True).
- **`solver='lbfgs'`**: The optimization algorithm. LBFGS is a **gradient-based** method (unlike OLS which uses direct matrix solving). This is why normalization matters here.
- **`max_iter=100`**: Maximum number of gradient descent iterations allowed.

### Measuring Accuracy (Not MSE!)

```python
acc_training = np.mean(y_train == model.predict(X_train)) * 100
acc_testing = np.mean(y_test == y_predict) * 100
print("Training accuracy (%) =", acc_training)    # 99.34%
print("Testing accuracy (%) =", acc_testing)       # 97.37%
```

For classification, we use **accuracy** instead of MSE:
- `y_train == model.predict(X_train)` creates a boolean array: True where the prediction matches the actual label, False otherwise.
- `np.mean(...)` of booleans gives the fraction of True values (since True=1, False=0).
- Multiply by 100 to get a percentage.

**Result:** 99.3% training accuracy and 97.4% testing accuracy — excellent performance!

### Without Normalization

```python
model = linear_model.LogisticRegression(C=100, fit_intercept=True, solver='lbfgs', max_iter=10000)
```

Notice `max_iter=10000` (vs 100 before). Without normalization, the gradient-based solver struggles:
- Training accuracy drops to **97.8%** (from 99.3%)
- Testing accuracy drops to **95.6%** (from 97.4%)
- It needs **100x more iterations** to converge

This dramatically demonstrates why normalization is essential for gradient-based methods.

---

<a name="part-9"></a>
## Part 9: Why Normalization Matters for Gradient-Based Solvers

### The Math

The gradient update rule is:

```
βᵢ = βᵢ - α × (d/dβᵢ) Loss
```

For MSE loss with a linear model, the gradient with respect to βᵢ works out to:

```
βᵢ = βᵢ - (2α/n) × Σ (ŷⱼ - yⱼ) × xⱼᵢ
```

**The problem:** The gradient is **multiplied by xᵢ** (the feature value). If feature `x₁` ranges 0–1 but feature `x₂` ranges 0–1000, then:
- The gradient for β₂ will be ~1000x larger than for β₁
- The same learning rate α can't work well for both: too small for β₁, too large for β₂

**The solution:** Normalize all features to the same range. Then all gradients are on a similar scale, and a single learning rate works well for all weights.

---

<a name="part-10"></a>
## Part 10: Regularization — L1 and L2

### What is Regularization?

Sometimes a model's learned weights become very large, causing it to fit the training data too tightly (overfitting) and perform poorly on new data. **Regularization** adds a penalty for large weights to the loss function, forcing the model to keep weights small.

### Two Types

```
L1 regularization:  Loss = Base Loss + λ × Σ|w|
L2 regularization:  Loss = Base Loss + λ × Σw²
```

- **L1 (Lasso):** Penalizes the sum of **absolute values** of weights. Tends to push some weights to exactly zero (feature selection).
- **L2 (Ridge):** Penalizes the sum of **squared values** of weights. Shrinks all weights but rarely makes them exactly zero.
- **λ (lambda):** Controls how strong the penalty is. Higher λ = more penalty = smaller weights.

### Dummy Case — Seeing Regularization in Action

```python
x_train = np.random.randn(800)
x_test = np.random.randn(200)
y_train = x_train*3 + 2 + 0.5*np.random.randn(x_train.shape[0])
y_test = x_test*2.5 + 2 + 0.5*np.random.randn(x_test.shape[0])
```

**Intentional design:** The training data has slope=3, but the test data has slope=2.5. The model will learn a slope of ~3 from training, but the true test slope is lower.

```python
x_train = preprocessing.minmax_scale(x_train)
x_test = preprocessing.minmax_scale(x_test)
```

Normalize both sets to [0,1].

### Linear Regression (No Regularization)

```python
model = linear_model.LinearRegression()
model = model.fit(x_train.reshape([-1, 1]), y_train)
```

- `.reshape([-1, 1])` converts the 1D array to 2D: `(800,)` → `(800, 1)`. The `-1` means "figure out this dimension automatically."
- This fits a line with the training slope (~3), which overshoots the test data.

### Ridge Regression (L2 Regularization)

```python
model = linear_model.Ridge(alpha=10)
model = model.fit(x_train.reshape([-1, 1]), y_train)
```

- `alpha=10` is the λ (regularization strength). Higher alpha = more penalty on large weights.
- Ridge regression penalizes the squared weights, pushing the slope **downward** from 3 toward something smaller.
- In this specific case, the regularized slope is closer to the test slope of 2.5, so the green line (regularized) fits the test data better than the blue line (unregularized).

**Important caveat from the lab:** This only works because the test slope happens to be lower than the training slope. If the test slope were higher, penalizing weights would make things worse. You need to use a **validation set** to decide the right alpha.

---

<a name="part-11"></a>
## Part 11: Ridge Regression on Boston Dataset with Validation Set

### Three-Way Split

```python
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.20, random_state=100)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=100)
```

**Two-step splitting:**
1. First split: 80% train, 20% test → (404 train, 102 test)
2. Second split: of the 404 training samples, 85% train, 15% validation → (343 train, 61 valid)

**Output:**
```
X_train shape = (343, 13)
X_valid shape = (61, 13)
X_test shape = (102, 13)
```

**Why three sets?**
- **Training set:** Used to learn the model's weights.
- **Validation set:** Used to tune hyperparameters (like alpha). You try different alpha values and pick the one with the lowest validation MSE.
- **Test set:** Used **only once at the very end** to report final performance. Never used for any decisions during training.

### Grid Search for Alpha

```python
alpha_test = [0, 0.01, 0.1, 1, 10, 100, 1000, 10000]
for alpha in alpha_test:
    model = linear_model.Ridge(alpha=alpha)
    model = model.fit(X_train, y_train)
    y_pred_valid = model.predict(X_valid)
    mse_validation = np.mean((y_pred_valid - y_valid)**2)
    print(f'Alpha value = {alpha}; MSE = {mse_validation}')
```

For each alpha value in the list:
1. Create a new Ridge model with that alpha
2. Train it on the training set
3. Predict on the **validation set** (not test set!)
4. Calculate MSE on validation predictions

**Output:**
```
Alpha = 0;     MSE = 19.88    ← Best!
Alpha = 0.01;  MSE = 19.90
Alpha = 0.1;   MSE = 20.01
Alpha = 1;     MSE = 21.20
Alpha = 10;    MSE = 31.17
Alpha = 100;   MSE = 56.33
Alpha = 1000;  MSE = 80.86
Alpha = 10000; MSE = 87.44
```

As alpha increases, MSE gets **worse**. The best alpha is **0** (no regularization). This means for this particular dataset, regularization hurts performance.

### Final Model

```python
final_model = linear_model.Ridge(alpha=0.0)
final_model = final_model.fit(X_train, y_train)
y_predict = final_model.predict(X_test)

mse_train = np.mean((y_train - final_model.predict(X_train))**2)  # 22.05
mse_test = np.mean((y_test - y_predict)**2)                        # 24.37
```

**Note:** The MSE here (22.05 train, 24.37 test) is slightly **worse** than the earlier run without validation (21.65 train, 23.62 test). Why? Because we **gave up 61 samples** from the training set to create the validation set. Less training data = slightly worse model. This is a real tradeoff in machine learning.

---

<a name="key-takeaways"></a>
## Key Takeaways

1. **Linear Regression** finds the best straight line (or hyperplane) through data by minimizing squared errors. Scikit-Learn solves this exactly using matrix math.

2. **Feature Engineering** can make non-linear problems solvable by linear regression. If y ≈ x², add x² as a feature.

3. **Train/Test Split** is essential to evaluate how well your model generalizes. Never evaluate on training data alone.

4. **Normalization (MinMax Scaling)** puts all features on the same [0, 1] scale. It's optional for OLS but **critical** for gradient-based methods like logistic regression.

5. **Logistic Regression** = Linear Regression + Sigmoid function. It outputs probabilities in [0, 1] for binary classification.

6. **Regularization (L1/L2)** penalizes large weights to prevent overfitting. Ridge (L2) shrinks weights; Lasso (L1) can zero them out. The strength is controlled by λ (alpha).

7. **Validation Sets** are used to tune hyperparameters (like alpha) without touching the test set. Grid search over candidate values and pick the one with the lowest validation error.

8. **The cost of validation:** Creating a validation set means less training data, which can slightly hurt performance.
