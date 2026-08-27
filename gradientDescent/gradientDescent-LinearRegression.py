import numpy as np;
import matplotlib.pyplot as plt;
np.random.seed(0);

n=200;
# create input x
X= np.linspace(-3, 3,  n).reshape(-1, 1);
# create output y
Y = 2* X.ravel() - 0.3 + 0.7 * np.random.rand(n);

w, b =0.0, 0.0;
lr = 0.05;
epochs =200;
losses =[];


for _ in range(epochs):
    preds = w * X.ravel() + b;
    err = preds -Y;
    loss = (err**2).mean() /2;
    losses.append(loss);
    dw = (X.ravel() * err).mean();
    db = err.mean();
    w -= lr * dw;
    b -= lr * db;




plt.figure()
plt.plot(losses, color="green")
plt.title("Linear Regression — Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

plt.figure()
plt.scatter(X, Y, color="lightgreen")
idx = np.argsort(X.ravel())
plt.plot(X[idx], (w * X + b)[idx], color="green", linewidth=2)
plt.title("Linear Regression — Fitted Line")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()