"""
CS229 Lectures 1-4 -- From-Scratch Coding Assignment
======================================================
Rules: numpy + matplotlib ONLY. No sklearn, no scipy.optimize, no autograd.

Everything in the "PROVIDED" sections is your grader -- do not edit it.
Everything else raises NotImplementedError until you fill it in.

Read ASSIGNMENT.md alongside this file.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# ======================================================================
# PROVIDED -- data generators. Do not modify.
# ======================================================================

def generate_regression_data(n=200, noise=1.0):
    X = np.random.uniform(-5, 5, size=(n, 1))
    true_theta = np.array([3.0, 2.5])  # [intercept, slope]
    y = true_theta[0] + true_theta[1] * X[:, 0] + np.random.normal(0, noise, size=n)
    return X, y, true_theta


def generate_nonlinear_data(n=200, noise=0.5):
    X = np.random.uniform(-6, 6, size=(n, 1))
    y = np.sin(X[:, 0]) * 3 + np.random.normal(0, noise, size=n)
    return X, y


def generate_classification_data(n=200, separable=True):
    n_half = n // 2
    if separable:
        X0 = np.random.randn(n_half, 2) * 0.8 + np.array([-2, -2])
        X1 = np.random.randn(n_half, 2) * 0.8 + np.array([2, 2])
    else:
        X0 = np.random.randn(n_half, 2) * 1.5 + np.array([-0.5, -0.5])
        X1 = np.random.randn(n_half, 2) * 1.5 + np.array([0.5, 0.5])
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_half), np.ones(n_half)])
    perm = np.random.permutation(n)
    return X[perm], y[perm]


def generate_multiclass_data(n=300, k=3):
    per_class = n // k
    centers = np.array([[3 * np.cos(2 * np.pi * i / k), 3 * np.sin(2 * np.pi * i / k)]
                         for i in range(k)])
    Xs, ys = [], []
    for i in range(k):
        Xs.append(np.random.randn(per_class, 2) * 0.6 + centers[i])
        ys.append(np.full(per_class, i))
    X = np.vstack(Xs)
    y = np.hstack(ys).astype(int)
    perm = np.random.permutation(len(y))
    return X[perm], y[perm]


def numerical_gradient(f, theta, eps=1e-4):
    """Central finite-difference gradient of scalar function f(theta).
    Use this to check YOUR analytical gradients. Do not modify."""
    grad = np.zeros_like(theta, dtype=float)
    for i in range(theta.shape[0]):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += eps
        theta_minus[i] -= eps
        grad[i] = (f(theta_plus) - f(theta_minus)) / (2 * eps)
    return grad


# ======================================================================
# PART 1 -- Linear Regression
# ======================================================================

def add_intercept(X):
    """(m, n) -> (m, n+1) with a leading column of ones."""
    m=X.shape[0]
    ones=np.ones((m,1))
    X=np.hstack((X,ones))
    return X
    raise NotImplementedError


def hypothesis_linear(X, theta):
    """h_theta(x) = theta^T x, vectorized. X: (m, n+1), theta: (n+1,). Returns (m,)."""
    return np.dot(X,theta)
    raise NotImplementedError


def mse_loss(X, y, theta):
    """J(theta) = (1/2m) * sum (h(x_i) - y_i)^2"""
    m=X.shape[0]
    hx=hypothesis_linear(X,theta)
    hx=hx-y
    hx=np.power(hx,2)
    sm=np.sum(hx)

    val=0.5*(1/m)*sm
    return val
    raise NotImplementedError


def gradient_mse(X, y, theta):
    """Analytical gradient of mse_loss w.r.t. theta. Derive it yourself."""
    ycap=hypothesis_linear(X,theta)
    m=X.shape[0]
    loss=ycap-y
    losst=np.transpose(loss)
    return (losst@X)/m

    raise NotImplementedError


def batch_gradient_descent(X, y, lr=0.01, n_iters=1000):
    """Returns (theta, loss_history)."""
    n=X.shape[1]-1
    m=X.shape[0]
    theta=np.ones((n+1))
    loss_history=np.ones(n_iters)
    for i in range(n_iters):
        loss=mse_loss(X,y,theta)
        loss_history[i]=loss
        theta=theta-lr*gradient_mse(X,y,theta)
    return(theta,loss_history)



    raise NotImplementedError


def stochastic_gradient_descent(X, y, lr=0.01, n_epochs=50):
    """Shuffle each epoch; update on one example at a time.
    Returns (theta, loss_history) where loss is measured on the FULL
    dataset at the end of each epoch (not per-example)."""
    n=X.shape[1]-1
    m=X.shape[0]
    theta=np.ones((n+1))
    loss_history=np.ones(n_epochs)
    for i in range(n_epochs):
        permu=np.random.permutation(m)
        for j in permu:
            ycap=hypothesis_linear(X[j],theta)
            loss=ycap-y[j]
            theta=theta-lr*X[j]*loss
        loss_history[i]=mse_loss(X,y,theta)
    return (theta,loss_history)


    raise NotImplementedError


def normal_equation(X, y):
    """Closed form: theta = (X^T X)^-1 X^T y. Use np.linalg.solve."""
    xt=np.transpose(X)
    left=(xt@X)
    right=xt@y
    return np.linalg.solve(left,right)
    raise NotImplementedError


# ======================================================================
# PART 2 -- Locally Weighted Regression
# ======================================================================

def lwr_weights(x_query, X, tau):
    """X: (m, n+1) already has intercept column; x_query: (n+1,) same form.
    Return (m,) weight vector w_i = exp(-||x_i - x_query||^2 / (2 tau^2)).
    (Only compute the distance over the non-intercept feature dims.)"""
    m=X.shape[0]
    w=np.ones((m))
    for i in range(m):
        xloss=X[i,1:]-x_query[1:]
        loss=np.sum(xloss**2)
        w[i]=np.exp((-loss)/(2*tau**2))
    return w


    raise NotImplementedError


def lwr_predict(x_query, X, y, tau):
    """X, y are RAW (no intercept added yet -- add it inside this function).
    Fit a fresh weighted least squares theta local to x_query and return h(x_query)."""
    m=X.shape[0]
    X=add_intercept(X)
    x_query = add_intercept(x_query.reshape(1, -1))[0]
    w=lwr_weights(x_query,X,tau)
    W=np.diag(w)
    xt=X.T

    theta=np.linalg.solve(xt@W@X,xt@W@y)

    return x_query@theta

    raise NotImplementedError


# ======================================================================
# PART 3 -- Logistic Regression
# ======================================================================

def sigmoid(z):
    """Numerically stable sigmoid, elementwise."""
    a=np.exp(-z)
    b=1+a
    c=1/b
    return c
    raise NotImplementedError


def hypothesis_logistic(X, theta):
    """sigmoid(X @ theta). X already has intercept column."""
    return sigmoid(X@theta)
    raise NotImplementedError


def nll_loss(X, y, theta):
    """Binary cross-entropy / negative log likelihood, averaged over m examples."""
    m=X.shape[0]
    h=hypothesis_logistic(X,theta)
    loss= -(y*np.log(h)+(1-y)*np.log(1-h))
    return np.sum(loss)/m


    raise NotImplementedError


def gradient_nll(X, y, theta):
    """Analytical gradient of nll_loss w.r.t. theta."""
    m=X.shape[0]
    h=hypothesis_logistic(X,theta)
    error=h-y
    xt=X.T
    return (xt@error)/m
    raise NotImplementedError


def logistic_gradient_descent(X, y, lr=0.1, n_iters=500):
    """Returns (theta, loss_history)."""
    m,n=X.shape
    theta=np.ones((n))
    loss_history=np.ones((n_iters))
    for i in range(n_iters):
        loss=nll_loss(X,y,theta)
        grad=gradient_nll(X,y,theta)
        loss_history[i]=loss
        theta=theta-lr*grad
    return (theta,loss_history)
    raise NotImplementedError



# ======================================================================
# PART 4 -- Newton's Method
# ======================================================================

def newton_scalar(f, fprime, fdoubleprime, x0, n_iters=10):
    """1-D Newton root-finding: x_{t+1} = x_t - f(x_t)/f'(x_t).
    (fdoubleprime is accepted for signature symmetry with the vector case
    but not needed here.) Returns (x_final, history_of_x)."""
    history=[x0]
    x=x0
    for i in range(n_iters):
        x=x-f(x)/fprime(x)
        history.append(x)
    return(x,history)
    raise NotImplementedError


def hessian_logistic(X, theta):
    """H = X^T D X, D = diag(h_i (1 - h_i)) for i=1..m."""
    h=hypothesis_logistic(X,theta)
    d=h*(1-h)
    D=np.diag(d)
    xt=X.T
    return xt@D@X
    raise NotImplementedError


def newton_logistic(X, y, n_iters=10):
    """theta := theta - H^-1 grad, via np.linalg.solve(H, grad).
    Returns (theta, loss_history)."""
    m,n=X.shape
    theta=np.ones(n)
    loss_history=np.ones(n_iters)
    for i in range(n_iters):
        loss=nll_loss(X,y,theta)
        grad=gradient_nll(X,y,theta)
        H=hessian_logistic(X,theta)
        loss_history[i]=loss
        theta=theta-np.linalg.solve(H,grad)
    return (theta,loss_history)
    raise NotImplementedError

# ======================================================================
# PART 5 -- Perceptron
# ======================================================================

def perceptron_predict(X, theta):
    """X has intercept column. Return {0,1} predictions via sign(X @ theta)."""
    z=X@theta
    return(z>= 0).astype(int)


def perceptron_train(X, y, lr=1.0, n_iters=200):
    """Classic perceptron update: only adjust theta on misclassified points.
    Returns (theta, n_mistakes_per_pass)."""
    m,n=X.shape
    theta=np.ones(n)
    error_history=[]
    for i in range(n_iters):
        errorcnt=0
        
        for j in range(m):
            ypred=perceptron_predict(X[j:j+1], theta)[0]
            if ypred!=y[j]:
                theta=theta-lr*X[j]*(ypred-y[j])
                errorcnt+=1
        error_history.append(errorcnt)
    return (theta,error_history)
    raise NotImplementedError


# ======================================================================
# PART 6 -- Exponential Family / GLM
# ======================================================================

def bernoulli_A(eta):
    """Log-partition function for Bernoulli in canonical form: log(1 + e^eta).
    Must be numerically stable for large |eta| (same trick as sigmoid)."""
    eta=np.asarray(eta)
    out=np.ones_like(eta,dtype=float)
    pos=eta>0
    neg=eta<=0
    out[pos] = eta[pos] + np.log(1 + np.exp(-eta[pos]))
    out[neg] = np.log(1 + np.exp(eta[neg]))

    return out

    raise NotImplementedError


def check_exp_family_moments(eta, h=1e-5):
    """Using ONLY central finite differences on bernoulli_A (no direct sigmoid
    call in the derivative computation itself), compute:
        dA_deta  ~= (A(eta+h) - A(eta-h)) / (2h)
        d2A_deta2 ~= (A(eta+h) - 2A(eta) + A(eta-h)) / h^2
    Return (dA_deta, d2A_deta2) as arrays matching eta's shape."""
    A_plus = bernoulli_A(eta + h)
    A_minus = bernoulli_A(eta - h)
    A = bernoulli_A(eta)

    dA_deta = (A_plus - A_minus) / (2*h)

    d2A_deta2 = (A_plus - 2*A + A_minus) / (h**2)

    return dA_deta, d2A_deta2
    raise NotImplementedError


# ======================================================================
# PART 7 -- Softmax Regression
# ======================================================================

def softmax(Z):
    """Row-wise numerically stable softmax. Z: (m, k) logits -> (m, k) probs."""
    raise NotImplementedError


def one_hot(y, k):
    """y: (m,) int labels in [0, k) -> (m, k) one-hot."""
    raise NotImplementedError


def softmax_loss(X, y_onehot, Theta):
    """X: (m, n+1) with intercept. Theta: (n+1, k). Average cross-entropy."""
    raise NotImplementedError


def softmax_gradient(X, y_onehot, Theta):
    """(1/m) * X^T @ (softmax(X @ Theta) - y_onehot). Shape (n+1, k)."""
    raise NotImplementedError


def softmax_gradient_descent(X, y, k, lr=0.1, n_iters=1000):
    """X raw (no intercept yet -- add inside). y: (m,) int labels.
    Returns (Theta, loss_history)."""
    raise NotImplementedError


# ======================================================================
# SELF-CHECK HARNESS -- run this file directly.
# ======================================================================

def _check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    return condition


if __name__ == "__main__":
    print("=== Part 1: Linear Regression ===")
    X, y, true_theta = generate_regression_data()
    Xi = add_intercept(X)

    theta_bgd, hist_bgd = batch_gradient_descent(Xi, y, lr=0.05, n_iters=2000)
    theta_sgd, hist_sgd = stochastic_gradient_descent(Xi, y, lr=0.01, n_epochs=100)
    theta_ne = normal_equation(Xi, y)

    _check("BGD ~= normal equation", np.allclose(theta_bgd, theta_ne, atol=1e-1))
    _check("SGD ~= normal equation", np.allclose(theta_sgd, theta_ne, atol=2e-1))

    g_analytic = gradient_mse(Xi, y, theta_ne)
    g_numeric = numerical_gradient(lambda t: mse_loss(Xi, y, t), theta_ne)
    _check("gradient_mse matches numerical gradient",
           np.allclose(g_analytic, g_numeric, atol=1e-2))

    plt.figure()
    plt.plot(hist_bgd, label="BGD")
    plt.plot(hist_sgd, label="SGD (per epoch)")
    plt.xlabel("iteration / epoch"); plt.ylabel("MSE loss"); plt.legend()
    plt.title("Part 1: BGD vs SGD convergence")

    print("\n=== Part 2: Locally Weighted Regression ===")
    Xn, yn = generate_nonlinear_data()
    xs_query = np.linspace(-6, 6, 200)
    plt.figure()
    plt.scatter(Xn[:, 0], yn, s=10, alpha=0.3, label="data")
    for tau in [0.05, 0.5, 5.0]:
        preds = np.array([lwr_predict(np.array([x]), Xn, yn, tau) for x in xs_query])
        plt.plot(xs_query, preds, label=f"tau={tau}")
    plt.legend(); plt.title("Part 2: LWR, effect of tau")

    print("\n=== Part 3: Logistic Regression ===")
    Xc, yc = generate_classification_data()
    Xci = add_intercept(Xc)
    theta0 = np.zeros(Xci.shape[1])
    g_analytic = gradient_nll(Xci, yc, theta0 + 0.1)
    g_numeric = numerical_gradient(lambda t: nll_loss(Xci, yc, t), theta0 + 0.1)
    _check("gradient_nll matches numerical gradient",
           np.allclose(g_analytic, g_numeric, atol=1e-3))

    theta_log, hist_log = logistic_gradient_descent(Xci, yc, lr=0.5, n_iters=1000)

    plt.figure()
    plt.scatter(Xc[:, 0], Xc[:, 1], c=yc, cmap="coolwarm", edgecolors="k")
    xx = np.linspace(Xc[:, 0].min(), Xc[:, 0].max(), 100)
    yy = -(theta_log[0] + theta_log[1] * xx) / theta_log[2]
    plt.plot(xx, yy, "k--", label="decision boundary")
    plt.legend(); plt.title("Part 3: Logistic regression")

    print("\n=== Part 4: Newton's Method ===")
    root, hist_root = newton_scalar(lambda x: x**2 - 2, lambda x: 2*x, lambda x: 2, x0=1.0)
    _check("newton_scalar finds sqrt(2)", np.isclose(root, np.sqrt(2), atol=1e-6))

    theta_newton, hist_newton = newton_logistic(Xci, yc, n_iters=10)
    target = min(hist_newton[-1], hist_log[-1]) * 1.01
    iters_gd = next((i for i, l in enumerate(hist_log) if l <= target), len(hist_log))
    iters_newton = next((i for i, l in enumerate(hist_newton) if l <= target), len(hist_newton))
    print(f"iterations to reach loss <= {target:.4f}: GD={iters_gd}, Newton={iters_newton}")
    _check("Newton converges in far fewer iterations than GD", iters_newton < iters_gd / 3)

    print("\n=== Part 5: Perceptron ===")
    theta_p, mistakes_sep = perceptron_train(Xci, yc)
    acc = np.mean(perceptron_predict(Xci, theta_p) == yc)
    _check("perceptron fits separable data (acc > 0.95)", acc > 0.95)

    Xns, yns = generate_classification_data(separable=False)
    Xnsi = add_intercept(Xns)
    theta_p2, mistakes_nonsep = perceptron_train(Xnsi, yns)
    print(f"mistakes per pass (separable): {mistakes_sep[:5]}...")
    print(f"mistakes per pass (non-separable): {mistakes_nonsep[:5]}...")

    print("\n=== Part 6: Exponential Family ===")
    eta_test = np.linspace(-4, 4, 50)
    dA, d2A = check_exp_family_moments(eta_test)
    sig = 1 / (1 + np.exp(-eta_test))
    _check("dA/deta ~= sigmoid(eta)", np.allclose(dA, sig, atol=1e-3))
    _check("d2A/deta2 ~= sigmoid(eta)(1-sigmoid(eta))", np.allclose(d2A, sig * (1 - sig), atol=1e-3))

    print("\n=== Part 7: Softmax Regression ===")
    Xm, ym = generate_multiclass_data(k=3)
    Xmi = add_intercept(Xm)
    Theta0 = np.random.randn(Xmi.shape[1], 3) * 0.01
    yoh = one_hot(ym, 3)
    g_analytic = softmax_gradient(Xmi, yoh, Theta0)
    g_numeric = np.zeros_like(Theta0)
    for i in range(Theta0.shape[0]):
        for j in range(Theta0.shape[1]):
            Tp, Tm = Theta0.copy(), Theta0.copy()
            Tp[i, j] += 1e-4; Tm[i, j] -= 1e-4
            g_numeric[i, j] = (softmax_loss(Xmi, yoh, Tp) - softmax_loss(Xmi, yoh, Tm)) / 2e-4
    _check("softmax_gradient matches numerical gradient",
           np.allclose(g_analytic, g_numeric, atol=1e-3))

    Theta_final, hist_softmax = softmax_gradient_descent(Xm, ym, k=3, lr=0.5, n_iters=1000)

    print("\nAll checks complete. Close plot windows to exit.")
    plt.show()