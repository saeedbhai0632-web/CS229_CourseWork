# CS229 Lectures 1–4: From-Scratch Coding Assignment

**Scope:** Linear regression (BGD/SGD/normal eq), locally weighted regression,
logistic regression, Newton's method (scalar + vector), perceptron,
exponential families / GLMs, softmax regression with cross-entropy.

**Rule:** `numpy` and `matplotlib` only. No `sklearn`, no `scipy.optimize`,
no autograd. You derive gradients/Hessians by hand and code them. This is
the entire point — an assignment you can only pass by having done the math.

**File:** `starter.py`. All data generators and the numerical-gradient
checker are provided and must not be modified — they are your grader.
Everything else is a stub raising `NotImplementedError`. Fill them in.

---

## How self-checking works (read this first)

You will not be told "correct" by me. Instead, every part has a **math
identity that only holds if your code is right**:

- Your batch gradient descent solution should converge to the same θ as
  your closed-form normal equation (same design matrix, well-conditioned
  data).
- Your analytically-derived gradient for logistic/softmax regression must
  match a numerical (finite-difference) gradient of your own loss function
  to ~4 decimal places.
- Newton's method on logistic regression should reach the gradient-descent
  optimum in **far fewer iterations** (this is the whole point of
  Lecture 4 — quadratic vs linear convergence).
- Your exponential-family check should show `dA/dη ≈ sigmoid(η)` and
  `d²A/dη² ≈ Var(y)` for the Bernoulli, purely from finite differences on
  a function you write from the log-partition formula — nobody hands you
  the sigmoid, you're supposed to *rediscover* it falls out of `A(η)`.

If any of these checks fail, your math or your code (not mine) is wrong.
The `if __name__ == "__main__":` block runs all checks and prints
PASS/FAIL with tolerances. Don't move to the next part until the current
one passes.

---

## Part 1 — Linear Regression

Implement, in this order:
1. `add_intercept` — prepend a column of ones.
2. `hypothesis_linear(X, theta)` — h_θ(x) = θᵀx, vectorized over all rows.
3. `mse_loss(X, y, theta)` — J(θ) = (1/2m)Σ(h(x⁽ⁱ⁾)−y⁽ⁱ⁾)².
4. `gradient_mse(X, y, theta)` — derive ∇J(θ) yourself (don't look it up —
   you already derived it in lecture; it's a clean matrix expression in
   X, y, θ). The numerical gradient checker will catch you if you get a
   sign or a 1/m factor wrong.
5. `batch_gradient_descent` and `stochastic_gradient_descent` — same
   update rule, different scan pattern over the data (SGD: one example
   at a time, shuffle each epoch).
6. `normal_equation(X, y)` — θ = (XᵀX)⁻¹Xᵀy. Use `np.linalg.solve`, not
   an explicit inverse (numerically better, and you should know why).

**Self-check:** BGD, SGD, and the normal equation should all land within
1e-2 of each other on `generate_regression_data()`.

**Written (put as a comment block at the top of the function):** explain
in 2-3 sentences why SGD's loss curve is noisier than BGD's, and why
that's an acceptable trade for large datasets.

## Part 2 — Locally Weighted Regression

1. `lwr_weights(x_query, X, tau)` — Gaussian kernel weights
   wᵢ = exp(−(xᵢ − x_query)²/2τ²) for each training point relative to one
   query point.
2. `lwr_predict(x_query, X, y, tau)` — solve the **weighted** normal
   equation θ = (XᵀWX)⁻¹XᵀWy where W = diag(w₁,...,wₘ), then return
   h_θ(x_query). Note θ is refit *per query point* — this is what "non
   parametric" means; there's no single global θ.

**Self-check:** on `generate_nonlinear_data()` (a noisy sine wave), sweep
τ over `[0.05, 0.5, 5.0]` and plot the fitted curve for each. You should
see: τ too small → overfits to noise (jagged), τ too large → underfits
toward a straight line (approaches unweighted linear regression as
τ→∞). Confirm this is what your plot actually shows.

## Part 3 — Logistic Regression

1. `sigmoid(z)` — must not overflow for large |z| (hint: branch on the
   sign of z, or clip).
2. `hypothesis_logistic(X, theta)` — sigmoid(Xθ).
3. `nll_loss(X, y, theta)` — negative log-likelihood, i.e. binary cross
   entropy: −(1/m)Σ[y log h + (1−y) log(1−h)].
4. `gradient_nll(X, y, theta)` — derive it. It comes out to almost the
   *identical form* as the linear regression gradient — that's not a
   coincidence, it's the GLM structure from lecture 4, and you'll prove
   why in Part 6.
5. `logistic_gradient_descent` — minimize `nll_loss` (equivalently,
   ascend the log-likelihood — pick one sign convention and be
   consistent).

**Self-check:** `gradient_nll` vs `numerical_gradient(nll_loss, theta)`
must match to 4 decimals. Plot the decision boundary on
`generate_classification_data()`.

## Part 4 — Newton's Method

1. `newton_scalar(f, fprime, fdoubleprime, x0, n_iters)` — 1-D Newton's
   method for root-finding: xₜ₊₁ = xₜ − f(xₜ)/f'(xₜ). Test it on a
   function you write yourself, e.g. finding a root of x² − 2. This is
   purely to build the intuition before the vector case — don't skip it
   even though it's not ML.
2. `hessian_logistic(X, theta)` — H = XᵀDX where D = diag(h₁(1−h₁), ...,
   hₘ(1−hₘ)). Derive why D has that form from the gradient you already
   wrote in Part 3.
3. `newton_logistic(X, y, n_iters)` — θ := θ − H⁻¹∇J(θ). Use
   `np.linalg.solve(H, grad)` rather than inverting H explicitly.

**Self-check:** run both `logistic_gradient_descent` (fixed reasonable
lr) and `newton_logistic` on the same data to the same final loss
threshold, and print the iteration count each needed. Newton should win
by roughly an order of magnitude. If it doesn't, your Hessian is wrong.

## Part 5 — Perceptron

1. `perceptron_predict(X, theta)` — sign(Xθ) thresholded to {0,1}.
2. `perceptron_train(X, y, lr, n_iters)` — the perceptron update rule
   (not gradient descent on a smooth loss — there isn't one here; update
   only on misclassified points).

**Self-check + written answer:** run it on `generate_classification_data()`
and on a version of that data that is *not* linearly separable (a helper
for this is provided). In a comment, explain what you observe about
convergence in each case and why — tie it back to what the lecture said
about the perceptron's convergence guarantee.

## Part 6 — Exponential Family / GLM

Work with the Bernoulli in canonical exponential-family form:
p(y;η) = b(y)·exp(ηT(y) − A(η)), with η = log(φ/(1−φ)), T(y)=y,
A(η) = log(1+eᵗᵃ), b(y)=1.

1. `bernoulli_A(eta)` — implement A(η) above (numerically stable — same
   care as your sigmoid).
2. `check_exp_family_moments(eta, h=1e-5)` — using **only** central finite
   differences on `bernoulli_A`, compute dA/dη and d²A/dη² numerically.
   Compare dA/dη to `sigmoid(eta)` and d²A/dη² to `sigmoid(eta)*(1-sigmoid(eta))`.
   They should match to ~1e-4. This is you re-deriving, from the raw
   log-partition function, the fact that E[y]=φ and Var(y)=φ(1−φ) —
   without ever hand-coding the mean/variance formulas directly.

**Written:** in 3-4 sentences, connect this back to Part 3 — explain why
the logistic regression gradient has the exact form it does, in terms of
the canonical response function g(η) = E[y;η] = dA/dη being the sigmoid.
This is the "why GLMs matter" punchline of lecture 4; if you can't
answer it in your own words, re-watch that segment before continuing.

## Part 7 — Softmax Regression

1. `softmax(Z)` — numerically stable (subtract row-max before exponentiating),
   applied row-wise to an (m, k) matrix of logits.
2. `one_hot(y, k)` — (m,) int labels → (m, k) one-hot matrix.
3. `softmax_loss(X, y_onehot, Theta)` — average cross-entropy,
   −(1/m)Σᵢ Σⱼ y_onehot[i,j]·log(softmax(XΘ)[i,j]).
4. `softmax_gradient(X, y_onehot, Theta)` — derive it (it generalizes the
   logistic gradient: ∇_Θ = (1/m)Xᵀ(softmax(XΘ) − y_onehot)).
5. `softmax_gradient_descent(X, y, k, lr, n_iters)`.

**Self-check:** gradient vs numerical gradient on a random Θ; also
verify that fitting softmax with k=2 classes gives the *same* decision
boundary (up to the parameterization) as your Part-3 logistic regression
on the same 2-class data. If they visibly disagree, one of your two
implementations has a bug — find out which by checking each against
its own numerical gradient first.

---

## Deliverables

- Completed `starter.py`, all self-checks printing PASS.
- The 4 short written comment-answers embedded where specified (Parts 1, 2, 5, 6).
- Decision boundary plots for logistic regression and softmax regression;
  the τ-sweep plot for LWR.

## What "done" looks like

Running `python starter.py` end to end prints nothing but PASS lines and
pops up ~4 plots. If you had to look anything outside the four lectures
up to derive a formula, that's a signal to rewatch that lecture segment
rather than search externally — everything needed is in what you've
already studied.