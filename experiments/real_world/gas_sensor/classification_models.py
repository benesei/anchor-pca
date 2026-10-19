"""Deterministic linear models used by the gas-classification experiment."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class LogisticFit:
    scaler: StandardScaler
    classifier: LogisticRegression
    converged: bool
    n_iter: int
    n_nonzero_coefficients: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.classifier.predict(self.scaler.transform(features))


@dataclass(frozen=True)
class LinearDGFit:
    method: str
    scaler: StandardScaler
    classes: np.ndarray
    parameters: np.ndarray
    converged: bool
    n_iter: int
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    objective: float
    gradient_norm: float
    constraint_violation: float
    kkt_complementarity: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        logits = sum_zero_logits(
            self.parameters,
            self.scaler.transform(features),
            n_classes=len(self.classes),
        )
        return self.classes[np.argmax(logits, axis=1)]


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    c_value: float,
    penalty: str,
    seed: int,
) -> LogisticFit:
    """Fit a class-balanced L2 or one-vs-rest L1 logistic classifier."""
    if penalty not in {"l1", "l2"}:
        raise ValueError("penalty must be 'l1' or 'l2'.")
    scaler = StandardScaler().fit(features)
    if penalty == "l2":
        settings = {"solver": "lbfgs", "max_iter": 3000}
    else:
        settings = {
            "solver": "saga",
            "max_iter": 20000,
            "tol": 1e-4,
            "multi_class": "ovr",
            "n_jobs": 1,
        }
    classifier = LogisticRegression(
        C=float(c_value),
        penalty=penalty,
        class_weight="balanced",
        random_state=int(seed),
        **settings,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(scaler.transform(features), labels)
    warned = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    n_iter = int(np.max(classifier.n_iter_))
    return LogisticFit(
        scaler=scaler,
        classifier=classifier,
        converged=bool(not warned and n_iter < classifier.max_iter),
        n_iter=n_iter,
        n_nonzero_coefficients=int(np.count_nonzero(np.abs(classifier.coef_) > 1e-12)),
    )


def domain_balanced_class_weights(labels: np.ndarray, domains: np.ndarray) -> np.ndarray:
    """Give each present class equal mass within each equal-mass domain."""
    labels = np.asarray(labels)
    domains = np.asarray(domains)
    weights = np.zeros(labels.size, dtype=float)
    for domain in np.unique(domains):
        domain_mask = domains == domain
        classes, counts = np.unique(labels[domain_mask], return_counts=True)
        for class_value, count in zip(classes, counts):
            weights[domain_mask & (labels == class_value)] = 1.0 / (
                float(classes.size) * float(count)
            )
    weights *= float(labels.size) / float(weights.sum())
    return weights


def sum_zero_coefficients(
    parameters: np.ndarray, *, n_features: int, n_classes: int
) -> np.ndarray:
    free = np.asarray(parameters, dtype=float).reshape(n_features + 1, n_classes - 1)
    return np.column_stack([free, -np.sum(free, axis=1)])


def sum_zero_logits(
    parameters: np.ndarray, features: np.ndarray, *, n_classes: int
) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    coefficients = sum_zero_coefficients(
        parameters, n_features=features.shape[1], n_classes=n_classes
    )
    return features @ coefficients[:-1] + coefficients[-1]


def one_group_loss_gradient(
    parameters: np.ndarray,
    features: np.ndarray,
    class_indices: np.ndarray,
    sample_weights: np.ndarray,
    *,
    n_classes: int,
) -> tuple[float, np.ndarray]:
    normalized = np.asarray(sample_weights, dtype=float)
    normalized = normalized / normalized.sum()
    features = np.asarray(features, dtype=float)
    class_indices = np.asarray(class_indices, dtype=int)
    logits = sum_zero_logits(parameters, features, n_classes=n_classes)
    log_probabilities = logits - logsumexp(logits, axis=1, keepdims=True)
    loss = -float(
        np.sum(normalized * log_probabilities[np.arange(class_indices.size), class_indices])
    )
    residual = np.exp(log_probabilities)
    residual[np.arange(class_indices.size), class_indices] -= 1.0
    residual *= normalized[:, None]
    full_gradient = np.vstack([features.T @ residual, residual.sum(axis=0)])
    gradient = (full_gradient[:, :-1] - full_gradient[:, [-1]]).reshape(-1)
    return loss, gradient


def group_losses_and_gradients(
    parameters: np.ndarray,
    features: np.ndarray,
    class_indices: np.ndarray,
    domains: np.ndarray,
    sample_weights: np.ndarray,
    *,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    losses = []
    gradients = []
    for domain in np.unique(domains):
        mask = domains == domain
        loss, gradient = one_group_loss_gradient(
            parameters,
            features[mask],
            class_indices[mask],
            sample_weights[mask],
            n_classes=n_classes,
        )
        losses.append(loss)
        gradients.append(gradient)
    return np.asarray(losses), np.vstack(gradients)


def regularization_value_gradient(
    parameters: np.ndarray,
    *,
    n_features: int,
    n_classes: int,
    l2_penalty: float,
) -> tuple[float, np.ndarray]:
    coefficients = sum_zero_coefficients(
        parameters, n_features=n_features, n_classes=n_classes
    )
    value = 0.5 * float(l2_penalty) * float(np.sum(coefficients[:-1] ** 2))
    full_gradient = np.zeros_like(coefficients)
    full_gradient[:-1] = float(l2_penalty) * coefficients[:-1]
    gradient = (full_gradient[:, :-1] - full_gradient[:, [-1]]).reshape(-1)
    return value, gradient


def fit_exact_group_dro(
    features: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    *,
    l2_penalty: float,
    max_iter: int = 1000,
) -> LinearDGFit:
    """Fit exact worst-source-risk linear softmax using an epigraph."""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    domains = np.asarray(domains)
    classes = np.unique(labels)
    class_indices = np.searchsorted(classes, labels)
    scaler = StandardScaler().fit(features)
    standardized = scaler.transform(features)
    sample_weights = domain_balanced_class_weights(labels, domains)
    n_parameters = (standardized.shape[1] + 1) * (classes.size - 1)
    initial = np.zeros(n_parameters + 1, dtype=float)
    initial[-1] = float(np.log(classes.size) + 1e-6)

    def objective(variables):
        theta = variables[:-1]
        regularization, gradient = regularization_value_gradient(
            theta,
            n_features=standardized.shape[1],
            n_classes=classes.size,
            l2_penalty=l2_penalty,
        )
        return float(variables[-1] + regularization), np.r_[gradient, 1.0]

    def constraints_and_jacobian(variables):
        losses, gradients = group_losses_and_gradients(
            variables[:-1],
            standardized,
            class_indices,
            domains,
            sample_weights,
            n_classes=classes.size,
        )
        return variables[-1] - losses, np.column_stack([-gradients, np.ones(losses.size)])

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=True,
        constraints={
            "type": "ineq",
            "fun": lambda variables: constraints_and_jacobian(variables)[0],
            "jac": lambda variables: constraints_and_jacobian(variables)[1],
        },
        options={"maxiter": int(max_iter), "ftol": 1e-10, "disp": False},
    )
    constraints, jacobian = constraints_and_jacobian(result.x)
    value, objective_gradient = objective(result.x)
    multipliers = np.asarray(getattr(result, "multipliers", []), dtype=float).reshape(-1)
    if multipliers.size == constraints.size:
        gradient_norm = float(
            np.linalg.norm(objective_gradient - jacobian.T @ multipliers, ord=np.inf)
        )
        complementarity = float(np.max(np.abs(multipliers * constraints)))
        dual_feasible = bool(np.min(multipliers) >= -1e-8)
    else:
        gradient_norm = np.nan
        complementarity = np.nan
        dual_feasible = True
    violation = float(max(0.0, -float(np.min(constraints))))
    converged = bool(
        result.success
        and np.isfinite(value)
        and np.all(np.isfinite(result.x))
        and violation <= 1e-7
        and dual_feasible
        and (not np.isfinite(gradient_norm) or gradient_norm <= 2e-4)
        and (not np.isfinite(complementarity) or complementarity <= 2e-5)
    )
    return LinearDGFit(
        method="exact_group_dro",
        scaler=scaler,
        classes=classes,
        parameters=np.asarray(result.x[:-1], dtype=float),
        converged=converged,
        n_iter=int(result.nit),
        optimizer_success=bool(result.success),
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        objective=float(value),
        gradient_norm=gradient_norm,
        constraint_violation=violation,
        kkt_complementarity=complementarity,
    )


def vrex_value_gradient(
    parameters: np.ndarray,
    features: np.ndarray,
    class_indices: np.ndarray,
    domains: np.ndarray,
    sample_weights: np.ndarray,
    *,
    n_classes: int,
    beta: float,
    l2_penalty: float,
) -> tuple[float, np.ndarray]:
    losses, gradients = group_losses_and_gradients(
        parameters,
        features,
        class_indices,
        domains,
        sample_weights,
        n_classes=n_classes,
    )
    centered = losses - losses.mean()
    value = float(losses.mean() + beta * np.mean(centered**2))
    gradient = gradients.mean(axis=0) + 2.0 * beta * np.mean(
        centered[:, None] * gradients, axis=0
    )
    regularization, regularization_gradient = regularization_value_gradient(
        parameters,
        n_features=features.shape[1],
        n_classes=n_classes,
        l2_penalty=l2_penalty,
    )
    return value + regularization, gradient + regularization_gradient


def fit_vrex(
    features: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    *,
    beta: float,
    l2_penalty: float,
    max_iter: int = 1000,
    initial_parameters: np.ndarray | None = None,
) -> LinearDGFit:
    """Fit a deterministic linear VREx classifier."""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    domains = np.asarray(domains)
    classes = np.unique(labels)
    class_indices = np.searchsorted(classes, labels)
    scaler = StandardScaler().fit(features)
    standardized = scaler.transform(features)
    sample_weights = domain_balanced_class_weights(labels, domains)
    n_parameters = (standardized.shape[1] + 1) * (classes.size - 1)
    initial = (
        np.zeros(n_parameters, dtype=float)
        if initial_parameters is None
        else np.asarray(initial_parameters, dtype=float).copy()
    )

    def objective(parameters):
        return vrex_value_gradient(
            parameters,
            standardized,
            class_indices,
            domains,
            sample_weights,
            n_classes=classes.size,
            beta=float(beta),
            l2_penalty=float(l2_penalty),
        )

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iter), "ftol": 1e-12, "gtol": 1e-6, "maxls": 50},
    )
    value, gradient = objective(result.x)
    gradient_norm = float(np.linalg.norm(gradient, ord=np.inf))
    converged = bool(
        result.success
        and np.isfinite(value)
        and np.all(np.isfinite(result.x))
        and gradient_norm <= 1e-4
    )
    return LinearDGFit(
        method="vrex",
        scaler=scaler,
        classes=classes,
        parameters=np.asarray(result.x, dtype=float),
        converged=converged,
        n_iter=int(result.nit),
        optimizer_success=bool(result.success),
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        objective=float(value),
        gradient_norm=gradient_norm,
        constraint_violation=0.0,
        kkt_complementarity=0.0,
    )
