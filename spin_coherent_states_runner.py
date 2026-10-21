from __future__ import annotations
import math
import cmath
import os
import time
from typing import Tuple, Dict, List, Any
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, DiagonalGate
from qiskit_aer import AerSimulator

import quantum_prime_algorithm as qpa

D = 8
P = 80           # p=60 for d=8 is suggested in paper (Appendix C)
SHOTS = 8192     # 2^13 shots are suggested in the paper
OMEGA = 0.1
T_VALS = qpa.get_t_vals(P, OMEGA)
SAVE_DIR = "state_prep_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Matplotlib Styling ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

GREEN = "#7FFF00"
BLUE = "#1E56FF"
PINK = "#FF00A2"
ORANGE = "#FFAE00"

# 1. State Generator
def get_uniform_amplitudes(d: int) -> np.ndarray:
    """Returns normalized probability amplitudes for the original uniform state."""
    return np.ones(d, dtype=complex) / np.sqrt(d)

def get_spin_coherent_amplitudes(d: int, theta: float = math.pi / 2, phi: float = 0.0) -> np.ndarray:
    """
    Generalized spin coherent state |theta, phi> mapped to computational levels k in 1..d.
    Eqs. (C2) and (C12) in arXiv:2605.28964.
    """
    c = np.zeros(d, dtype=complex)
    s = (d - 1) / 2.0
    cos_half = math.cos(theta / 2.0)
    sin_half = math.sin(theta / 2.0)

    for k in range(1, d + 1):
        m = s - (k - 1)
        binom_coeff = math.comb(d - 1, d - k)
        amp = (
            math.sqrt(binom_coeff)
            * (cos_half ** (s + m))
            * (sin_half ** (s - m))
            * cmath.exp(-1j * m * phi)
        )
        c[k - 1] = amp

    norm = np.linalg.norm(c)
    if norm > 0:
        c /= norm
    return c

def get_divisors(n: int) -> List[int]:
    """Return all positive divisors of n used for the calculation of the expected Fourier modes."""
    divs = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i * i != n:
                divs.append(n // i)
    return sorted(divs)

def analytic_Bn_generic(c: np.ndarray, n: int) -> float:
    """General lower bound B_n for any subsystem state c (Eq. 8)."""
    d = len(c)
    if n >= d:
        return 0.0

    p_weights = np.abs(c) ** 2
    total = 0.0
    for k in range(1, d - n + 1):
        for m in range(1, d):
            total += p_weights[k - 1] * p_weights[m - 1] * p_weights[k + n - 1] * p_weights[m]

    return 8.0 * total

def analytic_alpha_generic(c: np.ndarray) -> Dict[int, float]:
    """Exact expected Fourier modes alpha_n for any subsystem state c (Eq. 7)."""
    d = len(c)
    n_max = 2 * (d - 1)
    p_weights = np.abs(c) ** 2
    alpha = {}

    for n in range(2, n_max + 1):
        divs = get_divisors(n)
        mode_sum = 0.0
        for y in divs:
            step_a = n // y
            step_b = y
            if step_a >= d or step_b >= d:
                continue

            for k in range(1, d - step_a + 1):
                for m in range(1, d - step_b + 1):
                    mode_sum += (
                        p_weights[k - 1]
                        * p_weights[m - 1]
                        * p_weights[k + step_a - 1]
                        * p_weights[m + step_b - 1]
                    )
        alpha[n] = 4.0 * mode_sum

    return alpha

# 2. Circuit Builder for Arbitrary Amplitudes
def walsh_hadamard_transform(f: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform (FWHT) for 1D array of length 2^N."""
    a = f.copy()
    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return a / len(f)

def build_Ut_walsh(d: int, omega: float, t: float) -> QuantumCircuit:
    d_embed = qpa.next_power_of_two(d)
    qA = math.ceil(math.log2(d_embed))
    num_qubits = 2 * qA
    N = 1 << num_qubits

    # Subsystem A: bits 0..qA-1, Subsystem B: bits qA..2qA-1
    f = np.zeros(N, dtype=float)
    for j in range(N):
        nA = (j % d_embed) + 1
        nB = (j // d_embed) + 1
        f[j] = omega * nA * nB

    # 2. Compute Walsh coefficients
    a = walsh_hadamard_transform(f)

    # 3. Circuit using multi-qubit Pauli-Z rotations
    qc = QuantumCircuit(num_qubits, name=f"Ut_walsh_{t:.2f}")

    for u in range(1, N):
        coeff = a[u]
        if abs(coeff) < 1e-12:
            continue

        theta = -2.0 * coeff * t
        
        active_qubits = [i for i in range(num_qubits) if (u >> i) & 1]

        if len(active_qubits) == 1:
            qc.rz(theta, active_qubits[0])
        elif len(active_qubits) > 1:
            # Standard CNOT ladder for Z tensor product
            for i in range(len(active_qubits) - 1):
                qc.cx(active_qubits[i], active_qubits[i + 1])
            qc.rz(theta, active_qubits[-1])
            for i in range(len(active_qubits) - 2, -1, -1):
                qc.cx(active_qubits[i], active_qubits[i + 1])

    return qc

def build_swaptest_circuit_arbitrary(
    c: np.ndarray,
    omega: float,
    t: float,
    evolution_method: str = "walsh"
) -> Tuple[QuantumCircuit, int]:
    """Builds a SWAP-test circuit initialized with an arbitrary amplitude vector c."""
    d = len(c)
    d_embed = qpa.next_power_of_two(d)
    q_single = 2 * math.ceil(math.log2(d_embed))
    qA = q_single // 2

    qc = QuantumCircuit(1 + 2 * q_single, 1)
    anc = 0
    copy1 = list(range(1, 1 + q_single))
    copy2 = list(range(1 + q_single, 1 + 2 * q_single))

    # State preparation for Subsystems A and B
    sub_prep = QuantumCircuit(qA, name="StatePrep")
    sub_prep.append(StatePreparation(c), range(qA))

    for copy_qubits in (copy1, copy2):
        qc.compose(sub_prep, qubits=copy_qubits[:qA], inplace=True)
        qc.compose(sub_prep, qubits=copy_qubits[qA:], inplace=True)

    # Time evolution U(t)
    if evolution_method == "walsh":
        Ut = build_Ut_walsh(d, omega, t)
        qc.compose(Ut, qubits=copy1, inplace=True)
        qc.compose(Ut, qubits=copy2, inplace=True)
    elif evolution_method == "diagonal":
        phases = [
            cmath.exp(-1j * omega * ((k % d_embed) + 1) * ((k // d_embed) + 1) * t)
            for k in range(d_embed * d_embed)
        ]
        Uc = QuantumCircuit(q_single)
        Uc.append(DiagonalGate(phases), Uc.qubits)
        qc.compose(Uc, qubits=copy1, inplace=True)
        qc.compose(Uc, qubits=copy2, inplace=True)
    else:
        raise ValueError("evolution_method must be 'walsh' or 'diagonal'")

    # Controlled-SWAP on subsystem A
    qc.h(anc)
    for i in range(qA):
        qc.cswap(anc, copy1[i], copy2[i])
    qc.h(anc)
    qc.measure(anc, 0)

    return qc, d_embed


def prepare_swaptest_circuits_arbitrary(
    c: np.ndarray,
    omega: float,
    t_vals: np.ndarray,
    use_walsh: bool = True
) -> List[QuantumCircuit]:
    evo = "walsh" if use_walsh else "diagonal"
    circuits: List[QuantumCircuit] = []
    for t in t_vals:
        qc, _ = build_swaptest_circuit_arbitrary(c, omega, float(t), evolution_method=evo)
        circuits.append(qc)
    return circuits

# 3. Classification & Scores
def classify_alpha_against_Bn_adaptive(
    alpha_n: Dict[int, float],
    d: int,
    c: np.ndarray,
    error: np.ndarray | None = None,
    tolerance_ratio: float = 0.5,
    parity: bool = False
) -> Tuple[List[int], Dict[int, Dict[str, float | bool]]]:
    primes = []
    details = {}
    alpha_exp_all = analytic_alpha_generic(c)

    for n, val in alpha_n.items():
        if n < 2 or n > 2 * (d - 1):
            continue

        if parity and (n % 2 == 0) and n != 2:
            Bn = analytic_Bn_generic(c, n)
            details[n] = {
                "alpha_n": float(val),
                "B_n": float(Bn),
                "delta": float(val - Bn),
                "error": 0.0,
                "within_tol": False
            }
            continue

        Bn = analytic_Bn_generic(c, n)

        if error is not None:
            tol = error[n - 2]
        else:
            exp_val = alpha_exp_all.get(n, 0.0)
            gap = max(exp_val - Bn, 0.0)
            tol = gap * tolerance_ratio if gap > 1e-6 else 0.015

        is_prime = bool(val <= Bn + tol)
        if is_prime:
            primes.append(n)

        details[n] = {
            "alpha_n": float(val),
            "B_n": float(Bn),
            "delta": float(val - Bn),
            "error": float(tol),
            "within_tol": is_prime
        }

    return primes, details


def compute_state_quality_score(
    n_vals: np.ndarray,
    mean_alphas: np.ndarray,
    std_alphas: np.ndarray,
    bound_n: np.ndarray,
    d: int
) -> float:
    """
    Evaluates state quality:
    Computes average composite margin above B_n relative to prime fluctuations.
    """
    known_primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
    
    prime_deltas = []
    composite_gaps = []

    for idx, n in enumerate(n_vals):
        gap = mean_alphas[idx] - bound_n[idx]
        sigma = max(float(std_alphas[idx]), 1e-5)

        if n in known_primes:
            prime_deltas.append(abs(gap) + sigma)
        else:
            weight = (2 * (d - 1) - n + 1) / (2 * (d - 1))
            composite_gaps.append(gap * weight)

    if not composite_gaps or not prime_deltas:
        return 0.0

    avg_composite_gap = np.mean(composite_gaps)
    avg_prime_noise = np.mean(prime_deltas)

    if avg_prime_noise < 1e-7:
        return 0.0

    score = max(0.0, float(avg_composite_gap / avg_prime_noise))
    return score

# 4. Plots
def plot_single_configuration_mean(
    d: int,
    label: str,
    theta: float | None,
    phi: float | None,
    n_vals: np.ndarray,
    mean_alphas: np.ndarray,
    std_alphas: np.ndarray,
    bound_n: np.ndarray,
    alpha_expected: np.ndarray,
    error: np.ndarray,
    save: bool = True
):
    """Plots the mean mode spectrum over multiple runs for one specific configuration."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_alpha(0.5)
    ax.set_facecolor("white")

    ax.fill_between(
        n_vals, bound_n - error, bound_n + error,
        color=GREEN, alpha=0.25, label=r"Certification $C_n$"
    )
    ax.plot(n_vals, bound_n, label=r"Bound $B_n$", color=PINK, linewidth=2)
    ax.plot(n_vals, alpha_expected, marker='o', linestyle='-', color=ORANGE, label="Expected", linewidth=2.0)

    ax.errorbar(
        n_vals, mean_alphas, yerr=std_alphas,
        fmt='o-', color=BLUE, ecolor=BLUE, elinewidth=1.5,
        capsize=5, capthick=1.2, markersize=5, linewidth=1.8,
        label=r"Measured ($\mu \pm 1\sigma$)"
    )

    clean_label = label.replace("$", "").replace("\\", "")
    ax.set_title(f"Modes: {clean_label} ($d = {d}$)")
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$\alpha_n$")
    ax.set_xticks(n_vals)
    ax.set_xticklabels([str(int(n)) for n in n_vals])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save:
        if theta is not None and phi is not None:
            fname = os.path.join(SAVE_DIR, f"d{D}p{P}s{SHOTS}_theta{theta:.3f}_phi{phi:.3f}.png")
        else:
            fname = os.path.join(SAVE_DIR, "mean_uniform.png")
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()

def plot_all_configurations_combined(
    d: int,
    all_results: Dict[str, Dict[str, Any]],
    n_vals: np.ndarray,
    save: bool = True
):
    """
    Combined plot showing the mean Fourier modes and separation gaps of all tested configurations.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 11), sharex=True)
    
    cmap = mpl.colormaps["tab20"].resampled(max(1, len(all_results)))

    for idx, (label, res) in enumerate(all_results.items()):
        color = cmap(idx)
        score = res["score"]
        mean_alphas = res["mean_alphas"]
        bound_n = res["bound_n"]
        gaps = mean_alphas - bound_n

        ax1.plot(
            n_vals, mean_alphas, marker='o', linestyle='-',
            linewidth=1.6, markersize=4, color=color,
            label=f"{label} (Score: {score:.2f})"
        )

        ax2.plot(
            n_vals, gaps, marker='s', linestyle='--',
            linewidth=1.4, markersize=3.5, color=color,
            label=f"{label}"
        )

    ax1.set_title(f"Comparison of All Initial State Configurations ($d = {d}$)")
    ax1.set_ylabel(r"$\alpha_n$")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.28, 1.02), fontsize=9, framealpha=0.9)

    ax2.axhline(0, color="black", linestyle=":", linewidth=1)
    ax2.set_title(r"Net Separation Gap: $\alpha_n - B_n$ (Composites must sit $> 0$)")
    ax2.set_xlabel("$n$")
    ax2.set_ylabel(r"$\alpha_n - B_n$")
    ax2.set_xticks(n_vals)
    ax2.set_xticklabels([str(int(n)) for n in n_vals])
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save:
        fname = "all_configurations_combined.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"\nSaved combined overview plot to: {fname}")
        plt.close()
    else:
        plt.show()

def compute_classification_accuracy(
    d: int,
    mean_alphas: np.ndarray,
    c: np.ndarray,
    n_vals: np.ndarray,
    error: np.ndarray
) -> Tuple[int, int, float]:
    """
    Computes how many numbers in [2, 2(d-1)] are correctly classified as prime/composite.
    Returns: (correct_count, total_count, accuracy_percentage).
    """
    def is_prime(num: int) -> bool:
        if num < 2:
            return False
        for i in range(2, int(math.isqrt(num)) + 1):
            if num % i == 0:
                return False
        return True

    alpha_dict = {n: float(mean_alphas[idx]) for idx, n in enumerate(n_vals)}

    primes_found, details = classify_alpha_against_Bn_adaptive(
        alpha_n=alpha_dict,
        d=d,
        c=c,
        error=error,
        tolerance_ratio=0.5,
        parity=False
    )

    primes_set = set(primes_found)
    total_count = len(n_vals)
    correct_count = 0

    for n in n_vals:
        actual_is_prime = is_prime(int(n))
        classified_is_prime = n in primes_set
        if actual_is_prime == classified_is_prime:
            correct_count += 1

    accuracy_pct = (correct_count / total_count) * 100.0
    return correct_count, total_count, accuracy_pct

# 5. Multi-State Evaluation
def evaluate_coherent_states(
    d: int = D,
    p: int = P,
    shots: int = SHOTS,
    omega: float = OMEGA,
    runs_per_state: int = 5,
    state_grid: List[Tuple[str, float | None, float | None]] | None = None
):
    """
    Automates testing across a grid of states with runs_per_state simulation runs each.
    """
    t_vals = qpa.get_t_vals(p, omega)
    print(f"t_vals {t_vals}")
    n_vals = qpa.get_n_vals(d)
    print(f"n_vals {n_vals}")
    error = qpa.error_estimate(shots, n_vals, p)
    backend = AerSimulator()

    if state_grid is None:
        state_grid = [
            # 1. Original 2024 uniform state
            ("Original Uniform State", None, None),

            # 2. Benchmark spin-coherent state from arXiv:2605.28964
            (r"$\theta=\pi/2, \phi=0$ (Paper)", math.pi / 2, 0.0),

            # 3. More or less random configurations
            (r"$\theta=\pi/6, \phi=0$", math.pi / 6, 0.0),
            (r"$\theta=\pi/4, \phi=0$", math.pi / 4, 0.0),
            (r"$\theta=\pi/3, \phi=0$", math.pi / 3, 0.0),
            (r"$\theta=2\pi/3, \phi=0$", 2 * math.pi / 3, 0.0),
            (r"$\theta=3\pi/4, \phi=0$", 3 * math.pi / 4, 0.0),
            (r"$\theta=5\pi/6, \phi=0$", 5 * math.pi / 6, 0.0),
            (r"$\theta=\pi/2, \phi=\pi/6$", math.pi / 2, math.pi / 6),
            (r"$\theta=\pi/2, \phi=\pi/4$", math.pi / 2, math.pi / 4),
            (r"$\theta=\pi/2, \phi=\pi/3$", math.pi / 2, math.pi / 3),
            (r"$\theta=\pi/2, \phi=\pi/2$", math.pi / 2, math.pi / 2),
            (r"$\theta=\pi/2, \phi=3\pi/4$", math.pi / 2, 3 * math.pi / 4),
            (r"$\theta=\pi/2, \phi=\pi$", math.pi / 2, math.pi),
            # (r"$\theta=\pi/3, \phi=\pi/4$", math.pi / 3, math.pi / 4),
            # (r"$\theta=\pi/3, \phi=\pi/2$", math.pi / 3, math.pi / 2),
            # (r"$\theta=2\pi/3, \phi=\pi/4$", 2 * math.pi / 3, math.pi / 4),
            # (r"$\theta=\pi/8, \phi=0$", math.pi / 8, 0.0),
            # (r"$\theta=3\pi/8, \phi=0$", 3 * math.pi / 8, 0.0),
            # (r"$\theta=5\pi/8, \phi=0$", 5 * math.pi / 8, 0.0),
            # (r"$\theta=7\pi/8, \phi=0$", 7 * math.pi / 8, 0.0),
            # (r"$\theta=5\pi/12, \phi=0$", 5 * math.pi / 12, 0.0),
            # (r"$\theta=7\pi/12, \phi=0$", 7 * math.pi / 12, 0.0),
            # (r"$\theta=\pi/2, \phi=7\pi/6$", math.pi / 2, 7 * math.pi / 6),
            # (r"$\theta=\pi/2, \phi=5\pi/4$", math.pi / 2, 5 * math.pi / 4),
            # (r"$\theta=\pi/2, \phi=4\pi/3$", math.pi / 2, 4 * math.pi / 3),
            # (r"$\theta=\pi/2, \phi=3\pi/2$", math.pi / 2, 3 * math.pi / 2),
            # (r"$\theta=\pi/2, \phi=7\pi/4$", math.pi / 2, 7 * math.pi / 4),
            # (r"$\theta=\pi/4, \phi=\pi/2$", math.pi / 4, math.pi / 2),
            # (r"$\theta=\pi/4, \phi=\pi$", math.pi / 4, math.pi),
            # (r"$\theta=\pi/4, \phi=3\pi/2$", math.pi / 4, 3 * math.pi / 2),
            # (r"$\theta=\pi/3, \phi=3\pi/4$", math.pi / 3, 3 * math.pi / 4),
            # (r"$\theta=\pi/3, \phi=\pi$", math.pi / 3, math.pi),
            # (r"$\theta=2\pi/3, \phi=\pi/2$", 2 * math.pi / 3, math.pi / 2),
            # (r"$\theta=2\pi/3, \phi=\pi$", 2 * math.pi / 3, math.pi),
            # (r"$\theta=3\pi/4, \phi=\pi/4$", 3 * math.pi / 4, math.pi / 4),
            # (r"$\theta=3\pi/4, \phi=\pi/2$", 3 * math.pi / 4, math.pi / 2),
            # (r"$\theta=\pi/12, \phi=0$", math.pi / 12, 0.0),
            # (r"$\theta=11\pi/12, \phi=0$", 11 * math.pi / 12, 0.0),
            # (r"$\theta=\pi/8, \phi=\pi/4$", math.pi / 8, math.pi / 4),
            # (r"$\theta=\pi/8, \phi=\pi/2$", math.pi / 8, math.pi / 2),
            # (r"$\theta=\pi/8, \phi=\pi$", math.pi / 8, math.pi),
            # (r"$\theta=\pi/8, \phi=3\pi/2$", math.pi / 8, 3 * math.pi / 2),
            # (r"$\theta=7\pi/8, \phi=\pi/4$", 7 * math.pi / 8, math.pi / 4),
            # (r"$\theta=7\pi/8, \phi=\pi/2$", 7 * math.pi / 8, math.pi / 2),
            # (r"$\theta=7\pi/8, \phi=\pi$", 7 * math.pi / 8, math.pi),
            # (r"$\theta=7\pi/8, \phi=3\pi/2$", 7 * math.pi / 8, 3 * math.pi / 2),
            # (r"$\theta=\pi/6, \phi=\pi/6$", math.pi / 6, math.pi / 6),
            # (r"$\theta=\pi/6, \phi=\pi/4$", math.pi / 6, math.pi / 4),
            # (r"$\theta=\pi/6, \phi=\pi/2$", math.pi / 6, math.pi / 2),
            # (r"$\theta=\pi/6, \phi=\pi$", math.pi / 6, math.pi),
            # (r"$\theta=\pi/4, \phi=\pi/6$", math.pi / 4, math.pi / 6),
            # (r"$\theta=\pi/4, \phi=\pi/3$", math.pi / 4, math.pi / 3),
            # (r"$\theta=\pi/4, \phi=3\pi/4$", math.pi / 4, 3 * math.pi / 4),
            # (r"$\theta=\pi/4, \phi=7\pi/4$", math.pi / 4, 7 * math.pi / 4),
            # (r"$\theta=3\pi/8, \phi=\pi/4$", 3 * math.pi / 8, math.pi / 4),
            # (r"$\theta=3\pi/8, \phi=\pi/2$", 3 * math.pi / 8, math.pi / 2),
            # (r"$\theta=3\pi/8, \phi=3\pi/4$", 3 * math.pi / 8, 3 * math.pi / 4),
            # (r"$\theta=3\pi/8, \phi=\pi$", 3 * math.pi / 8, math.pi),
            # (r"$\theta=5\pi/12, \phi=\pi/4$", 5 * math.pi / 12, math.pi / 4),
            # (r"$\theta=5\pi/12, \phi=\pi/2$", 5 * math.pi / 12, math.pi / 2),
            # (r"$\theta=5\pi/12, \phi=\pi$", 5 * math.pi / 12, math.pi),
            # (r"$\theta=7\pi/12, \phi=\pi/4$", 7 * math.pi / 12, math.pi / 4),
            # (r"$\theta=7\pi/12, \phi=\pi/2$", 7 * math.pi / 12, math.pi / 2),
            # (r"$\theta=7\pi/12, \phi=\pi$", 7 * math.pi / 12, math.pi),
            # (r"$\theta=5\pi/8, \phi=\pi/4$", 5 * math.pi / 8, math.pi / 4),
            # (r"$\theta=5\pi/8, \phi=\pi/2$", 5 * math.pi / 8, math.pi / 2),
            # (r"$\theta=5\pi/8, \phi=3\pi/4$", 5 * math.pi / 8, 3 * math.pi / 4),
            # (r"$\theta=5\pi/8, \phi=\pi$", 5 * math.pi / 8, math.pi),
            # (r"$\theta=2\pi/3, \phi=3\pi/4$", 2 * math.pi / 3, 3 * math.pi / 4),
            # (r"$\theta=2\pi/3, \phi=3\pi/2$", 2 * math.pi / 3, 3 * math.pi / 2),
            # (r"$\theta=3\pi/4, \phi=\pi$", 3 * math.pi / 4, math.pi),
            # (r"$\theta=3\pi/4, \phi=3\pi/2$", 3 * math.pi / 4, 3 * math.pi / 2),
            # (r"$\theta=5\pi/6, \phi=\pi/4$", 5 * math.pi / 6, math.pi / 4),
            # (r"$\theta=5\pi/6, \phi=\pi/2$", 5 * math.pi / 6, math.pi / 2),
            # (r"$\theta=5\pi/6, \phi=\pi$", 5 * math.pi / 6, math.pi),
            # (r"$\theta=5\pi/6, \phi=3\pi/2$", 5 * math.pi / 6, 3 * math.pi / 2),
        ]

    all_results = {}

    print(f"=== Starting Automation: {len(state_grid)} Configurations, {runs_per_state} Runs Each ===")
    print(f"Dimension: d={d}, Shots: {shots}, Time partitions: p={p}\n")

    for label, theta, phi in state_grid:
        if theta is None or phi is None:
            print(f"--- Running Configuration: {label} (Hadamard Superposition) ---")
            c = get_uniform_amplitudes(d)
        else:
            print(f"--- Running Configuration: {label} (theta={theta:.3f}, phi={phi:.3f}) ---")
            c = get_spin_coherent_amplitudes(d, theta=theta, phi=phi)
        print(f"c {c}")

        alpha_exp_dict = analytic_alpha_generic(c)
        alpha_expected = np.array([alpha_exp_dict.get(n, 0.0) for n in n_vals])
        bound_n = np.array([analytic_Bn_generic(c, n) for n in n_vals])

        circuits = prepare_swaptest_circuits_arbitrary(c, omega, t_vals, use_walsh=False)
        transpiled = transpile(circuits, backend=backend)

        measured_runs = []
        for r in range(runs_per_state):
            job = backend.run(transpiled, shots=shots)
            result = job.result()

            gamma_vals = np.array([
                2.0 * (result.get_counts(i).get("0", 0) / shots) - 1.0
                for i in range(len(t_vals))
            ])
            # gamma0 = float(gamma_vals[0])
            # kappa = 1.0 if gamma0 <= 0 else min(3.0, 1.0 / gamma0)
            # gamma_vals = np.clip(gamma_vals * kappa, -1.0, 1.0)

            modes = qpa.fourier_modes(d, omega=omega, t_vals=t_vals, gamma_vals=gamma_vals)
            measured_runs.append([modes.get(n, 0.0) for n in n_vals])

        measured_array = np.array(measured_runs)
        mean_alphas = np.mean(measured_array, axis=0)
        std_alphas = np.std(measured_array, axis=0)

        # Compute quality score, accuracy and plot the Fourier modes
        score = compute_state_quality_score(n_vals, mean_alphas, std_alphas, bound_n, d=d)
        correct_cnt, total_cnt, acc_pct = compute_classification_accuracy(
            d=d,
            mean_alphas=mean_alphas,
            c=c,
            n_vals=n_vals,
            error=error
        )

        plot_single_configuration_mean(
            d=d,
            label=label,
            theta=theta,
            phi=phi,
            n_vals=n_vals,
            mean_alphas=mean_alphas,
            std_alphas=std_alphas,
            bound_n=bound_n,
            alpha_expected=alpha_expected,
            error=error,
            save=True
        )

        all_results[label] = {
            "theta": theta,
            "phi": phi,
            "c": c,
            "bound_n": bound_n,
            "alpha_expected": alpha_expected,
            "mean_alphas": mean_alphas,
            "std_alphas": std_alphas,
            "score": score,
            "correct_count": correct_cnt,
            "total_count": total_cnt,
            "accuracy_pct": acc_pct
        }
        print(f"-> Completed. Quality Score: {score:.3f} | Correct: {correct_cnt}/{total_cnt} ({acc_pct:.1f}%)\n")

    plot_all_configurations_combined(d, all_results, n_vals, save=True)

    # Print to console and log to a txt file
    # log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # log_filename = os.path.join(SAVE_DIR, f"leaderboard_d{d}_{log_timestamp}.txt")
    log_filename = os.path.join(SAVE_DIR, f"leaderboard_d{d}.txt")

    sorted_configs = sorted(all_results.items(), key=lambda x: x[1]["score"], reverse=True)

    header = "\n======================================== LEADERBOARD ========================================\n"
    header += f"{'Rank':<5} | {'Configuration':<32} | {'Score':<8} | {'Correct':<10} | {'Accuracy':<10} | {'Status'}\n"
    header += "-" * 83

    print(header)

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"Leaderboard Results for d={d}, p={p}, shots={shots}, runs={runs_per_state}\n")
        f.write(f"Evaluated Search Range: [2, {2 * (d - 1)}] ({len(n_vals)} integers)\n")
        f.write(header + "\n")

        for rank, (cfg_name, data) in enumerate(sorted_configs, start=1):
            clean_name = cfg_name.replace("$", "").replace("\\", "")
            status = "BEST" if rank == 1 else ""
            correct_str = f"{data['correct_count']}/{data['total_count']}"
            acc_str = f"{data['accuracy_pct']:.1f}%"

            line = f"{rank:<5} | {clean_name:<32} | {data['score']:<8.3f} | {correct_str:<10} | {acc_str:<10} | {status}"
            print_line = f"{rank:<5} | {cfg_name:<32} | {data['score']:<8.3f} | {correct_str:<10} | {acc_str:<10} | {status}"

            print(print_line)
            f.write(line + "\n")

    print(f"\n[Saved leaderboard log to: {log_filename}]")

    return sorted_configs


# Main
if __name__ == "__main__":
    leaderboard = evaluate_coherent_states(
        d=D,
        p=P,
        shots=SHOTS,
        omega=OMEGA,
        runs_per_state=5
    )