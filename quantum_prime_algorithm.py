from __future__ import annotations
import math
import cmath
import time
from typing import Tuple, Dict, List
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate, Initialize
from qiskit.quantum_info import Statevector

def next_power_of_two(n: int) -> int:
    """Return the next power of two greater than or equal to n."""
    return 1 << (n - 1).bit_length()

def get_n_vals(d: int) -> np.ndarray:
    """Return n values in D = [2, 2(d-1)] for given d."""
    return np.arange(2, 2*(d-1)+1)

def get_t_vals(p: int, omega: float) -> np.ndarray:
    """Return p+1 time values t in [0, π/(2ω)] for given p and ω."""
    half_period = math.pi / omega
    if p % 2 == 0:
        p += 1
    return np.linspace(0.0, half_period, p)

def q_from_d(d: int) -> int:
    """Compute number of qubits q for given d via d^2 = 2^q (Eq. (10))."""
    d_embed = next_power_of_two(d)
    return 2 * math.ceil(math.log2(d_embed))

def error_estimate(shots: int, n_val: np.ndarray, p: int, delta: float = 0.01) -> np.ndarray:
    """
    Estimate the error in the alpha values. Default delta is 0.01 for 99% confidence interval.
    Returns an array of error estimates for each n in n_val.
    """
    epsilon = np.sqrt(1/(2*shots)*np.log(2/delta))+0.00002
    return np.array([4*epsilon/(3*p)*np.sqrt(5*p-2) for n in n_val])

# -------- initial state and evolution --------

def prepare_uniform_product_state(d: int) -> Statevector:
    """
    |ψ(0)⟩ = (1/√d) Σ_{nA=1..d}|nA⟩ ⊗ (1/√d) Σ_{nB=1..d}|nB⟩
    embedded in d′=2^{⌈log₂ d⌉}. (Indexing: k=(nB-1)*d′+(nA-1))
    """
    d_embed = next_power_of_two(d)
    amp = np.zeros(d_embed * d_embed, dtype=complex)
    scale = 1.0 / d
    for nB in range(1, d + 1):
        for nA in range(1, d + 1):
            k = (nB - 1) * d_embed + (nA - 1)
            amp[k] = scale
    q = 2 * math.ceil(math.log2(d_embed))
    return Statevector(amp, dims=[2] * q)

def evolve_by_diagonal(d: int, omega: float, t: float, psi0: Statevector) -> Statevector:
    """Apply U(t)=exp(-i ω n_A n_B t)."""
    d_embed = next_power_of_two(d)
    nA = np.arange(1, d_embed + 1)
    nB = np.arange(1, d_embed + 1)
    phase = np.exp(-1j * omega * np.outer(nB, nA) * t).ravel()
    vec = psi0.data * phase
    return Statevector(vec, dims=psi0.dims())

# ----- U(t) operator via Walsh angles (Appendices B–D) -----

def precompute_walsh_coeffs(d: int) -> Dict[int, float]:
    """
    Precompute the static (time-independent) Walsh coefficients a_j0
    for hamming weight 1 and 2 terms (Appendix D).
    a_j(t) = a_j0 * (ω t).
    """
    d_embed = next_power_of_two(d)
    q = 2 * math.ceil(math.log2(d_embed))
    coeffs = {}

    half = q // 2
    d_half = 1 << half
    d_over_2 = d_half // 2

    # ---- weight 1 terms (h=1)
    for b in range(q):
        j = 1 << b
        if j <= d_over_2:
            a0 = (d_embed * (1 + d_embed)) / (8.0 * j)
        else:
            a0 = (d_embed**2 * (1 + d_embed)) / (8.0 * j)
        coeffs[j] = a0

    # ---- weight 2 terms (h=2)
    for i in range(half):
        l1 = 1 << i
        for jpos in range(half, q):
            l2 = 1 << jpos
            j_idx = l1 + l2
            a0 = -(d_embed**3) / (16.0 * l1 * l2)
            coeffs[j_idx] = a0

    return coeffs

def _apply_exp_i_a_wj(qc: QuantumCircuit, j: int, a: float, tol: float = 1e-12):
    """Implement e^{i a w_j} using the staircase identity (Appendix B)."""
    if abs(a) < tol:
        return
    bits = [b for b in range(qc.num_qubits) if (j >> b) & 1]
    if not bits:
        return
    msb = max(bits)
    controls = [b for b in bits if b != msb]
    for c in controls:
        qc.cx(c, msb)
    qc.rz(-2.0 * a, msb)

    for c in reversed(controls):
        qc.cx(c, msb)

def build_Ut_walsh(d: int, omega: float, t: float,
                   coeffs: Dict[int, float] | None = None) -> QuantumCircuit:
    """
    Build U(t)=exp(-i ω n_A n_B t) using sparse Walsh synthesis.
    Reuses precomputed coefficients if provided.
    """
    d_embed = next_power_of_two(d)
    q = 2 * math.ceil(math.log2(d_embed))
    qc = QuantumCircuit(q, name=f"U(t={t:.0f})")

    if abs(t) < 1e-12:
        return qc

    if coeffs is None:
        coeffs = precompute_walsh_coeffs(d)

    for j, a0 in coeffs.items():
        angle = a0 * omega * t
        _apply_exp_i_a_wj(qc, j, angle)

    return qc

def build_U_only(d: int, omega: float, t: float,
                evolution_method: str = "walsh",
                coeffs: Dict[int, float] | None = None) -> QuantumCircuit:
    """ 
    Build evolution circuit U(t) using either Walsh or Diagonal method. 
    Accepts precomputed coeffs for speed. 
    """
    d_embed = next_power_of_two(d)
    q = 2 * math.ceil(math.log2(d_embed))
    qc = QuantumCircuit(q, name=f"U_only(t={t:.4f})")

    if evolution_method == "walsh":
        qc.compose(build_Ut_walsh(d, omega, t, coeffs), inplace=True)
    elif evolution_method == "diagonal":
        phases = [
            cmath.exp(-1j * omega * ((k % d_embed) + 1) * ((k // d_embed) + 1) * t) for k
            in range(d_embed * d_embed)
        ]
        qc.append(DiagonalGate(phases), qc.qubits)
    else:
        raise ValueError("evolution_method must be 'walsh' or 'diagonal'")
    return qc

# -------- SWAP test --------
def prepare_swaptest_circuits(
        d: int, omega: float, t_vals: np.ndarray, init_exact_state: bool, use_walsh: bool
    ) -> List[QuantumCircuit]:
    """
    Prepare a list of SWAP-test circuits for given d, omega, and t values.
    Each circuit prepares the initial state, applies U(t), and performs the SWAP test.
    """
    evo = "walsh" if use_walsh else "diagonal"
    circuits: List[QuantumCircuit] = []
    for t in t_vals:
        qc, _ = build_swaptest_circuit(
            d, omega, float(t),
            evolution_method=evo,
            init_exact_state=init_exact_state
        )
        circuits.append(qc)
    return circuits

def build_swaptest_circuit(d: int, omega: float, t: float,
                           evolution_method: str = "walsh",
                           init_exact_state: bool = True) -> Tuple[QuantumCircuit, int]:
    """
    Build a SWAP-test circuit that:
      - prepares |ψ0> (exact embedded uniform state) on 2 copies via Initialize
        (if init_exact_state), otherwise uses H^{⊗q} as an approximation;
      - applies U(t) on each copy (Walsh or Diagonal);
      - performs controlled SWAP on subsystem A only; measures ancilla.
    """
    d_embed = next_power_of_two(d)
    q_single = 2 * math.ceil(math.log2(d_embed))
    qA = q_single // 2

    qc = QuantumCircuit(1 + 2 * q_single, 1)
    anc = 0
    copy1 = range(1, 1 + q_single)
    copy2 = range(1 + q_single, 1 + 2 * q_single)

    if init_exact_state and (next_power_of_two(d) != d):
        psi0 = prepare_uniform_product_state(d)
        init = Initialize(psi0.data)
        qc.append(init, list(copy1))
        qc.append(init, list(copy2))
    else:
        qc.h(list(copy1))
        qc.h(list(copy2))

    # Apply U(t) to both copies
    if evolution_method == "walsh":
        Ut = build_Ut_walsh(d, omega, t)
        Ut_instr = Ut.to_instruction()
        qc.append(Ut_instr, list(copy1))
        qc.append(Ut_instr, list(copy2))
    elif evolution_method == "diagonal":
        phases = [cmath.exp(-1j * omega * ((k % d_embed) + 1) * ((k // d_embed) + 1) * t)
                  for k in range(d_embed * d_embed)]
        Uc = QuantumCircuit(q_single)
        Uc.append(DiagonalGate(phases), Uc.qubits)
        qc.compose(Uc, qubits=copy1, inplace=True)
        qc.compose(Uc, qubits=copy2, inplace=True)
    else:
        raise ValueError("evolution_method must be 'walsh' or 'diagonal'.")

    # SWAP test on subsystem A (lower half of each copy)
    qc.h(anc)
    for i in range(qA):
        qc.cswap(anc, copy1[i], copy2[i])
    qc.h(anc)
    qc.measure(anc, 0)

    return qc, d_embed

# -------- Reduced purity and Fourier modes --------

def reduced_purity(state: Statevector) -> float:
    """γ_A = Tr(ρ_A^2)."""
    q = int(round(math.log2(state.dim)))
    half = q // 2
    D = 1 << half
    psi = state.data.reshape(D, D)
    rho_A = psi @ psi.conj().T
    return float(np.real(np.trace(rho_A @ rho_A)))

def simpson_integral(y: np.ndarray, dx: float) -> float:
    """Composite Simpson's rule over equispaced samples y[0..M-1].
    Returns ∫ y(x) dx approximated over the grid with spacing dx.
    """
    if len(y) <= 3:
        raise ValueError("Simpson's rule requires even number of samples > 3.")
    s = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2])
    return s * dx / 3.0

def fourier_modes(d: int, omega: float, t_vals: np.ndarray, gamma_vals: np.ndarray):
    """Compute alpha_n via Eq. (17): alpha_n = (2ω/π) ∫_{0}^{T/2} gamma_A(t) cos(n ω t) dt."""
    n_max = 2 * (d - 1)
    dt = t_vals[1] - t_vals[0]
    alpha = {}
    for n in range(2, n_max + 1):
        i = simpson_integral(gamma_vals * np.cos(n * omega * t_vals), dt)
        alpha[n] = float((2.0 * omega / math.pi) * i)
    return alpha

# ----- Analytic lower bound and classification (Sec. IV.5) -----

def analytic_Bn(n: int, d: int) -> float:
    """Analytic lower bound B_n for uniform initial state (Sec. IV.5, Regimes I & II).
    Regime I: 2 ≤ n ≤ d-1  ->  Bn = -8(d-1)/d^4 * n + (8d-8)/d^3
    Regime II: d ≤ n ≤ 2(d-1) -> Bn = 0
    Outside D = I ∪ II: return np.nan (not used for certification).
    """
    if 2 <= n <= d - 1:
        return (-8.0 * (d - 1) / (d**4)) * n + 8.0*(d - 1) / (d**3)
    if d <= n <= 2 * (d - 1):
        return 0.0
    return float("nan")

def analytic_Pn(n: int, d: int) -> float:
    """Calculates the B_n offset of the analytic bound P_n for semiprimes n=k^2:
    P_n = B_n + (4 / d^4) * (d^2 + n - 2 * d * sqrt(n))
    """
    return (4.0 / (d**4)) * (d**2 + n - 2.0 * d * np.sqrt(n))

def analytic_En(n: int, d: int) -> float:
    """Calculates the B_n offset of the analytic bound E_n for even semiprimes n=2*k:
    E_n = B_n + (8 / d^4) * (d^2 + n - d * (n / 2 + 2))
    """
    return (8.0 / (d**4)) * (d**2 + n - d * (n / 2.0 + 2.0))

def classify_alpha_against_Bn(alpha_n: Dict[int, float], d: int,
                              p: int, shots: int,
                              error: np.ndarray | None = None,
                              parity: bool = True):
    """Classify n in D := [2, 2(d-1)] as prime (alpha_n == B_n) vs composite (alpha_n > B_n),
    within absolute tolerance atol. Returns (primes, detailed_report).
    """
    primes, details = [], {}

    if error is None:
        error = error_estimate(shots, n_val=np.arange(2, 2 * (d - 1) + 1), p=p)

    for n, val in alpha_n.items():
        if n < 2 or n > 2*(d-1) or (parity and n & 1 == 0 and n != 2):
            continue
        Bn = analytic_Bn(n, d)

        ok = val <= Bn + error[n-2]
        if ok:
            primes.append(n)
        details[n] = {"alpha_n": float(val), "B_n": float(Bn), "delta": float(val - Bn),
                      "error": float(error[n-2]), "within_tol": bool(ok)}

    #print(f"Primes found: {primes}")
    return primes, details

def execute_classification(
    d: int,
    t_vals: np.ndarray,
    gamma_vals: np.ndarray,
    p: int | None = None,
    shots: int = 2000,
    error: np.ndarray | None = None,
    omega: float = 0.1,
    t_start: float | None = None,
    t_end: float | None = None,
    parity: bool = False,
    prints: bool = True):
    """
    Execute the prime classification of n in D = [2, 2(d-1)].
    Returns (alpha, primes, details, gamma_vals).
    """
    gamma0 = float(gamma_vals[0])
    kappa = 1.0 if gamma0 <= 0 else min(3.0, 1.0 / gamma0)
    gamma_vals = np.clip(gamma_vals * kappa, -1.0, 1.0)

    if error is None and p is not None:
        error = error_estimate(shots=shots, n_val=np.arange(2, 2 * (d - 1) + 1), p=p)

    t0 = time.time()
    alpha = fourier_modes(d, omega=omega, t_vals=t_vals, gamma_vals=gamma_vals)
    t1 = time.time()
    primes, details = classify_alpha_against_Bn(
        alpha, d, p if p is not None else len(t_vals) - 1, shots=shots, parity=parity)
    t2 = time.time()
    
    if prints:
        print(f"[time] Fourier computation: {t1 - t0:.4f} s, Classification: {t2 - t1:.4f} s")
        print(f"[result] Certified primes in [2, {2*(d-1)}] → ", primes)
        if t_start is not None and t_end is not None:
            print(f"[time] {(t_end - t_start):.4f} s")

    return alpha, primes, details, gamma_vals
