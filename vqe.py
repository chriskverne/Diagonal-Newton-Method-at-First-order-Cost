import numpy as np
from dataclasses import dataclass

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Statevector


# Use Aer Estimator primitive (fast, supports sampling)
try:
    from qiskit_aer.primitives import Estimator as AerEstimator
    Estimator = AerEstimator
except Exception:
    # Fallback to reference Estimator (slower but OK)
    from qiskit.primitives import Estimator  # type: ignore


# -----------------------------
# Hamiltonian builders (benchmarks)
# -----------------------------

def tfim_hamiltonian(n_qubits: int, h: float = 1.0, open_boundary: bool = True) -> SparsePauliOp:
    """Transverse-Field Ising Model (TFIM)
    H = -sum_{i} Z_i Z_{i+1} - h * sum_i X_i
    Args:
        n_qubits: number of qubits (>=2)
        h: transverse field strength
        open_boundary: if False, uses periodic boundary conditions
    Returns:
        SparsePauliOp encoding the Hamiltonian
    """
    paulis = []
    coeffs = []

    # Z Z interactions
    last = n_qubits if open_boundary else n_qubits + 1
    for i in range(n_qubits - 1 + (0 if open_boundary else 1)):
        a = i
        b = (i + 1) % n_qubits
        zstring = ["I"] * n_qubits
        zstring[a] = "Z"
        zstring[b] = "Z"
        paulis.append("".join(reversed(zstring)))  # Qiskit uses little-endian order in strings
        coeffs.append(-1.0)

    # Transverse field X
    for i in range(n_qubits):
        xstring = ["I"] * n_qubits
        xstring[i] = "X"
        paulis.append("".join(reversed(xstring)))
        coeffs.append(-h)

    return SparsePauliOp.from_list([(p, c) for p, c in zip(paulis, coeffs)])


def heisenberg_xxz_hamiltonian(n_qubits: int, delta: float = 1.0, open_boundary: bool = True) -> SparsePauliOp:
    """XXZ Heisenberg chain: H = sum_i (X_i X_{i+1} + Y_i Y_{i+1} + delta * Z_i Z_{i+1})
    Widely used as a second benchmark.
    """
    paulis = []
    coeffs = []
    links = range(n_qubits - 1) if open_boundary else range(n_qubits)
    for i in links:
        a = i
        b = (i + 1) % n_qubits
        for term, coef in (("X", 1.0), ("Y", 1.0), ("Z", delta)):
            s = ["I"] * n_qubits
            s[a] = term
            s[b] = term
            paulis.append("".join(reversed(s)))
            coeffs.append(coef)
    return SparsePauliOp.from_list([(p, c) for p, c in zip(paulis, coeffs)])

# -----------------------------
# Ansatz
# -----------------------------

def build_ansatz(n_qubits: int, n_layers: int):
    """Hardware-efficient ansatz: input-agnostic, Rx-Rz layers + CZ ring.
    Returns circuit and ParameterVector θ of length n_layers * (2*n_qubits).
    """
    qc = QuantumCircuit(n_qubits)
    thetas = ParameterVector("θ", n_layers * 2 * n_qubits)
    idx = 0
    for _ in range(n_layers):
        # single-qubit rotations
        for q in range(n_qubits):
            qc.rx(thetas[idx], q); idx += 1
            qc.rz(thetas[idx], q); idx += 1

        # entangling CZ ring (linear chain for open boundary)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

    return qc, thetas

# -----------------------------
# VQE core
# -----------------------------

@dataclass
class VQEConfig:
    n_qubits: int = 4
    n_layers: int = 2
    lr: float = 0.1
    epochs: int = 150
    seed: int = 7
    h: float = 1.0  # TFIM field strength
    shots: int | None = None  # None -> exact (default for AerEstimator), or set an int for sampling
    model: str = "tfim"  # or "xxz"
    open_boundary: bool = True
    mode: str = None

class VQE:
    def __init__(self, cfg: VQEConfig):
        #assert 2 <= cfg.n_qubits <= 4, "This script targets 2-4 qubits as requested."
        self.cfg = cfg
        self.estimator = Estimator(run_options={"shots": cfg.shots} if cfg.shots else None)
        self.ansatz, self.params = build_ansatz(cfg.n_qubits, cfg.n_layers)
        rng = np.random.default_rng(cfg.seed)
        self.theta = rng.uniform(low=-2*np.pi, high=2*np.pi, size=len(self.params))

        # Adam / momentum
        self.momentum = np.zeros_like(self.theta)
        self.velocity = np.zeros_like(self.theta)
        self.beta = 0.9
        self.beta2 = 0.999
        self.t = 0
        self.eps = 1e-8

        # SPSA
        self.spsa_a = 0.2        # base learning rate scale
        self.spsa_c = 0.1        # base perturbation size
        self.spsa_alpha = 0.602  # exponent for learning rate decay
        self.spsa_gamma = 0.101  # exponent for perturbation decay
        self.spsa_A_frac = 0.1   # fraction of total epochs for stability offset
        self.rng = np.random.default_rng(cfg.seed)

        # Data tracking
        self.fp = 0

        # Hamiltonian
        if cfg.model == "tfim":
            self.H = tfim_hamiltonian(cfg.n_qubits, h=cfg.h, open_boundary=cfg.open_boundary)
        elif cfg.model == "xxz":
            self.H = heisenberg_xxz_hamiltonian(cfg.n_qubits, delta=1.0, open_boundary=cfg.open_boundary)
        else:
            raise ValueError("Unknown model: choose 'tfim' or 'xxz'")

    def find_GSE(self) -> float:
        """
        Return the exact ground-state energy (lowest eigenvalue) of self.H.
        For 2–4 qubits we can safely form the dense matrix and diagonalize.
        """
        # Dense Hamiltonian matrix (2^n × 2^n), complex Hermitian
        Hmat = self.H.to_matrix()
        # Symmetrize for numerical stability (should already be Hermitian)
        Hmat = 0.5 * (Hmat + Hmat.conj().T)
        # Lowest eigenvalue is the exact GSE
        E0 = np.linalg.eigvalsh(Hmat).min().real
        return float(E0)

    def energy(self, theta: np.ndarray) -> float:
        bound = self.ansatz.assign_parameters({p: float(v) for p, v in zip(self.params, theta)})
        res = self.estimator.run([bound], [self.H]).result()
        return float(res.values[0])

    def compute_gradients(self, theta: np.ndarray):
        grads = np.zeros_like(theta)
        shift = np.pi / 2
        
        for i in range(theta.size):
            t_plus = theta.copy()
            t_plus[i] += shift
            t_minus = theta.copy()
            t_minus[i] -= shift
            e_plus = self.energy(t_plus)
            e_minus = self.energy(t_minus)
            grads[i] = 0.5 * (e_plus - e_minus)

            self.fp += 2
        
        return grads
    
    def compute_gradient_and_curvature(self, theta: np.ndarray, E:float):
        grads = np.zeros_like(theta)
        curvature = np.zeros_like(theta)
        shift = np.pi / 2
        
        for i in range(theta.size):
            t_plus = theta.copy()
            t_plus[i] += shift
            t_minus = theta.copy()
            t_minus[i] -= shift
            e_plus = self.energy(t_plus)
            e_minus = self.energy(t_minus)
            grads[i] = 0.5 * (e_plus - e_minus)
            curvature[i] = 0.5*((e_plus + e_minus) - 2.0 * E)

            self.fp += 2

        return grads, curvature
    
    def curvature_step(self, theta: np.ndarray, E:float) -> np.ndarray:
        grads, curvature = self.compute_gradient_and_curvature(theta, E)
        step_size = np.zeros_like(theta)

        self.t += 1


        scaled_curvature = np.maximum(curvature, 1e-6)
        step_size = grads / (scaled_curvature)
            # step_size = grads/(np.abs(curvature) + 1e-8)
        return step_size
    
    def curvature_step_momentum(self, theta: np.ndarray, E:float) -> np.ndarray:
        grads, curvature = self.compute_gradient_and_curvature(theta, E)
        step_size = np.zeros_like(theta)

        self.t += 1

        self.momentum = self.beta*self.momentum + (1-self.beta)*grads
        m_hat = self.momentum / (1 - self.beta ** self.t)
        scaled_curvature = np.maximum(curvature, 1e-6)
        step_size = m_hat / (scaled_curvature)
        # step_size = m_hat/(np.abs(curvature) + 1e-8)
        # Could also add dampin factor to fix negative curvature problem
        # step = grads / (curvature + \lambda) where lambda is some constant would work
        # if curvature > 0 make lambda 0, if curvature is negative make lambda larger
        
        return step_size

    def adam_step(self, theta: np.ndarray):
        grads = self.compute_gradients(theta)

        self.t += 1

        # m_t and v_t
        self.momentum = self.beta * self.momentum + (1 - self.beta) * grads
        self.velocity = self.beta2 * self.velocity + (1 - self.beta2) * (grads * grads)

        # bias corrected
        m_hat = self.momentum / (1 - self.beta ** self.t)
        v_hat = self.velocity / (1 - self.beta2 ** self.t)

        return m_hat / (np.sqrt(v_hat) + self.eps)
   
    def spsa_step(self, theta: np.ndarray) -> np.ndarray:
        """Simplest SPSA implementation (Spall, 1992)."""
        self.t += 1
        k = self.t

        a0 = self.spsa_a
        c0 = self.spsa_c
        alpha = self.spsa_alpha
        gamma = self.spsa_gamma
        A = self.spsa_A_frac * self.cfg.epochs

        a_k = a0 / ((k + A) ** alpha)
        c_k = c0 / (k ** gamma)

        d = len(theta)
        delta = self.rng.choice([-1.0, 1.0], size=d)

        e_plus = self.energy(theta + c_k * delta)
        e_minus = self.energy(theta - c_k * delta)

        self.fp += 2

        g_hat = (e_plus - e_minus) / (2.0 * c_k) * delta
        #theta_new = theta - a_k * g_hat
        return a_k * g_hat
    
    def QNG_step(self, theta:np.ndarray):
        
        return 0


    def train(self):
        history = []
        for epoch in range(1, self.cfg.epochs + 1):
            E = self.energy(self.theta)
            
            mode = self.cfg.mode
            if mode == 'curv':
                step = self.curvature_step(self.theta, E)
            elif mode == 'curv_mom':
                step = self.curvature_step_momentum(self.theta, E)
            elif mode == 'adam':
                step = self.adam_step(self.theta)
            elif mode == 'spsa':
                self.cfg.lr = 1
                step = self.spsa_step(self.theta)
            else:
                print("Unknown mode")

            self.theta = self.theta - self.cfg.lr * step
            history.append(E)
            if epoch % 10 == 0 or epoch == 1:
                print(f"Step {epoch:3d} | Energy: {E:.6f} | ||Step||: {np.linalg.norm(step):.4e} | FP : {self.fp}")
        final_E = self.energy(self.theta)
        print(f"\nConverged Energy: {final_E:.8f}")
        return final_E, self.theta, np.array(history)


if __name__ == "__main__":
    # Hyper params
    n_qubits = 6
    n_layers = 1
    model = "tfim"

    # What to test
    modes = ['curv','curv_mom', 'adam', 'spsa']
    lrs = [1,1,0.3,1]
    epoch_combos = [50,50,50, 200]

    for i in range(len(modes)):
        mode = modes[i]
        lr = lrs[i]
        epochs =  epoch_combos[i]

        print(f'Training {n_qubits}q, {n_layers}l VQE with: {mode}, lr: {lr}, Num Epochs: {epochs}')

        cfg = VQEConfig(n_qubits=n_qubits, n_layers=n_layers, lr=lr, epochs=epochs, seed=42, h=1.0, shots=None, 
                        model=model, mode=mode)
        
        vqe = VQE(cfg)
        print(f'Target energy: {vqe.find_GSE()}')
        print(f'Initial energy: {vqe.energy(vqe.theta)}')
        vqe.train()