import pennylane as qml
import pennylane.numpy as np
import numpy as onp

# data
x = onp.linspace(0, 10, 200)
y = onp.sin(x**3 + x)

# model
n_layers, n_qubits = 2, 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def model(xi, params):
    qml.H(0)
    qml.RY(xi, 0)
    for l in range(n_layers):
        qml.RY(params[l,0,0], 0); qml.RZ(params[l,0,1], 0)
    return qml.expval(qml.PauliZ(0))

def loss(p, xi, yi):  # scalar loss per sample
    return (model(xi, p) - yi) ** 2

# training
params = 0.01 * np.random.randn(n_layers, n_qubits, 2)
for i, (xi, yi) in enumerate(zip(x, y), 1):
    g = qml.grad(loss)(params, xi, yi)
    params = params - 0.1 * g
    if i == 1 or i % 10 == 0:
        print(f"Step {i:3d} | Loss: {float(loss(params, xi, yi)):.6f}")

print("\nExample:", float(model(x[0], params)), y[0])
