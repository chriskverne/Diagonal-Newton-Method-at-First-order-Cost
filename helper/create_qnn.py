import pennylane as qml
def create_qnn(n_layers, n_qubits):
    dev = qml.device('default.qubit', wires=n_qubits)

    #@qml.qnode(dev)
    @qml.qnode(dev, interface="autograd", diff_method="parameter-shift", max_diff=2)
    def circuit(inputs, params, num_meas):
        for i in range(len(inputs)):
            qml.RX(inputs[i], wires=i)

        for layer in range(n_layers):
            # Rotational Gates
            for qubit in range(n_qubits):
                qml.RX(params[layer][qubit][0], wires=qubit)
                qml.RY(params[layer][qubit][1], wires=qubit)

            for qubit in range(n_qubits):
                next_qubit = (qubit + 1) % n_qubits
                qml.CNOT(wires=[qubit, next_qubit])

        return qml.probs(wires=range(num_meas))

    return circuit

