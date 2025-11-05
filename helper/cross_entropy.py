import pennylane.numpy as pnp
def cross_entropy_loss(output_distribution, true):
    epsilon = 1e-10
    pred = output_distribution[true]
    return -pnp.log(pred + epsilon)

def cross_entropy_grad(output_distribution, true):
    epsilon = 1e-10
    return -1.0 / (output_distribution[true] + epsilon)
    