"""Quantum device setup and batched variational forward circuit."""

import pennylane as qp
import torch

from scripts.constants import DEFAULT_NOISE_RATE, DEFAULT_WEIGHT_INIT_EPS, DEFAULT_REUPLOAD


def create_quantum_device(num_qubits, backend="default.mixed"):
    """
    Create a quantum device with the given number of qubits and backend.
    """
    return qp.device(backend, wires=num_qubits)


def initialize_zero_weights(n_layers, n_wires, device):
    """
    Initialize weights for the variational circuit using identity rotations to be barren-plateau-safe.
    """
    shape = qp.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires)
    return torch.nn.Parameter(torch.zeros(shape, device=device))


def initialize_random_weights(n_layers, n_wires, device, eps=DEFAULT_WEIGHT_INIT_EPS, seed=None):
    """
    Initialize weights for the variational circuit using random identity rotations.
    """
    shape = qp.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires)
    gen = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    theta = eps * torch.randn(shape, generator=gen, device=device)
    return torch.nn.Parameter(theta)


def build_forward_circuit(dev, num_qubits, num_layers, noise_rate=DEFAULT_NOISE_RATE, reupload=DEFAULT_REUPLOAD):
    """
    Build the QS-Net forward QNode: encode, variational layers, depolarizing noise, readout.

    Returns a callable `forward_circuit() -> (z, rho)` where:
    - z is a list of num_qubits tensors, each shaped (batch,)
    - rho is a tensor shaped (batch, dim, dim)
    """

    @qp.qnode(dev, interface="torch", diff_method="backprop")
    def forward_circuit(X, theta):
        """
        Circuit for encoding, variational, adding noise, and readout.
        """
        for layer in range(num_layers):
            if reupload or layer == 0: # if data reupload is enabled, or it's the first layer, embed the data
                qp.AngleEmbedding(X, wires=range(num_qubits), rotation="Y")
            qp.StronglyEntanglingLayers(theta[layer : layer + 1], wires=range(num_qubits))
            for w in range(num_qubits):
                qp.DepolarizingChannel(noise_rate, wires=w)
        z = [qp.expval(qp.PauliZ(w)) for w in range(num_qubits)]
        rho = qp.density_matrix(wires=range(num_qubits))
        return z, rho

    return forward_circuit