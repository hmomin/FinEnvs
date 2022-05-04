import numpy as np
import torch
import torch.nn as nn
from ...device_utils import set_device


class ParallelMLP(object):
    def __init__(
        self,
        num_envs: int,
        shape: tuple,
        learning_rate: float = 3e-4,
        noise_std_dev: float = 1.0,
        layer_activation=nn.ELU,
        output_activation=nn.Tanh,
        device_id: int = 0,
    ):
        self.device = set_device(device_id)
        self.num_envs = num_envs
        self.shape = shape
        self.learning_rate = learning_rate
        self.noise_std_dev = noise_std_dev
        self.weight_layers: "list[torch.Tensor]" = []
        self.perturbed_weights: "list[torch.Tensor]" = []
        self.bias_layers: "list[torch.Tensor]" = []
        self.perturbed_biases: "list[torch.Tensor]" = []
        self.activation_functions: "list[type]" = []
        self.create_layers(layer_activation, output_activation)

    def create_layers(self, layer_activation: type, output_activation: type) -> None:
        for i, current_size in enumerate(self.shape):
            if i == len(self.shape) - 1:
                break
            next_size = self.shape[i + 1]
            weight_shape = (current_size, next_size)
            bias_shape = (1, next_size)
            weight_layer = torch.normal(
                0, np.sqrt(2 / current_size), weight_shape, device=self.device
            )
            bias_layer = torch.normal(
                0, np.sqrt(2 / current_size), bias_shape, device=self.device
            )
            self.weight_layers.append(weight_layer)
            self.bias_layers.append(bias_layer)
            final_activation = i == len(self.shape) - 2
            activation_function = (
                output_activation() if final_activation else layer_activation()
            )
            self.activation_functions.append(activation_function)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.check_inputs(inputs)
        computation = inputs.unsqueeze(1)
        for weight_layer, bias_layer, activation in zip(
            self.perturbed_weights, self.perturbed_biases, self.activation_functions
        ):
            computation = torch.matmul(computation, weight_layer)
            computation = torch.add(computation, bias_layer)
            computation = activation(computation)
        outputs = torch.squeeze(computation, 1)
        return outputs

    def check_inputs(self, inputs: torch.Tensor) -> None:
        num_observations = self.shape[0]
        if inputs.shape != (self.num_envs, num_observations):
            raise Exception(
                f"inputs to ParallelMLP must have shape {(self.num_envs, num_observations)}"
            )

    def perturb_parameters(self):
        for weight_layer, bias_layer in zip(self.weight_layers, self.bias_layers):
            new_weights = weight_layer.repeat((self.num_envs, 1, 1))
            new_biases = bias_layer.repeat((self.num_envs, 1, 1))
            weight_perturbation = torch.normal(
                0, self.noise_std_dev, new_weights.shape, device=self.device
            )
            bias_perturbation = torch.normal(
                0, self.noise_std_dev, new_biases.shape, device=self.device
            )
            new_weights.add_(weight_perturbation)
            new_biases.add_(bias_perturbation)
            self.perturbed_weights.append(new_weights)
            self.perturbed_biases.append(new_biases)

    def reconstruct_perturbations(self):
        # transform the perturbed parameters to just the perturbations by subtracting
        # the original parameters
        for (
            weight_layer,
            perturbed_weight_layer,
            bias_layer,
            perturbed_bias_layer,
        ) in zip(
            self.weight_layers,
            self.perturbed_weights,
            self.bias_layers,
            self.perturbed_biases,
        ):
            perturbed_weight_layer.sub_(weight_layer.repeat((self.num_envs, 1, 1)))
            perturbed_bias_layer.sub_(bias_layer.repeat((self.num_envs, 1, 1)))

    def update_parameters(self, fitnesses: torch.Tensor):
        expanded_fitnesses = fitnesses.unsqueeze(1).unsqueeze(1)
        for (
            weight_layer,
            perturbed_weight_layer,
            bias_layer,
            perturbed_bias_layer,
        ) in zip(
            self.weight_layers,
            self.perturbed_weights,
            self.bias_layers,
            self.perturbed_biases,
        ):
            total_weight_grad = torch.mul(expanded_fitnesses, perturbed_weight_layer)
            total_bias_grad = torch.mul(expanded_fitnesses, perturbed_bias_layer)
            weight_grad = (
                self.learning_rate
                / self.noise_std_dev
                * torch.mean(total_weight_grad, 0)
            )
            bias_grad = (
                self.learning_rate / self.noise_std_dev * torch.mean(total_bias_grad, 0)
            )
            weight_layer.add_(weight_grad)
            bias_layer.add_(bias_grad)
        self.perturbed_weights = []
        self.perturbed_biases = []
