import numpy as np
import torch
import torch.nn as nn
from ...base_object import BaseObject
from ...device_utils import set_device


class ParallelMLP(BaseObject):
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
        assert (
            num_envs % 2 == 0
        ), f"num_envs ({num_envs}) must be divisible by 2 for mirror sampling."
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
            # use the first environment for parameter evaluation
            weight_perturbation[0, :, :] = 0
            bias_perturbation[0, :, :] = 0
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
        transformed_fitnesses = self.compute_centered_rank(fitnesses)
        expanded_fitnesses = transformed_fitnesses.unsqueeze(1).unsqueeze(1)
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

    def compute_ranks(self, values: torch.Tensor):
        values_1d = values.squeeze()
        ranks = torch.empty(values_1d.shape[0], device=self.device)
        sort_indices = values_1d.argsort()
        linear_ranks = torch.arange(
            0, values.shape[0], dtype=torch.float32, device=self.device
        )
        ranks[sort_indices] = linear_ranks
        ranks = ranks.reshape(values.shape)
        return ranks

    def compute_centered_rank(self, values: torch.Tensor) -> torch.Tensor:
        ranks = self.compute_ranks(values)
        N = len(ranks)
        centered_ranks = ranks / (N - 1) - 0.5
        return centered_ranks

    # def update_parameters_with_max(self, fitnesses: torch.Tensor) -> None:
    #     max_index = torch.argmax(fitnesses)
    #     for (
    #         weight_layer,
    #         perturbed_weight_layer,
    #         bias_layer,
    #         perturbed_bias_layer,
    #     ) in zip(
    #         self.weight_layers,
    #         self.perturbed_weights,
    #         self.bias_layers,
    #         self.perturbed_biases,
    #     ):
    #         weight_layer[:] = perturbed_weight_layer[max_index, :, :]
    #         bias_layer[:] = perturbed_bias_layer[max_index, :, :]
    #     self.perturbed_weights = []
    #     self.perturbed_biases = []
