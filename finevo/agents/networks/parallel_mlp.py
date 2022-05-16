from logging import exception
import numpy as np
import torch
import torch.nn as nn
from ...base_object import BaseObject
from ...device_utils import set_device
from typing import Tuple


class ParallelMLP(BaseObject):
    def __init__(
        self,
        num_envs: int,
        shape: tuple,
        learning_rate: float = 0.01,
        noise_std_dev: float = 0.02,
        l2_coefficient: float = 0.005,
        layer_activation=nn.ELU,
        output_activation=nn.Tanh,
        device_id: int = 0,
    ):
        self.device = set_device(device_id)
        assert num_envs % 2 != 0, (
            f"num_envs ({num_envs}) must be odd (2n + 1): "
            + "2n environments for mirrored sampling and "
            + "1 environment for evaluation."
        )
        self.num_envs = num_envs
        self.shape = shape
        self.learning_rate = learning_rate
        self.noise_std_dev = noise_std_dev
        self.l2_coefficient = l2_coefficient
        self.weight_layers: "list[torch.Tensor]" = []
        self.perturbed_weights: "list[torch.Tensor]" = []
        self.bias_layers: "list[torch.Tensor]" = []
        self.perturbed_biases: "list[torch.Tensor]" = []
        self.activation_functions: "list[type]" = []
        self.create_layers(layer_activation, output_activation)
        self.set_adam_parameters()

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

    def set_adam_parameters(self) -> None:
        self.adam_timestep = 0
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.first_moment_weights: "list[torch.Tensor]" = []
        self.first_moment_biases: "list[torch.Tensor]" = []
        self.second_moment_weights: "list[torch.Tensor]" = []
        self.second_moment_biases: "list[torch.Tensor]" = []
        for weight_layer, bias_layer in zip(self.weight_layers, self.bias_layers):
            moment_weight_layer = torch.zeros(weight_layer.shape, device=self.device)
            moment_bias_layer = torch.zeros(bias_layer.shape, device=self.device)
            self.first_moment_weights.append(moment_weight_layer)
            self.first_moment_biases.append(moment_bias_layer)
            self.second_moment_weights.append(moment_weight_layer.clone())
            self.second_moment_biases.append(moment_bias_layer.clone())

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
            positive_weight_perturbation = torch.normal(
                0,
                self.noise_std_dev,
                (self.num_envs // 2, *weight_layer.shape),
                device=self.device,
            )
            positive_bias_perturbation = torch.normal(
                0,
                self.noise_std_dev,
                (self.num_envs // 2, *bias_layer.shape),
                device=self.device,
            )
            negative_weight_perturbation = -positive_weight_perturbation
            negative_bias_perturbation = -positive_bias_perturbation
            zero_weight_perturbation = torch.zeros(
                (1, *weight_layer.shape), device=self.device
            )
            zero_bias_perturbation = torch.zeros(
                (1, *bias_layer.shape), device=self.device
            )
            weight_perturbation = torch.cat(
                [
                    positive_weight_perturbation,
                    negative_weight_perturbation,
                    zero_weight_perturbation,
                ],
                dim=0,
            )
            bias_perturbation = torch.cat(
                [
                    positive_bias_perturbation,
                    negative_bias_perturbation,
                    zero_bias_perturbation,
                ],
                dim=0,
            )
            new_weights.add_(weight_perturbation)
            new_biases.add_(bias_perturbation)
            self.perturbed_weights.append(new_weights)
            self.perturbed_biases.append(new_biases)

    def get_l2_loss(self):
        l2_loss = torch.zeros((self.num_envs,), device=self.device)
        for (weight_layer, bias_layer) in zip(
            self.perturbed_weights,
            self.perturbed_biases,
        ):
            l2_loss += torch.square(weight_layer).sum(-1).sum(-1)
            l2_loss += torch.square(bias_layer).sum(-1).sum(-1)
        return l2_loss

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

    def update_parameters(self, fitnesses: torch.Tensor, l2_loss: torch.Tensor):
        self.adam_timestep += 1
        transformed_fitnesses = (
            self.compute_centered_rank(fitnesses) - self.l2_coefficient * l2_loss
        )
        positive_perturbation_fitnesses = transformed_fitnesses[0 : self.num_envs // 2]
        negative_perturbation_fitnesses = transformed_fitnesses[self.num_envs // 2 : -1]
        diffed_fitnesses = (
            positive_perturbation_fitnesses - negative_perturbation_fitnesses
        )
        expanded_fitnesses = diffed_fitnesses.unsqueeze(1).unsqueeze(1)
        for idx, (
            weight_layer,
            perturbed_weight_layer,
            bias_layer,
            perturbed_bias_layer,
        ) in enumerate(
            zip(
                self.weight_layers,
                self.perturbed_weights,
                self.bias_layers,
                self.perturbed_biases,
            )
        ):
            positive_weight_perturbations = perturbed_weight_layer[
                0 : self.num_envs // 2, :, :
            ]
            positive_bias_perturbations = perturbed_bias_layer[
                0 : self.num_envs // 2, :, :
            ]
            total_weight_grad = torch.mul(
                expanded_fitnesses, positive_weight_perturbations
            )
            total_bias_grad = torch.mul(expanded_fitnesses, positive_bias_perturbations)
            mean_weight_grad = torch.mean(total_weight_grad, 0) / self.noise_std_dev
            mean_bias_grad = torch.mean(total_bias_grad, 0) / self.noise_std_dev
            self.adam_update(
                (weight_layer, bias_layer), (mean_weight_grad, mean_bias_grad), idx
            )
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

    def adam_update(
        self, layers: Tuple[torch.Tensor], grads: Tuple[torch.Tensor], idx: int
    ):
        assert (
            len(layers) == 2
        ), "layers passed to adam_update must be of the form (weight_layer, bias_layer)"
        assert (
            len(grads) == 2
        ), "grads passed to adam_update must be of the form (weight_grad, bias_grad)"
        t, alpha, beta_1, beta_2 = (
            self.adam_timestep,
            self.learning_rate,
            self.beta_1,
            self.beta_2,
        )
        weight_layer: torch.Tensor = layers[0]
        bias_layer: torch.Tensor = layers[1]
        mean_weight_grad: torch.Tensor = grads[0]
        mean_bias_grad: torch.Tensor = grads[1]
        self.first_moment_weights[idx] = (
            beta_1 * self.first_moment_weights[idx] + (1 - beta_1) * mean_weight_grad
        )
        self.first_moment_biases[idx] = (
            beta_1 * self.first_moment_biases[idx] + (1 - beta_1) * mean_bias_grad
        )
        self.second_moment_weights[idx] = beta_2 * self.second_moment_weights[idx] + (
            1 - beta_2
        ) * torch.square(mean_weight_grad)
        self.second_moment_biases[idx] = beta_2 * self.second_moment_biases[idx] + (
            1 - beta_2
        ) * torch.square(mean_bias_grad)
        alpha_t = np.sqrt(1 - beta_2**t) / (1 - beta_1**t) * alpha
        weight_grad = alpha_t * torch.div(
            self.first_moment_weights[idx],
            torch.sqrt(self.second_moment_weights[idx]) + 10**-8,
        )
        bias_grad = alpha_t * torch.div(
            self.first_moment_biases[idx],
            torch.sqrt(self.second_moment_biases[idx]) + 10**-8,
        )
        weight_layer.add_(weight_grad)
        bias_layer.add_(bias_grad)
