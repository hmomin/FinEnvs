import numpy as np
import torch
import torch.nn as nn
from ...base_object import BaseObject
from ...device_utils import set_device
from typing import Tuple


class ParallelMLP(BaseObject):
    @torch.jit.export
    def __init__(
        self,
        num_envs: int,
        num_eval_envs: int,
        shape: tuple,
        learning_rate: float = 0.01,
        noise_std_dev: float = 0.02,
        l2_coefficient: float = 0.005,
        layer_activation=nn.Tanh,
        output_activation=nn.Tanh,
        device_id: int = 0,
    ):
        self.device = set_device(device_id)
        num_training_envs = num_envs - num_eval_envs
        assert (num_training_envs) % 2 == 0 and (num_training_envs) > 0, (
            f"The number of training environments ({num_envs} - {num_eval_envs}) "
            + "must be positive and even for mirrored sampling."
        )
        self.num_envs = num_envs
        self.num_training_envs = num_training_envs
        self.num_eval_envs = num_eval_envs
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

    @torch.jit.export
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

    @torch.jit.export
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

    @torch.jit.export
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
        self.add_action_noise(outputs, 0.01)
        return outputs

    @torch.jit.export
    def check_inputs(self, inputs: torch.Tensor) -> None:
        num_observations = self.shape[0]
        if inputs.shape != (self.num_envs, num_observations):
            raise Exception(
                f"inputs to ParallelMLP must have shape {(self.num_envs, num_observations)}"
            )

    @torch.jit.export
    def add_action_noise(self, actions: torch.Tensor, std_dev: float) -> None:
        action_noise = torch.normal(0, std_dev, actions.shape, device=self.device)
        action_noise[-self.num_eval_envs :, :] = 0
        actions.add_(action_noise)

    @torch.jit.export
    def perturb_parameters(self):
        for weight_layer, bias_layer in zip(self.weight_layers, self.bias_layers):
            new_weights = weight_layer.repeat((self.num_envs, 1, 1))
            new_biases = bias_layer.repeat((self.num_envs, 1, 1))
            positive_weight_perturbation = torch.normal(
                0,
                self.noise_std_dev,
                (self.num_training_envs // 2, *weight_layer.shape),
                device=self.device,
            )
            positive_bias_perturbation = torch.normal(
                0,
                self.noise_std_dev,
                (self.num_training_envs // 2, *bias_layer.shape),
                device=self.device,
            )
            negative_weight_perturbation = -positive_weight_perturbation
            negative_bias_perturbation = -positive_bias_perturbation
            zero_weight_perturbation = torch.zeros(
                (self.num_eval_envs, *weight_layer.shape), device=self.device
            )
            zero_bias_perturbation = torch.zeros(
                (self.num_eval_envs, *bias_layer.shape), device=self.device
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

    @torch.jit.export
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

    @torch.jit.export
    def update_parameters(self, fitnesses: torch.Tensor):
        self.adam_timestep += 1
        positive_perturbation_fitnesses = fitnesses[0 : self.num_training_envs // 2]
        negative_perturbation_fitnesses = fitnesses[
            self.num_training_envs // 2 : self.num_training_envs
        ]
        diffed_fitnesses = (
            positive_perturbation_fitnesses - negative_perturbation_fitnesses
        )
        expanded_fitnesses = diffed_fitnesses.unsqueeze(1).unsqueeze(1)
        for (
            idx,
            (
                weight_layer,
                perturbed_weight_layer,
                bias_layer,
                perturbed_bias_layer,
            ),
        ) in enumerate(
            zip(
                self.weight_layers,
                self.perturbed_weights,
                self.bias_layers,
                self.perturbed_biases,
            )
        ):
            positive_weight_perturbations = perturbed_weight_layer[
                0 : self.num_training_envs // 2, :, :
            ]
            positive_bias_perturbations = perturbed_bias_layer[
                0 : self.num_training_envs // 2, :, :
            ]
            total_weight_grad = torch.mul(
                expanded_fitnesses, positive_weight_perturbations
            )
            total_bias_grad = torch.mul(expanded_fitnesses, positive_bias_perturbations)
            mean_weight_grad = torch.mean(total_weight_grad, 0) / self.noise_std_dev
            mean_bias_grad = torch.mean(total_bias_grad, 0) / self.noise_std_dev
            mean_weight_grad -= self.l2_coefficient * weight_layer
            mean_bias_grad -= self.l2_coefficient * bias_layer
            self.adam_update(
                (weight_layer, bias_layer), (mean_weight_grad, mean_bias_grad), idx
            )
        self.perturbed_weights = []
        self.perturbed_biases = []

    @torch.jit.export
    def get_l2_norm(self) -> float:
        l2_norm = 0
        for (weight_layer, bias_layer,) in zip(
            self.weight_layers,
            self.bias_layers,
        ):
            l2_norm += weight_layer.square().sum() + bias_layer.square().sum()
        l2_norm = l2_norm.sqrt().item()
        return l2_norm

    @torch.jit.export
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
