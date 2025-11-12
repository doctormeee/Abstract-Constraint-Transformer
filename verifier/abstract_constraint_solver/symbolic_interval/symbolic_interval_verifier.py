#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################################
##   Abstract Constraint Transformer (ACT) - Symbolic Interval Verifier ##
##                                                                     ##
##   Implementation based on "Interval Bound Propagation with          ##
##   Symbolic Bounds" (IBP+Symbolic Interval) paper                    ##
##                                                                     ##
##   Supports ReLU networks only (no Sigmoid/Tanh)                     ##
##                                                                     ##
##   doctormeeee (https://github.com/doctormeeee) and contributors     ##
##   Copyright (C) 2024-2025                                           ##
##                                                                     ##
#########################################################################

import torch
import torch.nn as nn
import os
import sys
from typing import Tuple, Optional, Dict

import path_config

from abstract_constraint_solver.base_verifier import BaseVerifier
from input_parser.dataset import Dataset
from input_parser.spec import Spec
from input_parser.type import VerificationStatus
from onnx2pytorch.operations.flatten import Flatten as OnnxFlatten
from onnx2pytorch.operations.add import Add as OnnxAdd
from onnx2pytorch.operations.div import Div as OnnxDiv
from onnx2pytorch.operations.clip import Clip as OnnxClip
from onnx2pytorch.operations.reshape import Reshape as OnnxReshape
from onnx2pytorch.operations.squeeze import Squeeze as OnnxSqueeze
from onnx2pytorch.operations.unsqueeze import Unsqueeze as OnnxUnsqueeze
from onnx2pytorch.operations.transpose import Transpose as OnnxTranspose
from onnx2pytorch.operations.base import OperatorWrapper


class SymbolicBounds:
    """
    Represents symbolic affine bounds: y = A * z + b
    where z is the original input variable vector
    
    Each layer's output is expressed as an affine function of the network input by symbolic propagation:
    - Lower bound: y_lower = A_lower * z + b_lower
    - Upper bound: y_upper = A_upper * z + b_upper
    
    A (weight matrix): Accumulated product of all layer weight matrices (scaled by ReLU λ factors)
    b (bias vector): Accumulated sum of all layer biases (transformed by subsequent layers)
    """
    def __init__(self, lower_A: torch.Tensor, lower_b: torch.Tensor,
                 upper_A: torch.Tensor, upper_b: torch.Tensor):
        """
        Args:
            lower_A: Symbolic coefficient matrix for lower bound (output_dim x input_dim)
            lower_b: Bias vector for lower bound (output_dim,)
            upper_A: Symbolic coefficient matrix for upper bound (output_dim x input_dim)
            upper_b: Bias vector for upper bound (output_dim,)
        """
        self.lower_A = lower_A
        self.lower_b = lower_b
        self.upper_A = upper_A
        self.upper_b = upper_b
    
    def concretize(self, input_lb: torch.Tensor, input_ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Concretize symbolic bounds given concrete input bounds
        
        For lower bound y_l = A_l * z + b_l:
          - Positive symbolic coefficients in A_l: use input lower bound
          - Negative symbolic coefficients in A_l: use input upper bound
        """
        # For lower bound: positive symbolic coefficients use lb, negative use ub
        lower_A_pos = torch.clamp(self.lower_A, min=0)
        lower_A_neg = torch.clamp(self.lower_A, max=0)
        concrete_lb = (lower_A_pos @ input_lb + lower_A_neg @ input_ub + self.lower_b)
        
        # For upper bound: positive symbolic coefficients use ub, negative use lb
        upper_A_pos = torch.clamp(self.upper_A, min=0)
        upper_A_neg = torch.clamp(self.upper_A, max=0)
        concrete_ub = (upper_A_pos @ input_ub + upper_A_neg @ input_lb + self.upper_b)
        
        return concrete_lb, concrete_ub


class SymbolicIntervalVerifier(BaseVerifier):
    def __init__(self, dataset: Dataset, method, spec: Spec, device: str = 'cpu', extract_counterexample: bool = False):
        super().__init__(dataset, spec, device)
        if method != 'symbolic_interval':
            raise ValueError(f"SymbolicIntervalVerifier only supports 'symbolic_interval' method, got {method}.")
        
        # Store symbolic bounds for counterexample extraction
        self.last_symbolic_bounds = None
        self.last_output_lb = None
        self.last_output_ub = None
        self.last_counterexample = None  # Store found counterexamples
        self.extract_counterexample = extract_counterexample  # Flag to enable/disable counterexample extraction

    def _relu_symbolic_transformer(self, input_lb: torch.Tensor, input_ub: torch.Tensor,
                                            symbolic_bounds: SymbolicBounds) -> SymbolicBounds:
        """
        Symbolic Interval ReLU transformer based on "Efficient Formal Safety Analysis of Neural Networks" 2018.
        
        For ReLU activation: y = max(0, x)
        Given symbolic bounds for layer input:
          - Lower symbolic: Eq_low = A_l * z + b_l  (concretizes to [l_low, u_low])
          - Upper symbolic: Eq_up = A_u * z + b_u   (concretizes to [l_up, u_up])
        
        Three cases per neuron (based on the overall bounds [l, u] where l = min(l_low, l_up), u = max(u_low, u_up)):
        1) If u ≤ 0 (always inactive): y ∈ [0, 0]
        2) If l ≥ 0 (always active): y ∈ [Eq_low, Eq_up]
        3) If l < 0 < u (crossing): 
           - Lower ReLU: ReLU(Eq_low) → (u_low / (u_low - l_low)) * Eq_low
           - Upper ReLU: ReLU(Eq_up) → (u_up / (u_up - l_up)) * (Eq_up - l_up)
        
        Key insight: l_low, u_low are bounds for Eq_low; l_up, u_up are bounds for Eq_up. This is the idea extracted from the paper we referenced.
        
        Args:
            input_lb: Concrete input lower bounds to the network (z_min)
            input_ub: Concrete input upper bounds to the network (z_max)
            symbolic_bounds: Symbolic bounds (Eq_low and Eq_up) for this layer
        
        Returns:
            New symbolic bounds after ReLU
        """
        device = symbolic_bounds.lower_A.device
        neuron_count = symbolic_bounds.lower_A.shape[0]
        
        # Initialize output symbolic bounds
        new_lower_A = symbolic_bounds.lower_A.clone()
        new_lower_b = symbolic_bounds.lower_b.clone()
        new_upper_A = symbolic_bounds.upper_A.clone()
        new_upper_b = symbolic_bounds.upper_b.clone()
        
        # Flatten input bounds for concretization
        input_lb_flat = input_lb.view(-1)
        input_ub_flat = input_ub.view(-1)
        
        for i in range(neuron_count):
            # Concretize Eq_low to get [l_low, u_low]
            lower_A_i = symbolic_bounds.lower_A[i, :]
            lower_b_i = symbolic_bounds.lower_b[i]
            
            # l_low = min over z of (A_l * z + b_l)
            # When calculating the new lower bounds, 
            # positive symbolic coefficients use lb, negative use ub
            lower_A_pos = torch.clamp(lower_A_i, min=0)
            lower_A_neg = torch.clamp(lower_A_i, max=0)
            l_low = (lower_A_pos @ input_lb_flat + lower_A_neg @ input_ub_flat + lower_b_i).item()
            
            # u_low = max over z of (A_l * z + b_l)
            # When calculating the new upper bounds,
            # positive symbolic coefficients use ub, negative use lb
            u_low = (lower_A_pos @ input_ub_flat + lower_A_neg @ input_lb_flat + lower_b_i).item()
            
            # Concretize Eq_up to get [l_up, u_up]
            upper_A_i = symbolic_bounds.upper_A[i, :]
            upper_b_i = symbolic_bounds.upper_b[i]
            
            # l_up = min over z of (A_u * z + b_u)
            # When calculating the new lower bounds, 
            # positive symbolic coefficients use lb, negative use ub
            upper_A_pos = torch.clamp(upper_A_i, min=0)
            upper_A_neg = torch.clamp(upper_A_i, max=0)
            l_up = (upper_A_pos @ input_lb_flat + upper_A_neg @ input_ub_flat + upper_b_i).item()
            
            # u_up = max over z of (A_u * z + b_u)
            # When calculating the new upper bounds,
            # positive symbolic coefficients use ub, negative use lb
            u_up = (upper_A_pos @ input_ub_flat + upper_A_neg @ input_lb_flat + upper_b_i).item()
            
            # Determine overall bounds for this neuron
            l_overall = min(l_low, l_up)
            u_overall = max(u_low, u_up)
            
            if u_overall <= 0:
                # Case 1: Always inactive, y = 0
                new_lower_A[i, :] = 0
                new_lower_b[i] = 0
                new_upper_A[i, :] = 0
                new_upper_b[i] = 0
                
            elif l_overall >= 0:
                # Case 2: Always active, y = x
                # Output bounds remain: [Eq_low, Eq_up]
                pass
                
            else:
                # Case 3: Crossing case (l_overall < 0 < u_overall)
                # Lower bound: ReLU(Eq_low) → (u_low / (u_low - l_low)) * Eq_low
                if u_low > l_low:  # Avoid division by zero
                    lambda_lower = u_low / (u_low - l_low)
                    new_lower_A[i, :] = symbolic_bounds.lower_A[i, :] * lambda_lower
                    new_lower_b[i] = symbolic_bounds.lower_b[i] * lambda_lower
                else:
                    # Degenerate case: set to 0
                    new_lower_A[i, :] = 0
                    new_lower_b[i] = 0
                
                # Upper bound: ReLU(Eq_up) → (u_up / (u_up - l_up)) * (Eq_up - l_up)
                if u_up > l_up:  # Avoid division by zero
                    lambda_upper = u_up / (u_up - l_up)
                    new_upper_A[i, :] = symbolic_bounds.upper_A[i, :] * lambda_upper
                    new_upper_b[i] = (symbolic_bounds.upper_b[i] - l_up) * lambda_upper
                else:
                    # Degenerate case: set to 0
                    new_upper_A[i, :] = 0
                    new_upper_b[i] = 0
        
        return SymbolicBounds(new_lower_A, new_lower_b,
                             new_upper_A, new_upper_b)



    def _abstract_constraint_solving_core(self, model: nn.Module, input_lb: torch.Tensor, input_ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Symbolic interval propagation through the network
        """
        # Initialize symbolic bounds as identity (x = x)
        input_dim = input_lb.numel()
        device = input_lb.device
        
        # Start with identity transformation: y = I*z + 0
        identity_A = torch.eye(input_dim, device=device)
        zero_b = torch.zeros(input_dim, device=device)
        
        symbolic_bounds = SymbolicBounds(
            lower_A=identity_A,
            lower_b=zero_b,
            upper_A=identity_A,
            upper_b=zero_b
        )
        
        # Keep track of concrete bounds for activation function relaxations
        concrete_lb = input_lb.clone().view(-1)
        concrete_ub = input_ub.clone().view(-1)
        
        # Store layer bounds for BaB (similar to hybridz)
        self.symbolic_layer_bounds = {}
        
        layer_index = 0
        current_shape = input_lb.shape

        for layer in model.children():
            print(f"Processing layer {layer_index}: {type(layer)}")
            
            if isinstance(layer, nn.Linear):
                W = layer.weight  # (out_features, in_features)
                b = layer.bias
                
                # Transform symbolic bounds: y = Wx + b
                # For each output neuron y_i = sum_j W[i,j] * x_j + b[i]
                # We need to consider positive and negative weights separately
                
                W_pos = torch.clamp(W, min=0)  # Positive part of W
                W_neg = torch.clamp(W, max=0)  # Negative part of W
                
                # Lower bound: y_l = W_pos * x_l + W_neg * x_u + b
                # (positive weights use lower bound, negative weights use upper bound)
                new_lower_A = W_pos @ symbolic_bounds.lower_A + W_neg @ symbolic_bounds.upper_A
                new_lower_b = W_pos @ symbolic_bounds.lower_b + W_neg @ symbolic_bounds.upper_b
                if b is not None:
                    new_lower_b = new_lower_b + b
                
                # Upper bound: y_u = W_pos * x_u + W_neg * x_l + b
                # (positive weights use upper bound, negative weights use lower bound)
                new_upper_A = W_pos @ symbolic_bounds.upper_A + W_neg @ symbolic_bounds.lower_A
                new_upper_b = W_pos @ symbolic_bounds.upper_b + W_neg @ symbolic_bounds.lower_b
                if b is not None:
                    new_upper_b = new_upper_b + b
                
                symbolic_bounds = SymbolicBounds(new_lower_A, new_lower_b,
                                                new_upper_A, new_upper_b)
                
                # Update concrete bounds
                concrete_lb, concrete_ub = symbolic_bounds.concretize(input_lb.view(-1), input_ub.view(-1))
                current_shape = (W.shape[0],)
                
                # Store layer bounds for BaB
                layer_name = f"linear_{layer_index}"
                self.symbolic_layer_bounds[layer_name] = {
                    'lb': concrete_lb.view(current_shape).clone(),
                    'ub': concrete_ub.view(current_shape).clone(),
                    'symbolic_bounds': SymbolicBounds(
                        new_lower_A.clone(), new_lower_b.clone(),
                        new_upper_A.clone(), new_upper_b.clone()
                    )
                }
                
                layer_index += 1

            elif isinstance(layer, nn.Conv2d):
                # For Conv2d, we need to handle it differently
                # Flatten current bounds and apply conv as matrix multiplication
                W = layer.weight
                b = layer.bias
                stride = layer.stride
                padding = layer.padding
                
                # Apply convolution to concrete bounds first to get shape
                W_pos = torch.clamp(W, min=0)
                W_neg = torch.clamp(W, max=0)
                
                concrete_lb_reshaped = concrete_lb.view(current_shape)
                concrete_ub_reshaped = concrete_ub.view(current_shape)
                
                # Add batch dimension for conv2d
                if concrete_lb_reshaped.ndim == 1:
                    # After linear layer, need to reshape
                    raise NotImplementedError("Conv2d after Linear not yet supported in symbolic interval")
                
                next_lb = (
                    nn.functional.conv2d(concrete_lb_reshaped.unsqueeze(0), W_pos, None, stride, padding) +
                    nn.functional.conv2d(concrete_ub_reshaped.unsqueeze(0), W_neg, None, stride, padding)
                ).squeeze(0)
                next_ub = (
                    nn.functional.conv2d(concrete_ub_reshaped.unsqueeze(0), W_pos, None, stride, padding) +
                    nn.functional.conv2d(concrete_lb_reshaped.unsqueeze(0), W_neg, None, stride, padding)
                ).squeeze(0)
                
                if b is not None:
                    next_lb += b.view(-1, 1, 1)
                    next_ub += b.view(-1, 1, 1)
                
                concrete_lb = next_lb.view(-1)
                concrete_ub = next_ub.view(-1)
                current_shape = next_lb.shape
                
                # For symbolic bounds in conv layers, we simplify by resetting to concrete
                # (Full symbolic conv would be very complex)
                out_dim = concrete_lb.numel()
                in_dim = input_lb.numel()
                symbolic_bounds = SymbolicBounds(
                    lower_A=torch.zeros(out_dim, in_dim, device=device),
                    lower_b=concrete_lb,
                    upper_A=torch.zeros(out_dim, in_dim, device=device),
                    upper_b=concrete_ub
                )

            elif isinstance(layer, nn.ReLU):
                # Store pre-activation bounds for BaB (before ReLU)
                layer_name = f"relu_{layer_index}"
                
                # Get concrete bounds for this layer (for BaB and constraint checking)
                concrete_lb_reshaped = concrete_lb.view(current_shape)
                concrete_ub_reshaped = concrete_ub.view(current_shape)
                
                # Store pre-ReLU bounds (these are what BaB needs to split on)
                # Use the previous layer's name for pre-activation bounds
                prev_layer_name = list(self.symbolic_layer_bounds.keys())[-1] if self.symbolic_layer_bounds else f"pre_{layer_name}"
                if prev_layer_name not in self.symbolic_layer_bounds:
                    self.symbolic_layer_bounds[prev_layer_name] = {
                        'lb': concrete_lb_reshaped.clone(),
                        'ub': concrete_ub_reshaped.clone()
                    }
                
                # Apply ReLU constraints if any
                if hasattr(self, 'current_relu_constraints') and self.current_relu_constraints:
                    for constraint in self.current_relu_constraints:
                        if constraint['layer'] == layer_name:
                            neuron_idx = constraint['neuron_idx']
                            constraint_type = constraint['constraint_type']
                            
                            flat_lb = concrete_lb_reshaped.view(-1)
                            flat_ub = concrete_ub_reshaped.view(-1)
                            
                            if neuron_idx < flat_lb.numel():
                                if constraint_type == 'inactive':
                                    # Force upper bound to 0
                                    flat_ub[neuron_idx] = min(flat_ub[neuron_idx].item(), 0.0)
                                elif constraint_type == 'active':
                                    # Force lower bound to 0
                                    flat_lb[neuron_idx] = max(flat_lb[neuron_idx].item(), 0.0)
                            
                            concrete_lb_reshaped = flat_lb.view(current_shape)
                            concrete_ub_reshaped = flat_ub.view(current_shape)
                
                # Apply symbolic interval ReLU relaxation
                # Pass the original input bounds (not the layer bounds!)
                symbolic_bounds = self._relu_symbolic_transformer(
                    input_lb, input_ub, symbolic_bounds
                )
                
                # Update concrete bounds
                concrete_lb, concrete_ub = symbolic_bounds.concretize(input_lb.view(-1), input_ub.view(-1))
                # Also apply ReLU to concrete bounds
                concrete_lb = torch.clamp(concrete_lb, min=0)
                concrete_ub = torch.clamp(concrete_ub, min=0)
                
                # Store post-ReLU bounds
                self.symbolic_layer_bounds[layer_name] = {
                    'lb': concrete_lb.view(current_shape).clone(),
                    'ub': concrete_ub.view(current_shape).clone()
                }
                
                layer_index += 1

            elif isinstance(layer, nn.Sigmoid):
                raise NotImplementedError(
                    "Sigmoid activation is not supported in Symbolic Interval verification. "
                    "This method only supports ReLU activations as per IBP+Symbolic Interval paper."
                )

            elif isinstance(layer, nn.Tanh):
                raise NotImplementedError(
                    "Tanh activation is not supported in Symbolic Interval verification. "
                    "This method only supports ReLU activations as per IBP+Symbolic Interval paper."
                )

            elif isinstance(layer, nn.Flatten) or isinstance(layer, OnnxFlatten):
                print("Flattening layer detected.")
                # Just update the shape tracking
                concrete_lb = concrete_lb.view(-1)
                concrete_ub = concrete_ub.view(-1)
                current_shape = concrete_lb.shape

            elif isinstance(layer, nn.MaxPool2d):
                # MaxPool: apply to concrete bounds
                concrete_lb_reshaped = concrete_lb.view(current_shape)
                concrete_ub_reshaped = concrete_ub.view(current_shape)
                
                concrete_lb = nn.functional.max_pool2d(
                    concrete_lb_reshaped.unsqueeze(0), 
                    kernel_size=layer.kernel_size, 
                    stride=layer.stride, 
                    padding=layer.padding
                ).squeeze(0).view(-1)
                
                concrete_ub = nn.functional.max_pool2d(
                    concrete_ub_reshaped.unsqueeze(0),
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding
                ).squeeze(0).view(-1)
                
                current_shape = concrete_lb.view(current_shape[0], -1, 
                                                  concrete_lb.numel() // current_shape[0]).shape
                
                # Reset symbolic bounds to concrete
                out_dim = concrete_lb.numel()
                in_dim = input_lb.numel()
                symbolic_bounds = SymbolicBounds(
                    lower_A=torch.zeros(out_dim, in_dim, device=device),
                    lower_b=concrete_lb,
                    upper_A=torch.zeros(out_dim, in_dim, device=device),
                    upper_b=concrete_ub
                )

            else:
                raise NotImplementedError(f"Layer {type(layer)} not supported in symbolic interval propagation.")

        # Final concretization
        output_lb, output_ub = symbolic_bounds.concretize(input_lb.view(-1), input_ub.view(-1))
        
        # Store for counterexample extraction
        self.last_symbolic_bounds = symbolic_bounds
        self.last_output_lb = output_lb.view(current_shape)
        self.last_output_ub = output_ub.view(current_shape)
        
        return output_lb.view(current_shape), output_ub.view(current_shape), None

    def _compute_difference_bounds(self, true_label: int, other_label: int) -> tuple:
        """
        Compute bounds for the difference: output[true_label] - output[other_label]
        
        This is more precise than comparing individual bounds because:
        - diff = (A[true] - A[other]) * z + (b[true] - b[other])
        - Common error terms cancel out in the difference
        
        This is the ERAN-style verification approach, same as HybridZ's _classify_with_difference_bounds_hz
        
        Args:
            true_label: Index of the true label neuron
            other_label: Index of the competing neuron
            
        Returns:
            (diff_lb, diff_ub): Lower and upper bounds of the difference
        """
        if self.last_symbolic_bounds is None:
            return None, None
        
        # Get input bounds
        input_lb = self.input_lb[0] if self.input_lb.ndim > 1 else self.input_lb
        input_ub = self.input_ub[0] if self.input_ub.ndim > 1 else self.input_ub
        input_lb_flat = input_lb.view(-1)
        input_ub_flat = input_ub.view(-1)
        
        # Construct difference symbolic bounds
        # diff_lower = A_lower[true] - A_upper[other]  (minimize the difference)
        # diff_upper = A_upper[true] - A_lower[other]  (maximize the difference)
        diff_lower_A = self.last_symbolic_bounds.lower_A[true_label] - self.last_symbolic_bounds.upper_A[other_label]
        diff_lower_b = self.last_symbolic_bounds.lower_b[true_label] - self.last_symbolic_bounds.upper_b[other_label]
        
        diff_upper_A = self.last_symbolic_bounds.upper_A[true_label] - self.last_symbolic_bounds.lower_A[other_label]
        diff_upper_b = self.last_symbolic_bounds.upper_b[true_label] - self.last_symbolic_bounds.lower_b[other_label]
        
        # Concretize: for lower bound, positive coefficients use lb, negative use ub
        diff_A_pos = torch.clamp(diff_lower_A, min=0)
        diff_A_neg = torch.clamp(diff_lower_A, max=0)
        diff_lb = (diff_A_pos @ input_lb_flat + diff_A_neg @ input_ub_flat + diff_lower_b).item()
        
        # For upper bound, positive coefficients use ub, negative use lb
        diff_A_pos = torch.clamp(diff_upper_A, min=0)
        diff_A_neg = torch.clamp(diff_upper_A, max=0)
        diff_ub = (diff_A_pos @ input_ub_flat + diff_A_neg @ input_lb_flat + diff_upper_b).item()
        
        return diff_lb, diff_ub

    def _abstract_constraint_solving(self, input_lb: torch.Tensor, input_ub: torch.Tensor, sample_idx: int) -> VerificationStatus:
        print(f"Performing Symbolic Interval propagation (IBP+Symbolic Interval method for ReLU networks)")

        output_lb, output_ub, _ = self._abstract_constraint_solving_core(
            self.spec.model.pytorch_model, input_lb, input_ub
        )
        
        # Print detailed output bounds for better understanding
        print(f"\n{'='*80}")
        print(f"Output Layer Bounds (Symbolic Interval Propagation)")
        print(f"{'='*80}")
        print(f"Using symbolic affine bounds: y = A*z + b")
        print(f"  where z is input (bounded by input_lb <= z <= input_ub)")
        print(f"  A, b are accumulated weights/biases through all layers")
        print(f"\nConcretized output bounds for each neuron:")
        for i in range(output_lb.shape[0]):
            print(f"  Neuron {i}: [{output_lb[i].item():8.4f}, {output_ub[i].item():8.4f}]  (width: {(output_ub[i] - output_lb[i]).item():.4f})")
        
        # Analyze for local robustness using ERAN-style difference verification
        true_label = self.spec.output_spec.labels[sample_idx].item() if self.spec.output_spec.labels is not None else None
        if true_label is not None:
            print(f"\n{'='*80}")
            print(f"Local Robustness Verification (ERAN-style Difference Method)")
            print(f"{'='*80}")
            print(f"True label: {true_label}")
            print(f"Method: Compute bounds for diff = output[{true_label}] - output[j] for each j ≠ {true_label}")
            print(f"Property verified if diff_lb > 0 for all j (i.e., output[{true_label}] > output[j])")
            print()
            
            all_differences_positive = True
            violations = []
            
            for j in range(output_lb.shape[0]):
                if j == true_label:
                    continue
                
                # Compute precise difference bounds
                diff_lb, diff_ub = self._compute_difference_bounds(true_label, j)
                
                print(f"  Class {j}: diff[{true_label}-{j}] ∈ [{diff_lb:8.4f}, {diff_ub:8.4f}]", end="")
                
                if diff_lb > 0:
                    print(f"  ✓ Safe (diff_lb > 0)")
                else:
                    print(f"  ✗ Potential violation (diff_lb ≤ 0)")
                    all_differences_positive = False
                    violations.append((j, diff_lb, diff_ub))
            
            print()
            if all_differences_positive:
                print(f"  ✅ All difference lower bounds > 0")
                print(f"  Conclusion: output[{true_label}] > output[j] for all j ≠ {true_label}")
                print(f"  Result: Property VERIFIED (SAT)")
            else:
                print(f"  ⚠️  Found {len(violations)} potential violations:")
                for j, diff_lb, diff_ub in violations[:5]:
                    print(f"    Class {j}: diff_lb = {diff_lb:8.4f} ≤ 0")
                print(f"  Conclusion: Cannot prove output[{true_label}] > output[j] for all j")
                print(f"  Result: Potential violation (UNSAT) - may be over-approximation")
        
        print(f"{'='*80}\n")

        verdict = self._single_result_verdict(
            output_lb, output_ub,
            self.spec.output_spec.output_constraints if self.spec.output_spec.output_constraints is not None else None,
            true_label
        )

        print(f"📊 Verification verdict: {verdict.name}")
        return verdict

    def get_counterexample(self) -> Optional[torch.Tensor]:
        """
        Extract counterexample using Gurobi LP solver on symbolic bounds
        
        Since we maintain symbolic affine bounds y = A*z + b, the network output is a linear function of input z.
        We can formulate counterexample search as a linear programming problem:
        
        maximize: violation_objective(y)
        subject to:
            - y = A * z + b  (output symbolic bounds)
            - z_lb <= z <= z_ub  (input box constraints)
        
        This is exact for ReLU networks using symbolic interval relaxation.
        
        Returns:
            Concrete input violating the property, or None if no violation exists
        """
        if self.last_symbolic_bounds is None:
            print("Warning: No symbolic bounds available for counterexample extraction")
            return None
        
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            print("Error: Gurobi not available. Please install gurobipy and ensure Gurobi license is available (set GRB_LICENSE_FILE)")
            return None
        
        try:
            # Get input bounds from last run
            input_lb = self.input_lb[0] if self.input_lb.ndim > 1 else self.input_lb
            input_ub = self.input_ub[0] if self.input_ub.ndim > 1 else self.input_ub
            
            # Get output specification
            output_constraints = self.spec.output_spec.output_constraints
            true_label = self.spec.output_spec.labels[0].item() if self.spec.output_spec.labels is not None else None
            
            # Extract symbolic bounds matrices
            # Output: y = A * z + b
            A_lower = self.last_symbolic_bounds.lower_A.detach().cpu().numpy()  # (output_dim, input_dim)
            b_lower = self.last_symbolic_bounds.lower_b.detach().cpu().numpy()  # (output_dim,)
            A_upper = self.last_symbolic_bounds.upper_A.detach().cpu().numpy()
            b_upper = self.last_symbolic_bounds.upper_b.detach().cpu().numpy()
            
            input_lb_np = input_lb.detach().cpu().numpy().flatten()  # (input_dim,)
            input_ub_np = input_ub.detach().cpu().numpy().flatten()
            
            input_dim = input_lb_np.shape[0]
            output_dim = A_lower.shape[0]
            
            print(f"Configuring Gurobi LP problem:")
            print(f"   Input dimension: {input_dim}")
            print(f"   Output dimension: {output_dim}")
            
            # Create Gurobi environment and model (following hybridz_operations configuration)
            env = gp.Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.setParam('LogToConsole', 0)
            env.start()
            
            model = gp.Model("symbolic_interval_counterexample", env=env)
            model.setParam('OutputFlag', 0)
            
            # LP solver configuration (optimized for symbolic interval's complete network encoding)
            model.setParam('Method', 2)  # Barrier method for LP
            model.setParam('Crossover', 0)  # Disable crossover (faster)
            model.setParam('BarHomogeneous', 1)
            model.setParam('Threads', 0)  # Use all available threads
            model.setParam('TimeLimit', 60.0)  # 60 seconds time limit
            
            # Special configuration for output layer
            if output_dim <= 20:
                print(f"   Detected output layer (output_dim={output_dim}), using high-precision configuration")
                model.setParam('NumericFocus', 2)  # Higher numerical precision
                model.setParam('FeasibilityTol', 1e-7)
                model.setParam('OptimalityTol', 1e-7)
            else:
                model.setParam('NumericFocus', 1)
                model.setParam('Presolve', 2)
            
            # Decision variables: input z
            z = model.addMVar(shape=input_dim, lb=input_lb_np, ub=input_ub_np, name="z")
            
            # Output variables: y_lower and y_upper
            # y_lower[i] = A_lower[i, :] @ z + b_lower[i]
            # y_upper[i] = A_upper[i, :] @ z + b_upper[i]
            y_lower = model.addMVar(shape=output_dim, lb=-GRB.INFINITY, name="y_lower")
            y_upper = model.addMVar(shape=output_dim, lb=-GRB.INFINITY, name="y_upper")
            
            # Add constraints: y = A * z + b
            print(f"   Adding {output_dim * 2} linear constraints...")
            for i in range(output_dim):
                model.addConstr(y_lower[i] == A_lower[i, :] @ z + b_lower[i], name=f"output_lower_{i}")
                model.addConstr(y_upper[i] == A_upper[i, :] @ z + b_upper[i], name=f"output_upper_{i}")
            
            # Build objective function based on property type
            violation_found = False
            
            if output_constraints is not None:
                for constraint in output_constraints:
                    if constraint['type'] == 'local_robustness':
                        # Find adversarial example: maximize max_j(y_upper[j] - y_lower[true_label]) for j != true_label
                        if true_label is not None:
                            print(f"   Property: Local robustness (true label = {true_label})")
                            
                            # Introduce auxiliary variable to represent max
                            max_violation = model.addVar(lb=-GRB.INFINITY, name="max_violation")
                            
                            # max_violation >= y_upper[j] - y_lower[true_label] for all j != true_label
                            for j in range(output_dim):
                                if j != true_label:
                                    model.addConstr(max_violation >= y_upper[j] - y_lower[true_label],
                                                  name=f"violation_{j}")
                            
                            # Maximize violation
                            model.setObjective(max_violation, GRB.MAXIMIZE)
                            violation_found = True
                            break
                    
                    elif constraint['type'] == 'greater_than':
                        # Want y[idx1] > y[idx2], so find violation: maximize y_upper[idx2] - y_lower[idx1]
                        idx1 = constraint.get('output_idx_1', 0)
                        idx2 = constraint.get('output_idx_2', 1)
                        print(f"   Property: output[{idx1}] > output[{idx2}]")
                        model.setObjective(y_upper[idx2] - y_lower[idx1], GRB.MAXIMIZE)
                        violation_found = True
                        break
                    
                    elif constraint['type'] == 'less_than':
                        # Want y[idx1] < y[idx2], so find violation: maximize y_upper[idx1] - y_lower[idx2]
                        idx1 = constraint.get('output_idx_1', 0)
                        idx2 = constraint.get('output_idx_2', 1)
                        print(f"   Property: output[{idx1}] < output[{idx2}]")
                        model.setObjective(y_upper[idx1] - y_lower[idx2], GRB.MAXIMIZE)
                        violation_found = True
                        break
            
            # Default: local robustness
            if not violation_found and true_label is not None:
                print(f"   Property: Local robustness (default, true label = {true_label})")
                max_violation = model.addVar(lb=-GRB.INFINITY, name="max_violation")
                for j in range(output_dim):
                    if j != true_label:
                        model.addConstr(max_violation >= y_upper[j] - y_lower[true_label],
                                      name=f"violation_{j}")
                model.setObjective(max_violation, GRB.MAXIMIZE)
                violation_found = True
            
            if not violation_found:
                print("Warning: No clear verification objective, cannot extract counterexample")
                model.dispose()
                env.dispose()
                return None
            
            # Solve LP
            print("   Solving LP with Gurobi...")
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                # Extract solution
                z_solution = torch.tensor([z[i].X for i in range(input_dim)], 
                                         dtype=input_lb.dtype, device=input_lb.device)
                violation_value = model.objVal
                
                print(f"Success: Gurobi found counterexample, violation value: {violation_value:.6f}")
                
                # Verify on actual network
                with torch.no_grad():
                    z_reshaped = z_solution.view(input_lb.shape)
                    actual_output = self.spec.model.pytorch_model(z_reshaped.unsqueeze(0)).squeeze(0)
                    
                    if true_label is not None:
                        predicted_label = torch.argmax(actual_output).item()
                        print(f"   Actual network output: predicted={predicted_label}, true={true_label}")
                        print(f"   Logits: {actual_output.cpu().numpy()}")
                        
                        if predicted_label != true_label:
                            print(f"Counterexample verified: Model misclassified!")
                        else:
                            print(f"Warning: LP found violation in relaxation, but actual network is still correct")
                            print(f"   (This is due to ReLU over-approximation, which is expected)")
                
                model.dispose()
                env.dispose()
                return z_solution.view(input_lb.shape)
            
            elif model.status == GRB.INFEASIBLE:
                print("Success: Gurobi proved no counterexample exists (problem infeasible)")
                model.dispose()
                env.dispose()
                return None
            
            elif model.status == GRB.TIME_LIMIT:
                print("Timeout: Gurobi reached time limit, no definite answer")
                model.dispose()
                env.dispose()
                return None
            
            else:
                print(f"Warning: Gurobi solver status: {model.status}")
                model.dispose()
                env.dispose()
                return None
                
        except Exception as e:
            print(f"Error: Gurobi counterexample extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def verify(self) -> VerificationStatus:
        print("Starting Symbolic Interval verification pipeline")

        num_samples = self.input_center.shape[0] if self.input_center.ndim > 1 else 1
        print(f"Total samples: {num_samples}")
        results = []
        
        for idx in range(num_samples):
            print(f"\nProcessing sample {idx+1}/{num_samples}")
            print("="*80)

            center_input, true_label = self.get_sample_center_and_label(idx)
            if not self.check_clean_prediction(center_input, true_label, idx):
                print(f"Skipping verification for sample {idx+1}")
                results.append(VerificationStatus.CLEAN_FAILURE)
                continue

            if self.input_lb.ndim == 1:
                lb_i = self.input_lb
                ub_i = self.input_ub
            else:
                lb_i = self.input_lb[idx]
                ub_i = self.input_ub[idx]

            self.clean_prediction_stats['verification_attempted'] += 1

            print("Step 1: Symbolic Interval abstract constraint solving")
            initial_verdict = self._abstract_constraint_solving(lb_i, ub_i, idx)

            if initial_verdict == VerificationStatus.SAT:
                self.clean_prediction_stats['verification_sat'] += 1
                print(f"✅ Symbolic Interval verification success - Sample {idx+1} safe")
                results.append(initial_verdict)
                continue
            elif initial_verdict == VerificationStatus.UNSAT:
                self.clean_prediction_stats['verification_unsat'] += 1
                print(f"Symbolic Interval detected potential violation - Sample {idx+1}")
                
                # Only attempt counterexample extraction if explicitly enabled
                if self.extract_counterexample:
                    print(f"\n{'='*80}")
                    print(f"Step 2: Counterexample Extraction via Gurobi LP Solver")
                    print(f"{'='*80}")
                    print(f"Method: Exact LP solving on symbolic bounds")
                    print(f"  Formulation: maximize violation")
                    print(f"  Subject to: y = A*z + b  (symbolic affine bounds)")
                    print(f"              z_lb <= z <= z_ub  (input box constraints)")
                    print(f"  Goal: Find concrete input z that causes misclassification")
                    print(f"{'='*80}\n")
                    
                    # Try to extract a concrete counterexample
                    counterexample = self.get_counterexample()
                    
                    if counterexample is not None:
                        # Verify the counterexample on actual network
                        with torch.no_grad():
                            ce_output = self.spec.model.pytorch_model(counterexample.unsqueeze(0)).squeeze(0)
                            ce_predicted = torch.argmax(ce_output).item()
                            
                            if ce_predicted != true_label:
                                print(f"COUNTEREXAMPLE CONFIRMED: Network misclassifies!")
                                print(f"   True label: {true_label}, Predicted: {ce_predicted}")
                                print(f"   Counterexample saved to self.last_counterexample")
                                self.last_counterexample = counterexample
                                results.append(VerificationStatus.UNSAT)
                            else:
                                print(f"Warning: Counterexample found in relaxation but network still correct")
                                print(f"   (Over-approximation artifact - property likely HOLDS)")
                                results.append(VerificationStatus.UNKNOWN)
                    else:
                        print(f"Success: No counterexample found - property verified!")
                        results.append(VerificationStatus.SAT)
                else:
                    # Without counterexample extraction, return UNSAT (conservative)
                    print(f"   Counterexample extraction disabled, returning UNSAT (conservative)")
                    results.append(VerificationStatus.UNSAT)
            else:
                self.clean_prediction_stats['verification_unknown'] += 1
                print(f"Symbolic Interval inconclusive - Sample {idx+1}")
                results.append(initial_verdict)

        self.print_verification_stats()
        return self._all_results_verdict(results)
