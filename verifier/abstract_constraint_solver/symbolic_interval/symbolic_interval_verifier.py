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
    Represents symbolic affine bounds: a * x + b
    where x is the input variable vector
    """
    def __init__(self, lower_coef: torch.Tensor, lower_bias: torch.Tensor,
                 upper_coef: torch.Tensor, upper_bias: torch.Tensor):
        """
        Args:
            lower_coef: Coefficients for lower bound (output_dim x input_dim)
            lower_bias: Bias for lower bound (output_dim,)
            upper_coef: Coefficients for upper bound (output_dim x input_dim)
            upper_bias: Bias for upper bound (output_dim,)
        """
        self.lower_coef = lower_coef
        self.lower_bias = lower_bias
        self.upper_coef = upper_coef
        self.upper_bias = upper_bias
    
    def concretize(self, input_lb: torch.Tensor, input_ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Concretize symbolic bounds given concrete input bounds
        """
        # For lower bound: positive coefficients use lb, negative use ub
        lower_coef_pos = torch.clamp(self.lower_coef, min=0)
        lower_coef_neg = torch.clamp(self.lower_coef, max=0)
        concrete_lb = (lower_coef_pos @ input_lb + lower_coef_neg @ input_ub + self.lower_bias)
        
        # For upper bound: positive coefficients use ub, negative use lb
        upper_coef_pos = torch.clamp(self.upper_coef, min=0)
        upper_coef_neg = torch.clamp(self.upper_coef, max=0)
        concrete_ub = (upper_coef_pos @ input_ub + upper_coef_neg @ input_lb + self.upper_bias)
        
        return concrete_lb, concrete_ub


class SymbolicIntervalVerifier(BaseVerifier):
    def __init__(self, dataset: Dataset, method, spec: Spec, device: str = 'cpu'):
        super().__init__(dataset, spec, device)
        if method != 'symbolic_interval':
            raise ValueError(f"SymbolicIntervalVerifier only supports 'symbolic_interval' method, got {method}.")
        
        # Store symbolic bounds for counterexample extraction
        self.last_symbolic_bounds = None
        self.last_output_lb = None
        self.last_output_ub = None

    def _symbolic_interval_relu_relaxation(self, input_lb: torch.Tensor, input_ub: torch.Tensor,
                                            symbolic_bounds: SymbolicBounds) -> SymbolicBounds:
        """
        Symbolic Interval ReLU relaxation based on IBP+Symbolic Interval paper.
        
        For ReLU activation: y = max(0, x)
        Given symbolic bounds for layer input:
          - Lower symbolic: Eq_low = a_l * z + b_l  (concretizes to [l_low, u_low])
          - Upper symbolic: Eq_up = a_u * z + b_u   (concretizes to [l_up, u_up])
        
        Three cases per neuron (based on the overall bounds [l, u] where l = min(l_low, l_up), u = max(u_low, u_up)):
        1) If u ≤ 0 (always inactive): y ∈ [0, 0]
        2) If l ≥ 0 (always active): y ∈ [Eq_low, Eq_up]
        3) If l < 0 < u (crossing): 
           - Lower ReLU: ReLU(Eq_low) → (u_low / (u_low - l_low)) * Eq_low
           - Upper ReLU: ReLU(Eq_up) → (u_up / (u_up - l_up)) * (Eq_up - l_up)
        
        Key insight: l_low, u_low are bounds for Eq_low; l_up, u_up are bounds for Eq_up.
        They are different in general!
        
        Args:
            input_lb: Concrete input lower bounds to the network (z_min)
            input_ub: Concrete input upper bounds to the network (z_max)
            symbolic_bounds: Symbolic bounds (Eq_low and Eq_up) for this layer
        
        Returns:
            New symbolic bounds after ReLU
        """
        device = symbolic_bounds.lower_coef.device
        neuron_count = symbolic_bounds.lower_coef.shape[0]
        
        # Initialize output symbolic bounds
        new_lower_coef = symbolic_bounds.lower_coef.clone()
        new_lower_bias = symbolic_bounds.lower_bias.clone()
        new_upper_coef = symbolic_bounds.upper_coef.clone()
        new_upper_bias = symbolic_bounds.upper_bias.clone()
        
        # Flatten input bounds for concretization
        input_lb_flat = input_lb.view(-1)
        input_ub_flat = input_ub.view(-1)
        
        for i in range(neuron_count):
            # Concretize Eq_low to get [l_low, u_low]
            lower_coef_i = symbolic_bounds.lower_coef[i, :]
            lower_bias_i = symbolic_bounds.lower_bias[i]
            
            # l_low = min over z of (a_l * z + b_l)
            lower_coef_pos = torch.clamp(lower_coef_i, min=0)
            lower_coef_neg = torch.clamp(lower_coef_i, max=0)
            l_low = (lower_coef_pos @ input_lb_flat + lower_coef_neg @ input_ub_flat + lower_bias_i).item()
            
            # u_low = max over z of (a_l * z + b_l)
            u_low = (lower_coef_pos @ input_ub_flat + lower_coef_neg @ input_lb_flat + lower_bias_i).item()
            
            # Concretize Eq_up to get [l_up, u_up]
            upper_coef_i = symbolic_bounds.upper_coef[i, :]
            upper_bias_i = symbolic_bounds.upper_bias[i]
            
            # l_up = min over z of (a_u * z + b_u)
            upper_coef_pos = torch.clamp(upper_coef_i, min=0)
            upper_coef_neg = torch.clamp(upper_coef_i, max=0)
            l_up = (upper_coef_pos @ input_lb_flat + upper_coef_neg @ input_ub_flat + upper_bias_i).item()
            
            # u_up = max over z of (a_u * z + b_u)
            u_up = (upper_coef_pos @ input_ub_flat + upper_coef_neg @ input_lb_flat + upper_bias_i).item()
            
            # Determine overall bounds for this neuron
            l_overall = min(l_low, l_up)
            u_overall = max(u_low, u_up)
            
            if u_overall <= 0:
                # Case 1: Always inactive, y = 0
                new_lower_coef[i, :] = 0
                new_lower_bias[i] = 0
                new_upper_coef[i, :] = 0
                new_upper_bias[i] = 0
                
            elif l_overall >= 0:
                # Case 2: Always active, y = x
                # Output bounds remain: [Eq_low, Eq_up]
                pass
                
            else:
                # Case 3: Crossing case (l_overall < 0 < u_overall)
                # Lower bound: ReLU(Eq_low) → (u_low / (u_low - l_low)) * Eq_low
                if u_low > l_low:  # Avoid division by zero
                    lambda_lower = u_low / (u_low - l_low)
                    new_lower_coef[i, :] = symbolic_bounds.lower_coef[i, :] * lambda_lower
                    new_lower_bias[i] = symbolic_bounds.lower_bias[i] * lambda_lower
                else:
                    # Degenerate case: set to 0
                    new_lower_coef[i, :] = 0
                    new_lower_bias[i] = 0
                
                # Upper bound: ReLU(Eq_up) → (u_up / (u_up - l_up)) * (Eq_up - l_up)
                if u_up > l_up:  # Avoid division by zero
                    lambda_upper = u_up / (u_up - l_up)
                    new_upper_coef[i, :] = symbolic_bounds.upper_coef[i, :] * lambda_upper
                    new_upper_bias[i] = (symbolic_bounds.upper_bias[i] - l_up) * lambda_upper
                else:
                    # Degenerate case: set to 0
                    new_upper_coef[i, :] = 0
                    new_upper_bias[i] = 0
        
        return SymbolicBounds(new_lower_coef, new_lower_bias,
                             new_upper_coef, new_upper_bias)



    def _abstract_constraint_solving_core(self, model: nn.Module, input_lb: torch.Tensor, input_ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Symbolic interval propagation through the network
        """
        # Initialize symbolic bounds as identity (x = x)
        input_dim = input_lb.numel()
        device = input_lb.device
        
        # Start with identity transformation
        identity_coef = torch.eye(input_dim, device=device)
        zero_bias = torch.zeros(input_dim, device=device)
        
        symbolic_bounds = SymbolicBounds(
            lower_coef=identity_coef,
            lower_bias=zero_bias,
            upper_coef=identity_coef,
            upper_bias=zero_bias
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
                # Lower: y_l = W * x_l + b
                new_lower_coef = W @ symbolic_bounds.lower_coef
                new_lower_bias = W @ symbolic_bounds.lower_bias
                if b is not None:
                    new_lower_bias = new_lower_bias + b
                
                # Upper: y_u = W * x_u + b
                new_upper_coef = W @ symbolic_bounds.upper_coef
                new_upper_bias = W @ symbolic_bounds.upper_bias
                if b is not None:
                    new_upper_bias = new_upper_bias + b
                
                symbolic_bounds = SymbolicBounds(new_lower_coef, new_lower_bias,
                                                new_upper_coef, new_upper_bias)
                
                # Update concrete bounds
                concrete_lb, concrete_ub = symbolic_bounds.concretize(input_lb.view(-1), input_ub.view(-1))
                current_shape = (W.shape[0],)
                
                # Store layer bounds for BaB
                layer_name = f"linear_{layer_index}"
                self.symbolic_layer_bounds[layer_name] = {
                    'lb': concrete_lb.view(current_shape).clone(),
                    'ub': concrete_ub.view(current_shape).clone(),
                    'symbolic_bounds': SymbolicBounds(
                        new_lower_coef.clone(), new_lower_bias.clone(),
                        new_upper_coef.clone(), new_upper_bias.clone()
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
                    lower_coef=torch.zeros(out_dim, in_dim, device=device),
                    lower_bias=concrete_lb,
                    upper_coef=torch.zeros(out_dim, in_dim, device=device),
                    upper_bias=concrete_ub
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
                symbolic_bounds = self._symbolic_interval_relu_relaxation(
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
                    lower_coef=torch.zeros(out_dim, in_dim, device=device),
                    lower_bias=concrete_lb,
                    upper_coef=torch.zeros(out_dim, in_dim, device=device),
                    upper_bias=concrete_ub
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

    def _abstract_constraint_solving(self, input_lb: torch.Tensor, input_ub: torch.Tensor, sample_idx: int) -> VerificationStatus:
        print(f"Performing Symbolic Interval propagation (IBP+Symbolic Interval method for ReLU networks)")

        output_lb, output_ub, _ = self._abstract_constraint_solving_core(
            self.spec.model.pytorch_model, input_lb, input_ub
        )

        verdict = self._single_result_verdict(
            output_lb, output_ub,
            self.spec.output_spec.output_constraints if self.spec.output_spec.output_constraints is not None else None,
            self.spec.output_spec.labels[sample_idx].item() if self.spec.output_spec.labels is not None else None
        )

        print(f"📊 Verification verdict: {verdict.name}")
        return verdict

    def get_counterexample(self) -> Optional[torch.Tensor]:
        """
        Extract a counterexample by optimizing over the symbolic bounds.
        This uses the stored symbolic bounds from the last verification run.
        
        Returns:
            A concrete input that maximally violates the property, or None if extraction fails
        """
        if self.last_symbolic_bounds is None:
            print("⚠️  No symbolic bounds available for counterexample extraction")
            return None
        
        try:
            # Get the input bounds from the last run
            input_lb = self.input_lb[0] if self.input_lb.ndim > 1 else self.input_lb
            input_ub = self.input_ub[0] if self.input_ub.ndim > 1 else self.input_ub
            
            # Start from the center of the input region
            counterexample = ((input_lb + input_ub) / 2).clone().detach().requires_grad_(True)
            
            # Get output spec
            output_constraints = self.spec.output_spec.output_constraints
            true_label = self.spec.output_spec.labels[0].item() if self.spec.output_spec.labels is not None else None
            
            # Optimize to find a point that maximally violates the property
            optimizer = torch.optim.Adam([counterexample], lr=0.01)
            
            best_counterexample = counterexample.clone().detach()
            best_violation = float('-inf')
            
            for iteration in range(200):  # Optimization iterations
                optimizer.zero_grad()
                
                # Project back to valid input region
                with torch.no_grad():
                    counterexample.data = torch.clamp(counterexample.data, input_lb, input_ub)
                
                # Forward pass through the network
                output = self.spec.model.pytorch_model(counterexample.unsqueeze(0)).squeeze(0)
                
                # Compute violation objective based on property
                if output_constraints is not None:
                    # For set-based constraints, find maximum violation
                    violation = 0.0
                    for constraint in output_constraints:
                        if constraint['type'] == 'local_robustness':
                            # Maximize the logit difference: output[wrong] - output[true]
                            if true_label is not None:
                                wrong_logits = torch.cat([output[:true_label], output[true_label+1:]])
                                violation_per_class = wrong_logits - output[true_label]
                                violation = torch.max(violation_per_class)
                        elif constraint['type'] in ['greater_than', 'less_than']:
                            idx1 = constraint.get('output_idx_1', 0)
                            idx2 = constraint.get('output_idx_2', 1)
                            if constraint['type'] == 'greater_than':
                                # Want output[idx1] > output[idx2], so maximize output[idx2] - output[idx1]
                                violation = output[idx2] - output[idx1]
                            else:
                                # Want output[idx1] < output[idx2], so maximize output[idx1] - output[idx2]
                                violation = output[idx1] - output[idx2]
                else:
                    # Default: local robustness (maximize difference to true class)
                    if true_label is not None:
                        wrong_logits = torch.cat([output[:true_label], output[true_label+1:]])
                        violation_per_class = wrong_logits - output[true_label]
                        violation = torch.max(violation_per_class)
                    else:
                        # No clear objective, just return center
                        return counterexample.detach()
                
                # Track best violation
                if violation.item() > best_violation:
                    best_violation = violation.item()
                    best_counterexample = counterexample.clone().detach()
                
                # Maximize violation (gradient ascent)
                loss = -violation
                loss.backward()
                optimizer.step()
            
            # Final projection
            with torch.no_grad():
                best_counterexample = torch.clamp(best_counterexample, input_lb, input_ub)
            
            print(f"🔍 Counterexample extracted with violation score: {best_violation:.6f}")
            return best_counterexample
            
        except Exception as e:
            print(f"❌ Counterexample extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def verify(self) -> VerificationStatus:
        print("Starting Symbolic Interval verification pipeline")

        num_samples = self.input_center.shape[0] if self.input_center.ndim > 1 else 1
        print(f"Total samples: {num_samples}")
        results = []
        
        for idx in range(num_samples):
            print(f"\n🔍 Processing sample {idx+1}/{num_samples}")
            print("="*80)

            center_input, true_label = self.get_sample_center_and_label(idx)
            if not self.check_clean_prediction(center_input, true_label, idx):
                print(f"⏭️  Skipping verification for sample {idx+1}")
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
            else:
                self.clean_prediction_stats['verification_unknown'] += 1

            if initial_verdict == VerificationStatus.UNSAT:
                print(f"❌ Symbolic Interval potential violation detected - Sample {idx+1}")
            else:
                print(f"❓ Symbolic Interval inconclusive - Sample {idx+1}")

            print("Launching Specification Refinement BaB process")
            print("="*60)

            if self.bab_config['enabled']:
                refinement_verdict = self._spec_refinement_verification(lb_i, ub_i, idx)
                if refinement_verdict == VerificationStatus.SAT:
                    self.clean_prediction_stats['verification_sat'] += 1
                elif refinement_verdict == VerificationStatus.UNSAT:
                    self.clean_prediction_stats['verification_unsat'] += 1
                else:
                    self.clean_prediction_stats['verification_unknown'] += 1
                results.append(refinement_verdict)
            else:
                print("⚠️  BaB disabled, returning initial verdict")
                results.append(initial_verdict)

        self.print_verification_stats()
        return self._all_results_verdict(results)
