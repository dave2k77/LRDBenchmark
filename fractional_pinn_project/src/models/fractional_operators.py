"""
Fractional Calculus Operators using hpfracc library

This module provides fractional calculus operators for use in Physics-Informed Neural Networks.
It integrates with the hpfracc library to provide authentic fractional derivatives and integrals.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional, Tuple
import warnings

# Import hpfracc modules
try:
    import core as hpfracc_core
    import special as hpfracc_special
    HPFRACC_AVAILABLE = True
except ImportError:
    warnings.warn("hpfracc library not available. Using placeholder implementations.")
    HPFRACC_AVAILABLE = False


class FractionalOperator(nn.Module):
    """
    Base class for fractional calculus operators.
    
    This class provides a unified interface for different types of fractional
    derivatives and integrals that can be used in neural networks.
    """
    
    def __init__(self, order: float, definition_type: str = "caputo"):
        """
        Initialize the fractional operator.
        
        Args:
            order: Fractional order (alpha) of the derivative/integral
            definition_type: Type of fractional definition ("caputo", "riemann_liouville", "weyl", "marchaud")
        """
        super().__init__()
        self.order = order
        self.definition_type = definition_type
        
        if HPFRACC_AVAILABLE:
            self._setup_hpfracc_operator()
        else:
            self._setup_placeholder_operator()
    
    def _setup_hpfracc_operator(self):
        """Setup the operator using hpfracc library."""
        try:
            if self.definition_type == "caputo":
                self.operator = hpfracc_core.CaputoDefinition(order=self.order)
            elif self.definition_type == "riemann_liouville":
                self.operator = hpfracc_core.FractionalDefinition(
                    order=self.order, 
                    definition_type=hpfracc_core.DefinitionType.RIEMANN_LIOUVILLE
                )
            elif self.definition_type == "weyl":
                self.operator = hpfracc_core.FractionalDefinition(
                    order=self.order,
                    definition_type=hpfracc_core.DefinitionType.WEYL
                )
            elif self.definition_type == "marchaud":
                self.operator = hpfracc_core.FractionalDefinition(
                    order=self.order,
                    definition_type=hpfracc_core.DefinitionType.MARCHAUD
                )
            else:
                raise ValueError(f"Unknown definition type: {self.definition_type}")
                
        except Exception as e:
            warnings.warn(f"Failed to setup hpfracc operator: {e}. Using placeholder.")
            self._setup_placeholder_operator()
    
    def _setup_placeholder_operator(self):
        """Setup a placeholder operator when hpfracc is not available."""
        self.operator = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the fractional operator to the input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length) or (batch_size, sequence_length, features)
            
        Returns:
            Tensor with the same shape as input, containing the fractional derivative/integral
        """
        if self.operator is None:
            return self._placeholder_forward(x)
        
        # Convert to numpy for hpfracc processing
        if x.dim() == 2:
            # (batch_size, sequence_length)
            batch_size, seq_len = x.shape
            result = torch.zeros_like(x)
            
            for i in range(batch_size):
                x_np = x[i].detach().cpu().numpy()
                try:
                    result_np = self.operator.compute(x_np)
                    result[i] = torch.tensor(result_np, dtype=x.dtype, device=x.device)
                except Exception as e:
                    warnings.warn(f"hpfracc computation failed for batch {i}: {e}")
                    result[i] = self._placeholder_forward(x[i:i+1])[0]
            
            return result
            
        elif x.dim() == 3:
            # (batch_size, sequence_length, features)
            batch_size, seq_len, features = x.shape
            result = torch.zeros_like(x)
            
            for i in range(batch_size):
                for j in range(features):
                    x_np = x[i, :, j].detach().cpu().numpy()
                    try:
                        result_np = self.operator.compute(x_np)
                        result[i, :, j] = torch.tensor(result_np, dtype=x.dtype, device=x.device)
                    except Exception as e:
                        warnings.warn(f"hpfracc computation failed for batch {i}, feature {j}: {e}")
                        result[i, :, j] = self._placeholder_forward(x[i:i+1, :, j:j+1])[0, :, 0]
            
            return result
        else:
            raise ValueError(f"Input tensor must be 2D or 3D, got {x.dim()}D")
    
    def _placeholder_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Placeholder implementation when hpfracc is not available.
        
        This provides a basic approximation of fractional derivatives using
        finite differences and power-law scaling.
        """
        if x.dim() == 2:
            # Simple approximation using finite differences
            diff = torch.diff(x, dim=1, prepend=x[:, :1])
            # Apply power-law scaling based on order
            scale_factor = torch.pow(torch.arange(1, x.shape[1] + 1, dtype=x.dtype, device=x.device), -self.order)
            return diff * scale_factor.unsqueeze(0)
        else:
            # For 3D tensors, apply to each feature
            batch_size, seq_len, features = x.shape
            result = torch.zeros_like(x)
            for j in range(features):
                result[:, :, j] = self._placeholder_forward(x[:, :, j])
            return result


class MarchaudDerivative(FractionalOperator):
    """Marchaud fractional derivative operator."""
    
    def __init__(self, order: float):
        super().__init__(order, definition_type="marchaud")


class WeylDerivative(FractionalOperator):
    """Weyl fractional derivative operator."""
    
    def __init__(self, order: float):
        super().__init__(order, definition_type="weyl")


class CaputoDerivative(FractionalOperator):
    """Caputo fractional derivative operator."""
    
    def __init__(self, order: float):
        super().__init__(order, definition_type="caputo")


class RiemannLiouvilleDerivative(FractionalOperator):
    """Riemann-Liouville fractional derivative operator."""
    
    def __init__(self, order: float):
        super().__init__(order, definition_type="riemann_liouville")


class FractionalIntegral(FractionalOperator):
    """Fractional integral operator."""
    
    def __init__(self, order: float):
        # For integrals, we use negative order
        super().__init__(-order, definition_type="riemann_liouville")


class HybridFractionalOperator(nn.Module):
    """
    Hybrid fractional operator that combines multiple definitions.
    
    This operator can switch between different fractional definitions
    or combine them for enhanced performance.
    """
    
    def __init__(self, order: float, operators: list = None):
        """
        Initialize the hybrid operator.
        
        Args:
            order: Fractional order
            operators: List of operator types to combine
        """
        super().__init__()
        self.order = order
        
        if operators is None:
            operators = ["caputo", "marchaud", "weyl"]
        
        self.operators = nn.ModuleList([
            FractionalOperator(order, op_type) for op_type in operators
        ])
        
        # Learnable weights for combining operators
        self.weights = nn.Parameter(torch.ones(len(operators)) / len(operators))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the hybrid operator.
        
        Args:
            x: Input tensor
            
        Returns:
            Weighted combination of different fractional operators
        """
        results = []
        for operator in self.operators:
            results.append(operator(x))
        
        # Normalize weights
        weights = torch.softmax(self.weights, dim=0)
        
        # Combine results
        result = torch.zeros_like(x)
        for i, res in enumerate(results):
            result += weights[i] * res
        
        return result


class AdaptiveFractionalOperator(nn.Module):
    """
    Adaptive fractional operator that learns the optimal order.
    
    This operator can adapt its fractional order based on the input data.
    """
    
    def __init__(self, initial_order: float = 0.5, min_order: float = 0.1, max_order: float = 0.9):
        """
        Initialize the adaptive operator.
        
        Args:
            initial_order: Initial fractional order
            min_order: Minimum allowed order
            max_order: Maximum allowed order
        """
        super().__init__()
        self.min_order = min_order
        self.max_order = max_order
        
        # Learnable order parameter
        self.order_param = nn.Parameter(torch.tensor(initial_order))
        
        # Base operators for different orders
        self.base_operators = nn.ModuleDict({
            "caputo": CaputoDerivative(initial_order),
            "marchaud": MarchaudDerivative(initial_order),
            "weyl": WeylDerivative(initial_order)
        })
    
    @property
    def current_order(self) -> float:
        """Get the current fractional order."""
        return torch.clamp(self.order_param, self.min_order, self.max_order).item()
    
    def forward(self, x: torch.Tensor, operator_type: str = "caputo") -> torch.Tensor:
        """
        Apply the adaptive operator.
        
        Args:
            x: Input tensor
            operator_type: Type of operator to use
            
        Returns:
            Result of the adaptive fractional operator
        """
        current_order = self.current_order
        
        # Update the base operator with current order
        if operator_type == "caputo":
            self.base_operators["caputo"] = CaputoDerivative(current_order)
            return self.base_operators["caputo"](x)
        elif operator_type == "marchaud":
            self.base_operators["marchaud"] = MarchaudDerivative(current_order)
            return self.base_operators["marchaud"](x)
        elif operator_type == "weyl":
            self.base_operators["weyl"] = WeylDerivative(current_order)
            return self.base_operators["weyl"](x)
        else:
            raise ValueError(f"Unknown operator type: {operator_type}")


# Utility functions for fractional calculus
def hurst_to_fractional_order(hurst: float) -> float:
    """
    Convert Hurst parameter to fractional order.
    
    Args:
        hurst: Hurst parameter (0 < H < 1)
        
    Returns:
        Fractional order alpha
    """
    return 2 * hurst - 1


def fractional_order_to_hurst(alpha: float) -> float:
    """
    Convert fractional order to Hurst parameter.
    
    Args:
        alpha: Fractional order
        
    Returns:
        Hurst parameter H
    """
    return (alpha + 1) / 2


def validate_fractional_order(order: float) -> bool:
    """
    Validate that the fractional order is in a reasonable range.
    
    Args:
        order: Fractional order to validate
        
    Returns:
        True if valid, False otherwise
    """
    return -1.0 <= order <= 2.0
