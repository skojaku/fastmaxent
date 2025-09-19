"""
Adam Optimizer with Numba Acceleration

This module implements the Adam optimization algorithm with Numba JIT compilation
for fast parameter optimization in maximum entropy network inference.

References:
    Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
    arXiv preprint arXiv:1412.6980.
"""

import numpy as np
from numba import njit, jit
from numba.experimental import jitclass
from numba import float64, int64, boolean, types


# Define the specification for the JIT-compiled class
adam_spec = [
    ('lr', float64),
    ('beta1', float64),
    ('beta2', float64),
    ('eps', float64),
    ('weight_decay', float64),
    ('amsgrad', boolean),
    ('m', float64[:]),
    ('v', float64[:]),
    ('v_max', float64[:]),
    ('t', int64),
]

@jitclass(adam_spec)
class AdamOptimizerJIT:
    """
    JIT-compiled Adam optimizer class for maximum performance.
    
    This implementation uses Numba's @jitclass decorator for fast execution
    and supports both standard Adam and AMSGrad variants.
    """
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, 
                 weight_decay=0.0, amsgrad=False):
        """
        Initialize the Adam optimizer.
        
        Parameters
        ----------
        lr : float, default=0.001
            Learning rate (step size).
        beta1 : float, default=0.9
            Exponential decay rate for the first moment estimates.
        beta2 : float, default=0.999
            Exponential decay rate for the second moment estimates.
        eps : float, default=1e-8
            Small constant for numerical stability.
        weight_decay : float, default=0.0
            Weight decay (L2 penalty) coefficient.
        amsgrad : bool, default=False
            Whether to use the AMSGrad variant.
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        
        # These will be initialized when first called
        self.m = np.empty(0, dtype=float64)
        self.v = np.empty(0, dtype=float64)
        self.v_max = np.empty(0, dtype=float64)
        self.t = 0
    
    def _initialize_buffers(self, params):
        """Initialize moment buffers if not already initialized."""
        if len(self.m) != len(params):
            self.m = np.zeros(len(params), dtype=float64)
            self.v = np.zeros(len(params), dtype=float64)
            if self.amsgrad:
                self.v_max = np.zeros(len(params), dtype=float64)
            else:
                self.v_max = np.empty(0, dtype=float64)
    
    def step(self, params, grad):
        """
        Perform a single optimization step.
        
        Parameters
        ----------
        params : numpy.ndarray
            Current parameter values.
        grad : numpy.ndarray
            Gradient of the loss function w.r.t. parameters.
            
        Returns
        -------
        numpy.ndarray
            Updated parameter values.
        """
        self._initialize_buffers(params)
        self.t += 1
        
        # Apply weight decay if specified
        if self.weight_decay > 0:
            grad = grad + self.weight_decay * params
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad * grad)
        
        if self.amsgrad:
            # Update max of second moment estimate
            self.v_max = np.maximum(self.v_max, self.v)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v_max / (1 - self.beta2 ** self.t)
        else:
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)
        
        # Update parameters
        params = params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        return params
    
    def reset(self):
        """Reset the optimizer state."""
        self.m = np.zeros_like(self.m)
        self.v = np.zeros_like(self.v)
        if self.amsgrad:
            self.v_max = np.zeros_like(self.v_max)
        self.t = 0


class AdamOptimizer:
    """
    Wrapper class for the JIT-compiled Adam optimizer.
    
    This provides a more convenient interface while maintaining the performance
    benefits of the JIT-compiled implementation.
    """
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, 
                 weight_decay=0.0, amsgrad=False):
        """
        Initialize the Adam optimizer.
        
        Parameters
        ----------
        lr : float, default=0.001
            Learning rate (step size).
        beta1 : float, default=0.9
            Exponential decay rate for the first moment estimates.
        beta2 : float, default=0.999
            Exponential decay rate for the second moment estimates.
        eps : float, default=1e-8
            Small constant for numerical stability.
        weight_decay : float, default=0.0
            Weight decay (L2 penalty) coefficient.
        amsgrad : bool, default=False
            Whether to use the AMSGrad variant.
        """
        self._optimizer = AdamOptimizerJIT(lr, beta1, beta2, eps, weight_decay, amsgrad)
    
    def step(self, params, grad):
        """
        Perform a single optimization step.
        
        Parameters
        ----------
        params : numpy.ndarray
            Current parameter values.
        grad : numpy.ndarray
            Gradient of the loss function w.r.t. parameters.
            
        Returns
        -------
        numpy.ndarray
            Updated parameter values.
        """
        return self._optimizer.step(params, grad)
    
    def reset(self):
        """Reset the optimizer state."""
        self._optimizer.reset()
    
    @property
    def state(self):
        """Get the current optimizer state."""
        return {
            'step': self._optimizer.t,
            'exp_avg': np.array(self._optimizer.m),
            'exp_avg_sq': np.array(self._optimizer.v),
            'max_exp_avg_sq': np.array(self._optimizer.v_max) if len(self._optimizer.v_max) > 0 else None
        }


@njit(fastmath=True, cache=True)
def clip_grad_norm(grad, max_norm):
    """
    Clip gradient norm to prevent gradient explosion.
    
    Parameters
    ----------
    grad : numpy.ndarray
        Gradient vector.
    max_norm : float
        Maximum allowed gradient norm.
        
    Returns
    -------
    numpy.ndarray
        Clipped gradient vector.
    """
    grad_norm = np.sqrt(np.sum(grad * grad))
    if grad_norm > max_norm:
        grad = grad * (max_norm / grad_norm)
    return grad


@njit(fastmath=True, cache=True)
def schedule_lr(initial_lr, step, decay_steps, decay_rate):
    """
    Exponential decay learning rate schedule.
    
    Parameters
    ----------
    initial_lr : float
        Initial learning rate.
    step : int
        Current step number.
    decay_steps : int
        Number of steps between decay events.
    decay_rate : float
        Decay multiplier.
        
    Returns
    -------
    float
        Scheduled learning rate.
    """
    return initial_lr * (decay_rate ** (step // decay_steps))