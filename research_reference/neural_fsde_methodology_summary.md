# Neural Fractional Stochastic Differential Equations (fSDEs): Methodology Summary

## Overview

This document summarizes the methodological approaches for implementing neural fractional stochastic differential equations (fSDEs) based on recent research papers. The focus is on practical implementation strategies for generating time series with long-term memory and fractal properties using neural networks combined with fractional Brownian motion.

## 1. Core Theoretical Foundation

### 1.1 Fractional Stochastic Differential Equations

**Traditional Neural SDEs:**
```
dX(t) = f(X(t), t, θ) dt + g(X(t), t, θ) dW(t)
```

**Fractional Neural SDEs (fSDE-Net):**
```
dX(t) = f(X(t), t, θ) dt + g(X(t), t, θ) dB^H(t)
```

Where:
- `f(X(t), t, θ)` is the drift function parameterized by neural networks
- `g(X(t), t, θ)` is the diffusion function parameterized by neural networks  
- `B^H(t)` is fractional Brownian motion with Hurst parameter H
- `θ` represents neural network parameters

### 1.2 Key Mathematical Properties

**Hurst Index Characteristics:**
- `H > 0.5`: Long-range dependence (persistent)
- `H < 0.5`: Roughness/anti-persistence 
- `H = 0.5`: Standard Brownian motion

**Long-Range Dependence (LRD):**
- Autocorrelation function decays polynomially: `ρ(k) ~ k^(2H-2)`
- Power spectral density follows: `S(f) ~ f^(-(2H-1))`

## 2. Neural fSDE Architecture (fSDE-Net)

### 2.1 Core Components

**Drift Network:** `f_θ(x, t)`
- Deep neural network approximating the deterministic component
- Input: Current state `x(t)` and time `t`
- Output: Drift vector

**Diffusion Network:** `g_φ(x, t)` 
- Neural network parameterizing the stochastic component
- Input: Current state `x(t)` and time `t`
- Output: Diffusion coefficient matrix

**Fractional Noise Generator:**
- Generates fractional Brownian motion samples `B^H(t)`
- Uses methods like Cholesky decomposition or circulant matrix approach
- Parameterized by learnable Hurst index `H`

### 2.2 Architecture Design

```python
class fSDENet(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_layers):
        self.drift_net = MLP(state_dim + 1, hidden_dim, state_dim, num_layers)
        self.diffusion_net = MLP(state_dim + 1, hidden_dim, state_dim, num_layers)
        self.hurst_param = nn.Parameter(torch.tensor(0.7))  # Learnable H
        
    def forward(self, x, t, dt):
        # Drift component
        drift = self.drift_net(torch.cat([x, t], dim=-1))
        
        # Diffusion component  
        diffusion = self.diffusion_net(torch.cat([x, t], dim=-1))
        
        # Fractional Brownian motion increment
        fbm_increment = self.generate_fbm_increment(dt, self.hurst_param)
        
        # SDE step
        x_next = x + drift * dt + diffusion * fbm_increment
        return x_next
```

## 3. Latent Fractional Net (Lf-Net) Extension

### 3.1 Latent Space Formulation

**Key Innovation:** Operating in latent space to capture complex temporal dependencies

**Architecture:**
```
Encoder: x(t) → z(t)
fSDE in latent space: dZ(t) = f_θ(Z(t), t) dt + g_φ(Z(t), t) dB^H(t)  
Decoder: z(t) → x̂(t)
```

**Components:**
- **Encoder Network:** Maps observations to latent fractional process
- **Latent fSDE:** Evolves latent states with fractional dynamics
- **Decoder Network:** Reconstructs observations from latent states

### 3.2 Implementation Strategy

```python
class LatentFractionalNet(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim):
        self.encoder = Encoder(obs_dim, latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, obs_dim, hidden_dim)
        self.fsde = fSDENet(latent_dim, hidden_dim, num_layers=3)
        
    def forward(self, x_seq, time_points):
        # Encode to latent space
        z_0 = self.encoder(x_seq[0])
        
        # Evolve in latent space using fSDE
        z_trajectory = self.integrate_fsde(z_0, time_points)
        
        # Decode back to observation space
        x_pred = self.decoder(z_trajectory)
        return x_pred
```

## 4. Numerical Integration Methods

### 4.1 Euler-Maruyama Scheme for fSDEs

**Discretization:**
```
X_{n+1} = X_n + f(X_n, t_n, θ) Δt + g(X_n, t_n, θ) ΔB^H_n
```

Where `ΔB^H_n` are fractional Brownian motion increments.

### 4.2 Fractional Brownian Motion Generation

**Method 1: Cholesky Decomposition**
```python
def generate_fbm_path(n_steps, hurst, dt):
    # Covariance matrix for fBm
    times = torch.arange(n_steps) * dt
    K = 0.5 * (times[:, None]^(2*hurst) + times[None, :]^(2*hurst) - 
               torch.abs(times[:, None] - times[None, :])^(2*hurst))
    
    # Cholesky decomposition
    L = torch.linalg.cholesky(K)
    
    # Generate fBm path
    Z = torch.randn(n_steps)
    fbm_path = L @ Z
    return fbm_path
```

**Method 2: Circulant Matrix Approach (more efficient)**
```python
def generate_fbm_increments(n_steps, hurst, dt):
    # More efficient for large sequences
    # Uses FFT-based circulant matrix embedding
    # Implementation details in Dieker (2004)
    pass
```

### 4.3 Convergence Properties

**Theoretical Guarantees:**
- Existence and uniqueness of solutions proven under Lipschitz conditions
- Convergence rate: `O(Δt^H)` for H > 0.5
- Numerical stability requires careful treatment of fractional increments

## 5. Training Methodology

### 5.1 Loss Function Design

**Reconstruction Loss:**
```python
def reconstruction_loss(x_true, x_pred):
    return F.mse_loss(x_pred, x_true)
```

**Distributional Matching Loss:**
```python
def distributional_loss(x_true, x_pred):
    # Match higher-order moments
    loss = 0
    for k in range(1, 5):  # Up to 4th moment
        moment_true = torch.mean(x_true ** k)
        moment_pred = torch.mean(x_pred ** k)
        loss += F.mse_loss(moment_pred, moment_true)
    return loss
```

**Hurst Parameter Regularization:**
```python
def hurst_regularization(hurst_param, target_hurst=None):
    if target_hurst is not None:
        return F.mse_loss(hurst_param, target_hurst)
    else:
        # Encourage realistic Hurst values (0.1, 0.9)
        return torch.relu(0.1 - hurst_param) + torch.relu(hurst_param - 0.9)
```

**Total Loss:**
```python
def total_loss(x_true, x_pred, hurst_param, target_hurst=None):
    recon_loss = reconstruction_loss(x_true, x_pred)
    dist_loss = distributional_loss(x_true, x_pred)
    hurst_reg = hurst_regularization(hurst_param, target_hurst)
    
    return recon_loss + 0.1 * dist_loss + 0.01 * hurst_reg
```

### 5.2 Training Procedure

**1. Data Preparation:**
```python
def prepare_training_data(time_series, sequence_length, overlap=0.5):
    sequences = []
    step = int(sequence_length * (1 - overlap))
    
    for i in range(0, len(time_series) - sequence_length, step):
        seq = time_series[i:i + sequence_length]
        sequences.append(seq)
    
    return torch.stack(sequences)
```

**2. Training Loop:**
```python
def train_fsde_net(model, data_loader, num_epochs, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        for batch in data_loader:
            x_seq, time_points = batch
            
            # Forward pass
            x_pred = model(x_seq, time_points)
            
            # Compute loss
            loss = total_loss(x_seq, x_pred, model.fsde.hurst_param)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Clip Hurst parameter to valid range
            with torch.no_grad():
                model.fsde.hurst_param.clamp_(0.01, 0.99)
```

## 6. Implementation Considerations

### 6.1 Computational Challenges

**Memory Complexity:**
- fBm generation: O(n²) for Cholesky, O(n log n) for circulant
- Neural networks: Standard complexity
- Overall: Dominated by fBm generation for long sequences

**Numerical Stability:**
- Small Hurst values (H < 0.2) can cause numerical issues
- Large time steps may violate SDE assumptions
- Gradient explosion in diffusion networks

### 6.2 Practical Solutions

**Adaptive Time Stepping:**
```python
def adaptive_dt(x, drift, diffusion, max_dt=0.01):
    # Adjust time step based on gradient magnitude
    grad_norm = torch.norm(drift) + torch.norm(diffusion)
    dt = min(max_dt, 0.1 / (grad_norm + 1e-8))
    return dt
```

**Gradient Clipping:**
```python
# In training loop
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Batch Processing for fBm:**
```python
def batch_fbm_generation(batch_size, n_steps, hurst, dt):
    # Generate multiple fBm paths efficiently
    # Reuse Cholesky decomposition across batch
    pass
```

## 7. Model Architecture Variants

### 7.1 Multi-Scale fSDE

**Concept:** Different Hurst parameters for different time scales

```python
class MultiScalefSDENet(nn.Module):
    def __init__(self, scales=[1, 2, 4], state_dim=1):
        self.scales = scales
        self.fsde_modules = nn.ModuleList([
            fSDENet(state_dim) for _ in scales
        ])
        self.fusion_net = nn.Linear(len(scales) * state_dim, state_dim)
    
    def forward(self, x, t, dt):
        outputs = []
        for i, (scale, fsde) in enumerate(zip(self.scales, self.fsde_modules)):
            scaled_dt = dt * scale
            output = fsde(x, t, scaled_dt)
            outputs.append(output)
        
        # Fuse multi-scale outputs
        fused = self.fusion_net(torch.cat(outputs, dim=-1))
        return fused
```

### 7.2 Conditional fSDE

**For controlled generation with specific Hurst parameters:**

```python
class ConditionalfSDENet(nn.Module):
    def __init__(self, state_dim, condition_dim, hidden_dim):
        self.condition_encoder = nn.Linear(condition_dim, hidden_dim)
        self.drift_net = MLP(state_dim + hidden_dim + 1, hidden_dim, state_dim)
        self.diffusion_net = MLP(state_dim + hidden_dim + 1, hidden_dim, state_dim)
    
    def forward(self, x, t, dt, conditions):
        # Encode conditions (e.g., target Hurst parameter)
        cond_embed = self.condition_encoder(conditions)
        
        # Concatenate state, time, and conditions
        inputs = torch.cat([x, t, cond_embed], dim=-1)
        
        drift = self.drift_net(inputs)
        diffusion = self.diffusion_net(inputs)
        
        # Use condition to determine fBm properties
        hurst = conditions[..., 0]  # Assume first condition is Hurst
        fbm_increment = self.generate_conditional_fbm(dt, hurst)
        
        x_next = x + drift * dt + diffusion * fbm_increment
        return x_next
```

## 8. Evaluation Methodology

### 8.1 Quantitative Metrics

**Hurst Parameter Estimation:**
```python
def evaluate_hurst_preservation(generated_data, true_hurst):
    # Use DFA, R/S, or other estimators from our project
    estimated_hurst = dfa_estimator.estimate(generated_data)
    hurst_error = abs(estimated_hurst - true_hurst)
    return hurst_error
```

**Long-Range Dependence:**
```python
def evaluate_lrd(data, max_lag=100):
    autocorr = compute_autocorrelation(data, max_lag)
    # Check polynomial decay
    lags = np.arange(1, max_lag + 1)
    decay_rate = np.polyfit(np.log(lags), np.log(autocorr), 1)[0]
    return decay_rate
```

**Distributional Properties:**
```python
def evaluate_distributions(real_data, generated_data):
    # Wasserstein distance
    wd = wasserstein_distance(real_data, generated_data)
    
    # KS test
    ks_stat, ks_pvalue = kstest(generated_data, real_data)
    
    return {'wasserstein': wd, 'ks_stat': ks_stat, 'ks_pvalue': ks_pvalue}
```

### 8.2 Qualitative Assessment

**Visual Inspection:**
- Time series plots comparing real vs generated
- Autocorrelation function plots
- Power spectral density comparison
- Scaling behavior visualization

## 9. Integration with Existing Project

### 9.1 Model Interface

**Consistency with Project Structure:**
```python
class NeuralFSDEModel(BaseModel):
    def __init__(self, hurst_parameter=0.7, **kwargs):
        super().__init__()
        self.hurst_parameter = hurst_parameter
        self.fsde_net = fSDENet(**kwargs)
        
    def simulate(self, n_samples, **kwargs):
        # Generate fractional time series
        return self.fsde_net.generate(n_samples)
    
    def fit(self, data, **kwargs):
        # Train the neural fSDE on data
        return self.fsde_net.fit(data)
```

### 9.2 Estimator Integration

**Use existing estimators for validation:**
```python
def validate_neural_fsde(model, validation_data):
    # Generate synthetic data
    synthetic_data = model.simulate(len(validation_data))
    
    # Use project estimators for comparison
    estimators = {
        'DFA': DFAEstimator(),
        'R/S': RSEstimator(), 
        'Wavelet': WaveletVarianceEstimator()
    }
    
    results = {}
    for name, estimator in estimators.items():
        real_hurst = estimator.estimate(validation_data)
        synth_hurst = estimator.estimate(synthetic_data)
        results[name] = {
            'real': real_hurst,
            'synthetic': synth_hurst,
            'error': abs(real_hurst - synth_hurst)
        }
    
    return results
```

## 10. Implementation Roadmap

### Phase 1: Basic fSDE-Net
1. Implement fractional Brownian motion generator
2. Create basic neural SDE architecture
3. Implement Euler-Maruyama solver for fSDEs
4. Test on simple synthetic data

### Phase 2: Advanced Features
1. Implement Latent Fractional Net architecture
2. Add multi-scale and conditional variants
3. Optimize numerical methods for efficiency
4. Comprehensive evaluation framework

### Phase 3: Integration
1. Integrate with existing project structure
2. Add to model library alongside fBm, ARFIMA, etc.
3. Create comprehensive demos and documentation
4. Performance comparison with traditional methods

### Phase 4: Applications
1. Real-world data experiments
2. Robustness testing with contamination models
3. Performance benchmarking
4. User guides and tutorials

## References

1. **Hayashi, K., & Nakagawa, K. (2022).** fSDE-Net: Generating Time Series Data with Long-term Memory. arXiv:2201.05974.

2. **Nakagawa, K., & Hayashi, K. (2024).** Lf-Net: Generating Fractional Time-Series with Latent Fractional-Net. IJCNN 2024.

3. **Guerra, J., & Nualart, D. (2008).** Stochastic differential equations driven by fractional Brownian motion and standard Brownian motion. Stochastics and Dynamics, 8(4), 609-641.

4. **Jien, Y.-J., & Ma, J. (2009).** Stochastic differential equations driven by fractional Brownian motions. Bernoulli, 15(3), 846-870.

5. **Csanády, B., et al. (2024).** Parameter Estimation of Long Memory Stochastic Processes. Frontiers in Artificial Intelligence and Applications, 381, 2548-2559.

---

*This methodology summary provides the theoretical foundation and practical implementation strategies for neural fractional stochastic differential equations. The approaches outlined here should enable the development of sophisticated time series generators that capture long-range dependence and fractal properties in real-world data.*

