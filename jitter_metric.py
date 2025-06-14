import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import entropy
from numpy.fft import rfft, rfftfreq

def velocity(data, dt):
  return np.gradient(data, dt)

def velocity_series(coords):
    return np.linalg.norm(np.diff(coords, axis=0), axis=1)

def comparative_velocity_entropy(pred, gt, dt = 1/100, epsilon=1e-6):
    # v_pred = velocity_series(pred)
    # v_gt = velocity_series(gt)
    v_pred = velocity(pred, dt)
    v_gt = velocity(gt, dt)

    # Histogram-based probability distributions
    bins = np.histogram_bin_edges(np.concatenate([v_pred, v_gt]), bins='fd')
    p_pred, _ = np.histogram(v_pred, bins=bins, density=True)
    p_gt, _ = np.histogram(v_gt, bins=bins, density=True)

    # Add epsilon to avoid log(0)
    p_pred += epsilon
    p_gt += epsilon

    # Normalize to probability distributions
    p_pred /= np.sum(p_pred)
    p_gt /= np.sum(p_gt)

    kl_div = entropy(p_pred, p_gt)
    log_normalized_kl = np.log1p(kl_div)  # log(1 + KL)
    return log_normalized_kl

def sparc_1d(signal, delta_t=1/100, epsilon=1e-6):
    vel = np.gradient(signal, delta_t)
    fft_vals = np.abs(rfft(vel))
    freqs = rfftfreq(len(vel), d=delta_t)
    spectrum = fft_vals / (np.sum(fft_vals) + epsilon)
    sparc_val = -np.sum(np.log(freqs[1:] + epsilon) * spectrum[1:])  # skip DC
    return sparc_val

def comparative_sparc(pred, gt):
    sparc_pred = sparc_1d(pred) #np.mean([sparc_1d(pred[:, i]) for i in range(2)])
    sparc_gt = sparc_1d(gt) #np.mean([sparc_1d(gt[:, i]) for i in range(2)])
    return abs(sparc_pred - sparc_gt) / (abs(sparc_gt) + 1e-6)

def compute_combined_score(pred, gt, w_sparc=0.75, w_cve=0.25):
    norm_cve = comparative_velocity_entropy(pred, gt)
    norm_sparc = comparative_sparc(pred, gt)
    score = w_sparc * norm_sparc + w_cve * norm_cve
    return score , norm_sparc, norm_cve

def add_composite_noise(y_true, noise_params):
    """
    Adds at least two randomly chosen noise types (Gaussian, blinks, shifts).
    
    Parameters:
        y_true (np.array): Ground truth signal.
        noise_params (dict): Contains keys for all noise types:
            - 'gaussian_intensity': Std of Gaussian noise.
            - 'blink_intensity': Scale of blink spikes.
            - 'blink_prob': Probability of a blink.
            - 'shift_intensity': Constant shift.
    
    Returns:
        y_noisy (np.array): Noisy signal with ≥2 noise types.
    """
    y_noisy = y_true.copy()
    n = len(y_true)
    noise_types = []

    # Randomly choose 2-3 noise types
    chosen_types = np.random.choice(
        ['gaussian','blink', 'shift'], #'gaussian', 
        size=np.random.randint(1, 3),  # Pick 1 or 2 types
        replace=False
    )

    y_noisy += np.random.normal(0, noise_params['gaussian_intensity'], n)
    noise_types.append('gaussian')
    
    if 'blink' in chosen_types:
        blink_mask = np.random.rand(n) < noise_params['blink_prob']
        y_noisy[blink_mask] += noise_params['blink_intensity'] * np.random.randn(np.sum(blink_mask))
        noise_types.append('blink')
    
    if 'shift' in chosen_types:
        y_noisy += noise_params['shift_intensity']
        noise_types.append('shift')

    mandatory_noise = noise_params.get('mandatory_intensity', 2) * np.sin(50 * np.linspace(0, 10, n))  # Example: 50Hz jitter
    y_noisy += mandatory_noise
    noise_types.append('mandatory_jitter')  # Track mandatory noise

    return y_noisy #, noise_types  # Return noise types for debugging

def generate_prediction(y_true, target_mse, target_noise, noise_tolerance=0.1, max_iter=1000000):
    """
    Generates a noisy prediction that meets MSE (X) and noisiness (Y) conditions.
    Uses composite noise and iterative adjustment.
    """
    for _ in range(max_iter):
        # Randomize noise parameters (adjust ranges as needed)
        noise_params = {
            'gaussian_intensity': np.random.uniform(0.5, 5),
            'blink_intensity': np.random.uniform(2, 50.0),
            'blink_prob': np.random.uniform(0.01, 0.1),
            'shift_intensity': np.random.uniform(-10, 10)
        }
        
        # Add composite noise
        y_pred = add_composite_noise(y_true, noise_params)
        
        current_mse = np.mean((y_true - y_pred) ** 2)
        current_noise, norm_sparc, norm_cve = compute_combined_score(y_pred, y_true) 
        
        # Check if conditions are met within tolerance
        if (abs(current_mse - target_mse) < noise_tolerance * target_mse and
            abs(current_noise - target_noise) < noise_tolerance * target_noise):
            print(current_mse, current_noise, norm_sparc, norm_cve)
            return y_pred
    
    print(f"Warning: Max iterations reached for target MSE={target_mse}, Noise={target_noise}")
    return y_pred

X = 15  # Target MSE
Y = 0.2  # Target smoothness/noisiness

y_true = true_labels[400:480,0] # ground truth data

print("pred_1")
# Case (a): MSE ≈ X, Noisiness ≈ Y
pred_1 = generate_prediction(y_true, X, Y)

print("pred_2")
# Case (b): Higher noise, MSE < X
pred_2 = generate_prediction(y_true, X * 0.7, Y * 1.1)

print("pred_3")
# Case (c): Higher noise, MSE ≈ X
pred_3 = generate_prediction(y_true, X, Y * 1.1)

print("pred_4")
# Case (d): Higher noise, MSE > X
pred_4 = generate_prediction(y_true, X * 4, Y * 1.5)

print("pred_5")
# Case (e): MSE > X, Noisiness < Y
pred_5 = generate_prediction(y_true, X * 1.5, Y * 0.8)

print("pred_6")
# Case (f): MSE ≈ X, Noisiness < Y
pred_6 = generate_prediction(y_true, X, Y * 0.5)

print("pred_7")
# Case (g): MSE < X, Noisiness < Y
pred_7 = generate_prediction(y_true, X * 0.25, Y * 0.25)