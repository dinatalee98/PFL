import numpy as np
import math
import random

# -----------------------------
# 1. Parameter Initialization
# -----------------------------
# Table parameters as a dictionary for convenience
params = {
    "c_k": 10,              # cycles per sample
    "f_k": 5e9,             # CPU frequency (Hz)
    "alpha_k": 1e-28,       # effective capacitance coefficient * 2 (splitting the factor in the code)
    "zeta_1": 9.61,
    "zeta_2": 0.16,
    "H_u": 100.0,           # UAV altitude (m)
    "f_c": 2e9,             # carrier frequency (Hz)
    "eta_LoS": 1.0,         # LoS additional loss (dB)
    "eta_NLoS": 20.0,       # NLoS additional loss (dB)
    "alpha_0": 10**(-60/10),# avg power gain at 1m, convert dBM to linear
    "beta": 2.2,            # path loss exponent
    "K_Rician": 3.0,        # Rician factor
    "sigma2_dB": -110,      # noise power (dBm)
    "b_k": 80e3,            # subchannel bandwidth (Hz)
}

# Convert noise power to linear scale
params["sigma2"] = 10**(params["sigma2_dB"] / 10)

# Additional settings for the simulation
K = 30             # total number of IoT devices
M = 5              # number of OFDM subchannels (max parallel transmissions)
tau = 0.05         # length of a communication slot (in seconds)
N_rounds = 20      # number of global FL rounds
epsilon_init = 1.0 # initial epsilon for epsilon-greedy
epsilon_min = 0.1  # minimum epsilon
decay_rate = 0.9   # decay per round
model_size = 1e5   # size of the model (bits)
max_energy = 2.0   # each device’s initial battery energy (arbitrary units, e.g., Joules)

# -----------------------------------------------
# 2. Generate Devices (positions, compute times)
# -----------------------------------------------
# For simplicity, place each device randomly in a 2D plane
np.random.seed(42)
device_positions = np.random.uniform(low=-300, high=300, size=(K, 2))

# Each device has a dataset size D_k, chosen randomly for illustration
D_k = np.random.randint(low=100, high=1000, size=K)

# Each device can have a random CPU frequency around f_k with some variation
f_k_array = params["f_k"] * (1.0 + 0.1*np.random.rand(K))  # e.g., ±10%

# c_k is the number of cycles per data sample. For simplicity, assume each device uses the same c_k = params["c_k"]
c_k_array = np.full(K, params["c_k"])

# Battery for each device
battery_levels = np.full(K, max_energy)

# Precompute local compute times t_k^{comp} = c_k * D_k / f_k
t_comp = (c_k_array * D_k) / f_k_array

# UAV position (for simplicity, keep it fixed overhead, or you can simulate movement).
# Assume UAV is at (0,0,H_u)
uav_pos = np.array([0.0, 0.0, params["H_u"]])

# --------------------------------------------------------
# 3. Client Clustering (modified K-means or range-based)
# --------------------------------------------------------
def client_clustering(t_comp, tau, M):
    """
    A simplified version of the pseudo-code in your text.
    We cluster clients by sorting them in increasing t_comp,
    determining the number of clusters based on range/tau,
    and grouping them accordingly.
    """
    # Sort clients by compute time
    sorted_indices = np.argsort(t_comp)
    sorted_t_comp = t_comp[sorted_indices]

    # Determine the number of clusters J
    t_min, t_max = np.min(sorted_t_comp), np.max(sorted_t_comp)
    if tau <= 0:
        # fallback
        J = 1
    else:
        J = int(np.ceil((t_max - t_min) / tau))
    if J < 1:
        J = 1
    
    # Balanced cluster size
    n_ideal = max(1, K // J)

    # Prepare cluster list
    clusters = [[] for _ in range(J)]
    deadlines = [0]*J

    i = 0
    theta_prev = 0

    for j in range(J):
        # Fill cluster j up to n_ideal
        while len(clusters[j]) < n_ideal and i < K:
            k_idx = sorted_indices[i]
            clusters[j].append(k_idx)
            i += 1
        
        # Deadline for cluster j
        if len(clusters[j]) > 0:
            max_t_in_cluster = max(t_comp[idx] for idx in clusters[j])
            deadlines[j] = max(max_t_in_cluster, theta_prev + tau)
        else:
            deadlines[j] = theta_prev + tau
        
        # If cluster has fewer than M, add more clients if possible
        while len(clusters[j]) < M and i < K:
            k_idx = sorted_indices[i]
            clusters[j].append(k_idx)
            # update deadline
            deadlines[j] = max(deadlines[j], t_comp[k_idx])
            i += 1
        
        theta_prev = deadlines[j]

    # If there are leftover clients, assign them to the smallest cluster
    while i < K:
        # pick cluster with minimal size
        j_star = np.argmin([len(c) for c in clusters])
        k_idx = sorted_indices[i]
        clusters[j_star].append(k_idx)
        # update deadline
        if j_star == 0:
            deadlines[j_star] = max(deadlines[j_star], t_comp[k_idx])
        else:
            deadlines[j_star] = max(deadlines[j_star], t_comp[k_idx], deadlines[j_star-1] + tau)
        i += 1

    return clusters, deadlines

clusters, cluster_deadlines = client_clustering(t_comp, tau, M)

# --------------------------------------
# 4. Channel/Path-Loss Related Helpers
# --------------------------------------
def path_loss_probability_los(uav_pos, device_pos, params):
    """
    Compute LoS probability P^LoS_k
    P^{LoS}_k = 1/[1 + zeta1*exp(-zeta2( [180/pi]*theta_k - zeta1 ))]
    where theta_k = arctan(H_u/horizontal_distance)
    """
    H_u = uav_pos[2]
    dx = device_pos[0] - uav_pos[0]
    dy = device_pos[1] - uav_pos[1]
    dist_2d = math.sqrt(dx*dx + dy*dy)
    # angle of elevation (in degrees)
    if dist_2d < 1e-12:
        theta_deg = 90.0
    else:
        theta_deg = math.degrees(math.atan(H_u / dist_2d))

    z1 = params["zeta_1"]
    z2 = params["zeta_2"]
    p_los = 1.0 / (1.0 + z1 * math.exp(-z2*(theta_deg - z1)))
    return p_los

def avg_path_loss(uav_pos, device_pos, params):
    """
    PL_k = P^{LoS}_k * PL^{LoS}_k + (1 - P^{LoS}_k)*PL^{NLoS}_k
    PL^{LoS}_k = 20*log10(4*pi*f_c*d_k/c) + eta^{LoS}
    d_k = distance between UAV and device
    """
    # distance
    dx = device_pos[0] - uav_pos[0]
    dy = device_pos[1] - uav_pos[1]
    dz = device_pos[2] - uav_pos[2] if len(device_pos) == 3 else -uav_pos[2] 
    dist_3d = math.sqrt(dx*dx + dy*dy + dz*dz)

    p_los = path_loss_probability_los(uav_pos, device_pos, params)

    c = 3e8
    # LoS path loss
    pl_los = 20.0*math.log10(4*math.pi*params["f_c"]*dist_3d/c) + params["eta_LoS"]
    # NLoS path loss
    pl_nlos = 20.0*math.log10(4*math.pi*params["f_c"]*dist_3d/c) + params["eta_NLoS"]

    # Convert dB to linear scale for weighting
    pl_los_lin = 10**(-pl_los/10)
    pl_nlos_lin = 10**(-pl_nlos/10)

    # Weighted average in linear scale
    pl_avg_lin = p_los*pl_los_lin + (1-p_los)*pl_nlos_lin

    # Convert back to dB
    pl_avg_db = -10*math.log10(pl_avg_lin)
    return pl_avg_db

def channel_gain(uav_pos, device_pos, params):
    """
    Returns channel gain = alpha_0 * d_k^beta * small-scale (Rician factor).
    We'll incorporate the path-loss model above, turning that into a linear gain.
    """
    pl_db = avg_path_loss(uav_pos, device_pos, params)
    # Convert path loss from dB to linear scale
    pl_lin = 10**(-pl_db/10)

    # For simplicity, incorporate Rician factor + small scale
    # \bar{h}_k = sqrt( K/(K+1)*PL^LoS ) + sqrt( 1/(K+1)*PL^NLoS )
    # We'll treat it as a random variable drawn each time or
    # we can treat it as an expectation (which might be simpler).
    # Let's do a random sample from Rician distribution for demonstration.
    K_rician = params["K_Rician"]
    # A quick way to generate Rician random sample (magnitude) is:
    #   v = np.sqrt(K/(K+1)) * np.random.rayleigh(...) is not quite correct.
    # Proper approach: a Rician can be built from two Gaussians, but for simplicity,
    # let's just sample from a standard python function or approximate with a random factor.
    
    # We'll just do a small random factor around 1 for demonstration:
    # e.g. from 0.5 to 1.5
    small_scale_factor = 0.5 + random.random()
    
    # Final gain in linear scale (including path loss)
    # channel_gain = pl_lin * small_scale_factor
    # (Alternatively, we might incorporate alpha_0, etc. However, our path_loss
    # function above already includes large-scale path loss. 
    # If you want alpha_0 and beta in your code, you can do something like:
    # d_3d^beta factor. But our path-loss formula is already covering distance-based attenuation. 
    # So we can do a simpler approach:
    return pl_lin * small_scale_factor

def achievable_rate(uav_pos, device_pos, p_k, b_k, params):
    """
    R_k = b_k * log2(1 + p_k * |h_k|^2 / sigma^2).
    We'll estimate h_k via channel_gain, which includes path loss in linear scale.
    """
    h_lin = channel_gain(uav_pos, device_pos, params)
    snr = p_k * h_lin / params["sigma2"]
    return b_k * math.log2(1.0 + snr)

def comm_time(uav_pos, device_pos, p_k, b_k, data_size_bits, params):
    """
    t_k^{comm} = data_size_bits / R_k
    if R_k is in bits/s
    """
    R_k = achievable_rate(uav_pos, device_pos, p_k, b_k, params)
    if R_k < 1e-12:
        return 1e9  # effectively infinite time
    return data_size_bits / R_k

def comm_energy(uav_pos, device_pos, p_k, b_k, data_size_bits, params):
    """
    E_k^{comm} = p_k * t_k^{comm}, ignoring the formula for power control from the text
    for brevity. If you want the exact formula from the text:
      E_k^{comm} = (t^{comm}_k * sigma^2 / |h_k|^2 ) * (2^( (a_k*s)/(b_k*t^{comm}_k )) - 1)
    This code uses a simpler approach: E = P * time. Replace as needed.
    """
    t_comm = comm_time(uav_pos, device_pos, p_k, b_k, data_size_bits, params)
    return p_k * t_comm

def comp_energy(c_k, D_k, f_k, alpha_k):
    """
    E_k^{comp} = a_k * alpha_k/2 * c_k * D_k * f_k^2
    For simplicity, assume a_k=1 always if computing. We incorporate alpha_k/2 in code directly.
    """
    return (alpha_k * c_k * D_k * (f_k**2)) / 2.0

# --------------------------------------
# 5. Client Selection (epsilon-greedy)
# --------------------------------------
def client_selection(cluster_indices, battery_levels, device_positions, t_comp, 
                     params, model_size_bits, tau, epsilon):
    """
    Perform the selection for a single cluster, based on:
      1) Energy feasibility
      2) Comm feasibility (t_comm < tau)
      3) Utility-based probability selection
    """
    # Compute utility for each client (placeholder: e.g., U_k = D_k * random local loss)
    # In real code, you'd measure the local objective or use your stored "loss".
    # For demonstration, let's do:
    U_list = []
    for k in cluster_indices:
        # Loss can be random or a function of t_comp, etc.
        # Let's just do a random factor to mimic "loss^2" measure.
        # The code from text: U_k = D_k * sqrt( average(loss^2(...)) )
        # We'll do a random approximate:
        rand_loss = random.uniform(0.01, 1.0)
        U_k = D_k[k] * math.sqrt(rand_loss)
        U_list.append(U_k)
    U_array = np.array(U_list)
    
    U_sum = np.sum(U_array) if np.sum(U_array) > 0 else 1e-12
    
    # Step 1 & 2: Filter out clients that do not satisfy energy or time constraints
    selected_flags = np.ones(len(cluster_indices), dtype=int)  # 1 means candidate
    for i, k in enumerate(cluster_indices):
        # Check communication time
        # Assume a fixed transmit power for illustration
        p_k = 0.1  # 0.1 W
        b_k = params["b_k"]

        t_comm_k = comm_time(uav_pos, device_positions[k], p_k, b_k, model_size_bits, params)
        E_comm_k = comm_energy(uav_pos, device_positions[k], p_k, b_k, model_size_bits, params)
        E_comp_k = comp_energy(c_k_array[k], D_k[k], f_k_array[k], params["alpha_k"])

        if (t_comm_k >= tau) or (battery_levels[k] < (E_comm_k + E_comp_k)):
            # Deactivate
            selected_flags[i] = 0

    # Step 3: For the remaining (non-deactivated) clients, do epsilon-greedy
    final_selection = np.zeros(len(cluster_indices), dtype=int)
    if np.sum(selected_flags) == 0:
        # If no one is feasible, all remain 0
        return final_selection
    
    # Probability distribution among feasible
    feasible_indices = np.where(selected_flags == 1)[0]
    U_feasible = U_array[feasible_indices]
    U_feasible_sum = np.sum(U_feasible)
    if U_feasible_sum < 1e-12:
        U_feasible_sum = 1e-12

    for idx in feasible_indices:
        # Normalized utility among feasible
        P_k = U_array[idx] / U_feasible_sum
        # Selection probability: \epsilon P_k + (1-\epsilon) 1/|G_j|
        p_k = epsilon * P_k + (1 - epsilon)*(1.0/len(feasible_indices))
        r = random.random()
        if r <= p_k:
            final_selection[idx] = 1

    return final_selection

# --------------------------------------
# 6. Pipeline Scheduling & FL Execution
# --------------------------------------

# For demonstration, we will do a simple round-robin across clusters:
# In each global round, we iterate over clusters in order and schedule
# the feasible clients. The faster clusters can finish first, and the
# slower ones will "pipeline" behind. We'll keep track of total time
# spent. This is a simplified version to illustrate the concept.

# Initialize global model (placeholder: random vector)
dim_model = 100  # dimension of the model vector
global_model = np.zeros(dim_model)

current_time = 0.0
epsilon = epsilon_init

for n in range(N_rounds):
    print(f"=== Global Round {n+1}/{N_rounds}, epsilon={epsilon:.3f} ===")
    round_time = 0.0  # track the time cost for this round

    # For each cluster in order, do selection & "transmissions"
    for j, cluster_indices in enumerate(clusters):
        # 1) Client selection
        selection_flags = client_selection(cluster_indices, battery_levels, 
                                           device_positions, t_comp, 
                                           params, model_size, tau, epsilon)

        # 2) Pipelined "execution": we assume that the clients in this cluster start
        #    their local computations at cluster_deadlines[j-1] (or 0 if j=0).
        #    However, for simplicity, we do not explicitly measure partial overlaps.
        #    We demonstrate how one might track times:
        if np.sum(selection_flags) == 0:
            continue  # no client selected

        selected_indices = [cluster_indices[i] for i, flag in enumerate(selection_flags) if flag == 1]

        # Each selected client runs local training => t_comp[k]
        # Then it uploads => t_comm[k].
        # We'll track the max time among them, since they can (in principle) share subchannels M, but 
        # it’s an illustration, you can refine.

        max_end_time = 0.0
        for k in selected_indices:
            p_k = 0.1  # transmit power
            b_k = params["b_k"]
            # local comp time:
            comp_t = t_comp[k]
            # comm time:
            comm_t = comm_time(uav_pos, device_positions[k], p_k, b_k, model_size, params)
            # total time for k:
            t_total_k = comp_t + comm_t
            if t_total_k > max_end_time:
                max_end_time = t_total_k

            # energy consumption
            E_comm_k = comm_energy(uav_pos, device_positions[k], p_k, b_k, model_size, params)
            E_comp_k = comp_energy(c_k_array[k], D_k[k], f_k_array[k], params["alpha_k"])
            battery_levels[k] -= (E_comm_k + E_comp_k)
            if battery_levels[k] < 0:
                battery_levels[k] = 0

            # Placeholder: local model update (randomly generated)
            local_update = np.random.randn(dim_model)
            # Accumulate for averaging (basic FedAvg)
            # For demonstration, let's just store them to sum later
            # Weighted by data size
            # We'll store them in a list
        # end for selected clients

        round_time += max_end_time
    
    # After going through all clusters, we do a global aggregation step
    # Placeholder: we might do FedAvg with the local updates from all selected clients
    # For demonstration, let's do a random update:
    # (In a real FL system, you'd collect the local gradients or new local model from each client.)
    global_update = np.random.randn(dim_model) * 0.01
    global_model += global_update

    print(f"Round {n+1} finished. Round time = {round_time:.4f} s.")
    current_time += round_time

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * decay_rate)

print("\nSimulation complete.")
print(f"Total time elapsed: {current_time:.2f} s.")
