import numpy as np
import math

class IoTDevice:
    def __init__(self, x, y, D_k, c_k, model_param_size_bits, M, lambda_stale, local_epochs):
        self.x = x
        self.y = y
        self.num_of_data = D_k
        self.battery = 5000
        self.comm_power = 10  #10 ~ 30 dBm
        self.c_k = c_k          # cycles per sample
        self.f_k = np.random.uniform(1e9, 2e9)            # CPU frequency (Hz)
        self.b_k = 10e6 / M  # bandwidth per device
        self.sigma2_dB = -110  # noise power (dBm)
        self.model_param_size_bits = model_param_size_bits
        self.local_epochs = local_epochs
        
        self.last_selected_round = -1  # Track when this device was last selected
        self.last_loss_square = 0.0  # Store the last round's loss square value
        self.lambda_stale = lambda_stale  # UCB temporal bonus coefficient (set from args)

    def get_location(self):
        return np.array([self.x, self.y])
    
    def get_battery(self):
        return self.battery

    def channel_gain(self, uav_pos):
        device_pos = self.get_location()
        d = math.sqrt((device_pos[0] - uav_pos[0])**2 + (device_pos[1] - uav_pos[1])**2 + uav_pos[2]**2)

        K = 0.97  # Rician factor
        alpha_0 = 10**(-60/10)
        c = 3e8
        f_c = 2e9
        eta_LoS = 1.0         # LoS additional loss (dB)
        eta_NLoS = 20.0      # NLoS additional loss (dB)
        pl_los = 20.0*math.log10(4*math.pi*f_c*d/c) + eta_LoS
        pl_nlos = 20.0*math.log10(4*math.pi*f_c*d/c) + eta_NLoS
        pl_los_lin = 10**(-pl_los/10)
        pl_nlos_lin = 10**(-pl_nlos/10)

        small_scale_fading = np.sqrt((K/(K+1))*pl_los_lin) + np.sqrt((1/(K+1))*pl_nlos_lin)
        large_scale_fading = alpha_0 * d**2.2

        return large_scale_fading * small_scale_fading
    
    def achievable_rate(self, uav_pos):
        """
        R_k = b_k * log2(1 + p_k * |h_k|^2 / sigma^2).
        We'll estimate h_k via channel_gain, which includes path loss in linear scale.
        """
        channel_gain = self.channel_gain(uav_pos)
        sigma2 = 10**(self.sigma2_dB / 10)
        comm_power_linear = 10**(self.comm_power / 10)
        snr = comm_power_linear * channel_gain / sigma2
        return self.b_k * math.log2(1.0 + snr)
    
    def get_comm_time(self, uav_pos):
        """
        t_k^{comm} = data_size_bits / R_k
        if R_k is in bits/s
        """
        R_k = self.achievable_rate(uav_pos)
        return self.model_param_size_bits / R_k
    
    def get_comp_time(self):
        return (self.c_k * self.num_of_data * self.local_epochs) / self.f_k

    def get_comm_energy(self, uav_pos):
        """
        E_k^{comm} = (t^{comm}_k * sigma^2 / |h_k|^2) * (2^( (a_k*s)/(b_k*t^{comm}_k )) - 1)
        where:
        - t^{comm}_k: communication time
        - sigma^2: noise power
        - |h_k|^2: channel gain
        - a_k: data size in bits
        - s: data size in bits (same as a_k)
        - b_k: bandwidth
        """
        t_comm = self.get_comm_time(uav_pos)
        channel_gain = self.channel_gain(uav_pos)
        sigma2 = 10**(self.sigma2_dB / 10)  # convert to linear scale
        
        return (t_comm * sigma2 / channel_gain) * (2**(self.model_param_size_bits / (self.b_k * t_comm)) - 1)

    def get_comp_energy(self):
        """
        E_k^{comp} = rho_k/2 * c_k * D_k * f_k^2
        """
        rho_k = 1e-28       # effective capacitance coefficient
        return (rho_k / 2.0) * self.c_k * self.num_of_data * (self.f_k**2)
    

    def compute_utility(self, current_round):
        """
        Compute UCB-style saturated utility:
        U_k(r) = D_k * sqrt((1/D_k) * sum(l^2(x_ki, y_ki))) 
                + lambda * sqrt((log r) * (delta / (1 + delta)))

        where:
        - D_k: number of data samples
        - last_loss_square: sum of squared losses from last round
        - delta = current_round - last_selected_round (staleness)
        - lambda: stale term weight (self.lambda_stale)
        """
        D_k = self.num_of_data
        current_round = current_round + 1  # round index shift (1-based)
        
        # Calculate staleness delta_k(r)
        delta = current_round - self.last_selected_round
        
        # Saturated UCB-style temporal bonus
        temporal_bonus = self.lambda_stale * math.sqrt(
            math.log(current_round) * (delta / (1.0 + delta))
        )
        
        # Main utility (performance + exploration bonus)
        utility = D_k * math.sqrt(self.last_loss_square / D_k) + temporal_bonus
        
        return utility
