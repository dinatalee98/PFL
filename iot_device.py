import numpy as np
import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

class IoTDevice:
    def __init__(self, x, y, D_k, b, dataset):
        self.x = x
        self.y = y
        self.num_of_data = D_k
        self.battery = b
        self.comm_power = 10  #10 ~ 30 dBm
        self.dataset = dataset
    
    def get_location(self):
        return np.array([self.x, self.y])
    
    def get_computation_time(self): # Eq 3 & 4
        c_k = 10              # cycles per sample
        f_k = 5e9             # CPU frequency (Hz)
        data_size_per_sample = 0
        if self.dataset == "mnist":
            data_size_per_sample = 28 * 28
        D_k = self.num_of_data * data_size_per_sample   # data size in bits
        t_comp = (c_k * D_k) / f_k
        return t_comp
    
    def get_battery(self):
        return self.battery
    
    def path_loss_probability_los(self, uav_pos, device_pos):
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

        z1 = 9.61 # "zeta_1"
        z2 = 0.16 # "zeta_2"
        p_los = 1.0 / (1.0 + z1 * math.exp(-z2*(theta_deg - z1)))
        return p_los

    def avg_path_loss(self, uav_pos, device_pos):
        """
        PL_k = P^{LoS}_k * PL^{LoS}_k + (1 - P^{LoS}_k)*PL^{NLoS}_k
        PL^{LoS}_k = 20*log10(4*pi*f_c*d_k/c) + eta^{LoS}
        d_k = distance between UAV and device
        """
        # distance
        dx = device_pos[0] - uav_pos[0]
        dy = device_pos[1] - uav_pos[1]
        dz = - uav_pos[2] 
        dist_3d = math.sqrt(dx*dx + dy*dy + dz*dz)

        p_los = self.path_loss_probability_los(uav_pos, device_pos)

        c = 3e8
        f_c = 2e9
        eta_LoS = 1.0         # LoS additional loss (dB)
        eta_NLoS = 20.0      # NLoS additional loss (dB)
        # LoS path loss
        pl_los = 20.0*math.log10(4*math.pi*f_c*dist_3d/c) + eta_LoS
        # NLoS path loss
        pl_nlos = 20.0*math.log10(4*math.pi*f_c*dist_3d/c) + eta_NLoS

        # Convert dB to linear scale for weighting
        pl_los_lin = 10**(-pl_los/10)
        pl_nlos_lin = 10**(-pl_nlos/10)

        # Weighted average in linear scale
        pl_avg_lin = p_los*pl_los_lin + (1-p_los)*pl_nlos_lin

        # Convert back to dB
        pl_avg_db = -10*math.log10(pl_avg_lin)
        return pl_avg_db

    def channel_gain(self, uav_pos, device_pos):
        """
        Returns channel gain = alpha_0 * d_k^beta * small-scale (Rician factor).
        We'll incorporate the path-loss model above, turning that into a linear gain.
        """
        pl_db = self.avg_path_loss(uav_pos, device_pos)
        # Convert path loss from dB to linear scale
        pl_lin = 10**(-pl_db/10)

        # For simplicity, incorporate Rician factor + small scale
        # \bar{h}_k = sqrt( K/(K+1)*PL^LoS ) + sqrt( 1/(K+1)*PL^NLoS )
        # We'll treat it as a random variable drawn each time or
        # we can treat it as an expectation (which might be simpler).
        # Let's do a random sample from Rician distribution for demonstration.
        K_rician = 3.0  # Rician factor
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
    
    def achievable_rate(self, uav_pos, device_pos, p_k, b_k):
        """
        R_k = b_k * log2(1 + p_k * |h_k|^2 / sigma^2).
        We'll estimate h_k via channel_gain, which includes path loss in linear scale.
        """
        h_lin = self.channel_gain(uav_pos, device_pos)
        sigma2_DB = -110  # -110 dBm noise floor
        sigma2 = 10**(sigma2_DB / 10)
        snr = p_k * h_lin / sigma2
        return b_k * math.log2(1.0 + snr)
    
    def get_commtime(self, uav_pos, data_size_bits):
        """
        t_k^{comm} = data_size_bits / R_k
        if R_k is in bits/s
        """
        device_pos = self.get_location()
        b_k = 80e3  # 80 KHz bandwidth
        p_k = self.comm_power
        R_k = self.achievable_rate(uav_pos, device_pos, p_k, b_k)
        if R_k < 1e-12:
            return 1e9  # effectively infinite time
        return data_size_bits / R_k
    
    def comm_energy(self, uav_pos, data_size_bits):
        """
        E_k^{comm} = p_k * t_k^{comm}, ignoring the formula for power control from the text
        for brevity. If you want the exact formula from the text:
        E_k^{comm} = (t^{comm}_k * sigma^2 / |h_k|^2 ) * (2^( (a_k*s)/(b_k*t^{comm}_k )) - 1)
        This code uses a simpler approach: E = P * time. Replace as needed.
        """
        t_comm = self.get_commtime(uav_pos, data_size_bits)
        p_k = self.comm_power
        return p_k * t_comm

    def get_comp_energy(self):
        """
        E_k^{comp} = a_k * alpha_k/2 * c_k * D_k * f_k^2
        For simplicity, assume a_k=1 always if computing. We incorporate alpha_k/2 in code directly.
        """
        c_k = 10              # cycles per sample
        f_k = 5e9             # CPU frequency (Hz)
        alpha_k = 1e-28       # effective capacitance coefficient * 2 (splitting the factor in the code)
        return (alpha_k * c_k * self.D_k * (f_k**2)) / 2.0
    


    def compute_utility(model, device, dataset_k):
        """
        Compute the utility U_k = D_k * sqrt( (1/D_k) * sum_{d=1..D_k} [loss^2(x_{kd}, y_{kd})] )

        Args:
            model (nn.Module): The PyTorch model to evaluate.
            device (torch.device): The device (CPU/GPU) to run on.
            dataset_k (Dataset): The local dataset for client k, or a subset of data.

        Returns:
            float: The computed utility U_k.
        """
        # Number of samples in local dataset
        D_k = len(dataset_k)
        if D_k == 0:
            return 0.0  # or handle an empty dataset case differently

        # We'll accumulate sum of loss^2
        total_loss_sq = 0.0

        # Create a DataLoader for convenience
        loader = DataLoader(dataset_k, batch_size=32, shuffle=False)

        # Put model in eval mode
        model.eval()

        # Disable gradient computations
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                # Forward pass
                logits = model(batch_x)

                # We assume a classification problem with cross-entropy
                # If your problem is different, change accordingly
                # 'reduction=none' -> per-sample loss
                loss_vals = F.cross_entropy(logits, batch_y, reduction='none')

                # Add the sum of (loss^2) for this batch
                total_loss_sq += torch.sum(loss_vals**2).item()

        # Average of loss^2 over all samples
        avg_loss_sq = total_loss_sq / D_k

        # Utility
        U_k = D_k * math.sqrt(avg_loss_sq)
        return U_k
    
    def compute_pretraining_utility(self, model, local_dataset, device):
        """
        - model: 현재 글로벌 모델 (학습 전 상태)
        - local_dataset: 클라이언트 k의 로컬 데이터로 구성된 DataLoader (또는 리스트)
        - device: "gpu" or "cpu"

        U^{stat}_k = D_{k} * sqrt( (1/D_{k}) * Σ(loss^2(x_d, y_d)) )
        """
        model.eval()
        model.to(device)

        sum_loss_sq = 0.0
        total_samples = 0

        with torch.no_grad():
            for x, y in local_dataset:
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                # 예: CrossEntropy Loss를 샘플별로 계산
                loss_per_sample = F.cross_entropy(out, y, reduction='none')
                sum_loss_sq += (loss_per_sample ** 2).sum().item()
                total_samples += x.size(0)

        if total_samples > 0:
            stat_utility = total_samples * math.sqrt(sum_loss_sq / total_samples)
        else:
            stat_utility = 0.0

        return stat_utility
