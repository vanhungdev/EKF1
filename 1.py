import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

class BatteryEKF:
    """Extended Kalman Filter cho pin lithium-ion với mô hình ECM 1RC"""
    
    def __init__(self, battery_params):
        self.params = battery_params
        
        # State vector: [SOC, V1]
        self.x = np.array([1.0, 0.0])  # Initial state
        
        # Covariance matrix
        self.P = np.array([[0.01, 0.0], 
                          [0.0, 0.01]])
        
        # Process noise covariance
        self.Q = np.array([[1e-6, 0.0], 
                          [0.0, 1e-4]])
        
        # Measurement noise variance
        self.R = 0.001
        
        # Create interpolation functions for lookup tables
        self._create_interpolators()
        
    def _create_interpolators(self):
        """Tạo các hàm interpolation cho lookup tables"""
        soc_lut = self.params['SOC_LUT']
        
        self.interp_Em = interp1d(soc_lut, self.params['Em_LUT'], 
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
        self.interp_R0 = interp1d(soc_lut, self.params['R0_LUT'], 
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
        self.interp_R1 = interp1d(soc_lut, self.params['R1_LUT'], 
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
        self.interp_C1 = interp1d(soc_lut, self.params['C1_LUT'], 
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
        
    def get_parameters(self, soc):
        """Lấy tham số pin tại SOC hiện tại"""
        soc = np.clip(soc, 0.1, 1.0)  # Clip SOC trong khoảng lookup table
        
        return {
            'Em': float(self.interp_Em(soc)),
            'R0': float(self.interp_R0(soc)),
            'R1': float(self.interp_R1(soc)),
            'C1': float(self.interp_C1(soc))
        }
    
    def predict(self, current, dt):
        """Bước prediction của EKF"""
        soc, v1 = self.x
        
        # State prediction
        soc_pred = soc - current * dt / self.params['capacity_As']
        soc_pred = np.clip(soc_pred, 0.0, 1.0)
        
        # Get parameters for current SOC
        params = self.get_parameters(soc)
        tau = params['R1'] * params['C1']
        
        if tau > 0:
            exp_term = np.exp(-dt / tau)
            v1_pred = v1 * exp_term + params['R1'] * current * (1 - exp_term)
        else:
            v1_pred = v1
            
        self.x = np.array([soc_pred, v1_pred])
        
        # State transition Jacobian
        F = np.array([[1.0, 0.0],
                     [0.0, exp_term if tau > 0 else 1.0]])
        
        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, voltage_measurement, current):
        """Bước update của EKF"""
        soc, v1 = self.x
        
        # Get parameters
        params = self.get_parameters(soc)
        
        # Predicted measurement (terminal voltage)
        h_pred = params['Em'] - current * params['R0'] - v1
        
        # Innovation
        innovation = voltage_measurement - h_pred
        
        # Measurement Jacobian
        # h = Em(SOC) - I*R0(SOC) - V1
        # dh/dSOC = dEm/dSOC - I*dR0/dSOC
        # dh/dV1 = -1
        
        # Numerical derivatives
        delta_soc = 0.001
        soc_plus = np.clip(soc + delta_soc, 0.1, 1.0)
        soc_minus = np.clip(soc - delta_soc, 0.1, 1.0)
        
        em_plus = float(self.interp_Em(soc_plus))
        em_minus = float(self.interp_Em(soc_minus))
        r0_plus = float(self.interp_R0(soc_plus))
        r0_minus = float(self.interp_R0(soc_minus))
        
        dh_dsoc = (em_plus - em_minus) / (soc_plus - soc_minus) - \
                  current * (r0_plus - r0_minus) / (soc_plus - soc_minus)
        
        H = np.array([dh_dsoc, -1.0])
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T / S
        
        # State update
        self.x = self.x + K * innovation
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)  # Constrain SOC
        
        # Covariance update
        I = np.eye(2)
        self.P = (I - np.outer(K, H)) @ self.P
        
    def get_state(self):
        """Lấy trạng thái hiện tại"""
        return {
            'SOC': self.x[0],
            'V1': self.x[1],
            'uncertainty': np.sqrt(self.P[0, 0])
        }

class ECM1RC:
    """Mô hình Equivalent Circuit Model 1RC"""
    
    def __init__(self, battery_params):
        self.params = battery_params
        self.soc = 1.0
        self.v1 = 0.0
        
        # Create interpolation functions
        soc_lut = self.params['SOC_LUT']
        self.interp_Em = interp1d(soc_lut, self.params['Em_LUT'], 
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
        self.interp_R0 = interp1d(soc_lut, self.params['R0_LUT'], 
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
        self.interp_R1 = interp1d(soc_lut, self.params['R1_LUT'], 
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
        self.interp_C1 = interp1d(soc_lut, self.params['C1_LUT'], 
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
    
    def step(self, current, dt):
        """Mô phỏng một bước thời gian"""
        # Update SOC
        self.soc = self.soc - current * dt / self.params['capacity_As']
        self.soc = np.clip(self.soc, 0.0, 1.0)
        
        # Get parameters at current SOC
        soc_clipped = np.clip(self.soc, 0.1, 1.0)
        Em = float(self.interp_Em(soc_clipped))
        R0 = float(self.interp_R0(soc_clipped))
        R1 = float(self.interp_R1(soc_clipped))
        C1 = float(self.interp_C1(soc_clipped))
        
        # Update V1 (voltage across capacitor)
        tau = R1 * C1
        if tau > 0:
            exp_term = np.exp(-dt / tau)
            self.v1 = self.v1 * exp_term + R1 * current * (1 - exp_term)
        
        # Calculate terminal voltage
        terminal_voltage = Em - current * R0 - self.v1
        
        return {
            'SOC': self.soc,
            'terminal_voltage': terminal_voltage,
            'Em': Em,
            'R0': R0,
            'R1': R1,
            'C1': C1,
            'V1': self.v1
        }

def load_battery_data(filename):
    """Đọc dữ liệu pin từ file Excel"""
    try:
        df = pd.read_excel(filename)
        # Downsample để tăng tốc độ xử lý
        df = df.iloc[::10].reset_index(drop=True)
        return df
    except FileNotFoundError:
        print(f"Không tìm thấy file {filename}. Tạo dữ liệu mẫu...")
        return generate_sample_data()

def generate_sample_data():
    """Tạo dữ liệu mẫu nếu không có file"""
    # Parameters
    battery_params = {
        'capacity_As': 27.625 * 3600,
        'SOC_LUT': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'Em_LUT': np.array([3.567061, 3.612307, 3.650412, 3.682024, 3.714915,
                           3.800226, 3.885540, 3.979134, 4.080636, 4.192853]),
        'R0_LUT': np.array([0.008515, 0.008648, 0.008607, 0.008408, 0.008214,
                           0.008256, 0.008298, 0.008386, 0.008519, 0.008605]),
        'R1_LUT': np.array([0.002526, 0.002320, 0.002303, 0.002104, 0.001771,
                           0.001492, 0.001759, 0.002014, 0.001880, 0.001497]),
        'C1_LUT': np.array([15837.142163, 22418.135957, 34743.921371, 38015.301808, 28239.358269,
                           20112.208637, 22733.962312, 29791.306032, 31906.903822, 20040.110951])
    }
    
    # Generate synthetic discharge data
    ecm = ECM1RC(battery_params)
    current = 27.625  # 1C discharge
    dt = 10  # 10 seconds
    total_time = 3600  # 1 hour
    
    data = []
    for t in range(0, total_time + dt, dt):
        result = ecm.step(current, dt)
        data.append({
            'Time': t,
            'SOC': result['SOC'],
            'Terminal_Voltage': result['terminal_voltage'],
            'OCV': result['Em'],
            'V1': result['V1'],
            'R0': result['R0'],
            'R1': result['R1'],
            'C1': result['C1']
        })
    
    return pd.DataFrame(data)

def add_noise(data, voltage_noise=0.01, soc_noise=0.005, current_noise=0.1):
    """Thêm nhiễu vào dữ liệu"""
    np.random.seed(42)  # For reproducible results
    
    noisy_data = data.copy()
    n_points = len(data)
    
    # Add noise
    noisy_data['SOC_Noisy'] = data['SOC'] + np.random.normal(0, soc_noise, n_points)
    noisy_data['Terminal_Voltage_Noisy'] = data['Terminal_Voltage'] + np.random.normal(0, voltage_noise, n_points)
    noisy_data['Current_Noisy'] = 27.625 + np.random.normal(0, current_noise, n_points)
    
    # Clip values to valid ranges
    noisy_data['SOC_Noisy'] = np.clip(noisy_data['SOC_Noisy'], 0, 1)
    
    return noisy_data

def apply_ekf_filter(noisy_data, battery_params):
    """Áp dụng EKF để lọc nhiễu"""
    ekf = BatteryEKF(battery_params)
    
    filtered_results = []
    
    for i, row in noisy_data.iterrows():
        if i == 0:
            dt = 0
        else:
            dt = row['Time'] - noisy_data.iloc[i-1]['Time']
        
        current = row.get('Current_Noisy', 27.625)
        
        if dt > 0:
            # EKF prediction step
            ekf.predict(current, dt)
            
            # EKF update step
            ekf.update(row['Terminal_Voltage_Noisy'], current)
        
        # Get filtered state
        state = ekf.get_state()
        params = ekf.get_parameters(state['SOC'])
        
        filtered_results.append({
            'Time': row['Time'],
            'SOC_EKF': state['SOC'],
            'V1_EKF': state['V1'],
            'Terminal_Voltage_EKF': params['Em'] - current * params['R0'] - state['V1'],
            'Uncertainty': state['uncertainty'],
            'SOC_Error': abs(state['SOC'] - row['SOC']),
            'Voltage_Error': abs(params['Em'] - current * params['R0'] - state['V1'] - row['Terminal_Voltage'])
        })
    
    return pd.DataFrame(filtered_results)

def plot_detailed_zoom(original_data, noisy_data, ekf_results, zoom_start=15, zoom_end=25):
    """Vẽ biểu đồ phóng to chi tiết tại khoảng thời gian cụ thể"""
    
    time_min = original_data['Time'] / 60
    zoom_mask = (time_min >= zoom_start) & (time_min <= zoom_end)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # SOC zoom với markers
    ax1.plot(time_min[zoom_mask], original_data['SOC'][zoom_mask], 'k-', linewidth=4, 
             label='SOC Gốc', marker='o', markersize=6, markerfacecolor='white', markeredgewidth=2)
    ax1.plot(time_min[zoom_mask], noisy_data['SOC_Noisy'][zoom_mask], 'r--', linewidth=3, alpha=0.8, 
             label='SOC Nhiễu', marker='s', markersize=4)
    ax1.plot(time_min[zoom_mask], ekf_results['SOC_EKF'][zoom_mask], 'b-', linewidth=4, 
             label='SOC EKF', marker='^', markersize=5)
    ax1.set_xlabel('Thời gian (phút)', fontsize=12)
    ax1.set_ylabel('SOC', fontsize=12)
    ax1.set_title(f'SOC Chi tiết ({zoom_start}-{zoom_end} phút)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Voltage zoom với markers
    ax2.plot(time_min[zoom_mask], original_data['Terminal_Voltage'][zoom_mask], 'k-', linewidth=4, 
             label='Điện áp Gốc', marker='o', markersize=6, markerfacecolor='white', markeredgewidth=2)
    ax2.plot(time_min[zoom_mask], noisy_data['Terminal_Voltage_Noisy'][zoom_mask], 'r--', linewidth=3, alpha=0.8, 
             label='Điện áp Nhiễu', marker='s', markersize=4)
    ax2.plot(time_min[zoom_mask], ekf_results['Terminal_Voltage_EKF'][zoom_mask], 'b-', linewidth=4, 
             label='Điện áp EKF', marker='^', markersize=5)
    ax2.set_xlabel('Thời gian (phút)', fontsize=12)
    ax2.set_ylabel('Điện áp (V)', fontsize=12)
    ax2.set_title(f'Điện áp Chi tiết ({zoom_start}-{zoom_end} phút)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Sai lệch điện áp so với gốc
    voltage_noise_diff = (noisy_data['Terminal_Voltage_Noisy'][zoom_mask] - original_data['Terminal_Voltage'][zoom_mask]) * 1000
    voltage_ekf_diff = (ekf_results['Terminal_Voltage_EKF'][zoom_mask] - original_data['Terminal_Voltage'][zoom_mask]) * 1000
    
    ax3.plot(time_min[zoom_mask], voltage_noise_diff, 'r-', linewidth=3, alpha=0.8, 
             label='Sai lệch Nhiễu', marker='s', markersize=4)
    ax3.plot(time_min[zoom_mask], voltage_ekf_diff, 'b-', linewidth=4, 
             label='Sai lệch EKF', marker='^', markersize=5)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2)
    ax3.set_xlabel('Thời gian (phút)', fontsize=12)
    ax3.set_ylabel('Sai lệch Điện áp (mV)', fontsize=12)
    ax3.set_title('Sai lệch Điện áp so với Gốc', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Sai lệch SOC so với gốc
    soc_noise_diff = (noisy_data['SOC_Noisy'][zoom_mask] - original_data['SOC'][zoom_mask]) * 100
    soc_ekf_diff = (ekf_results['SOC_EKF'][zoom_mask] - original_data['SOC'][zoom_mask]) * 100
    
    ax4.plot(time_min[zoom_mask], soc_noise_diff, 'r-', linewidth=3, alpha=0.8, 
             label='Sai lệch Nhiễu', marker='s', markersize=4)
    ax4.plot(time_min[zoom_mask], soc_ekf_diff, 'b-', linewidth=4, 
             label='Sai lệch EKF', marker='^', markersize=5)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2)
    ax4.set_xlabel('Thời gian (phút)', fontsize=12)
    ax4.set_ylabel('Sai lệch SOC (%)', fontsize=12)
    ax4.set_title('Sai lệch SOC so với Gốc', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # In thống kê cho khoảng zoom
    print(f"\n{'='*60}")
    print(f"THỐNG KÊ CHI TIẾT CHO KHOẢNG {zoom_start}-{zoom_end} PHÚT")
    print(f"{'='*60}")
    print(f"Sai lệch điện áp nhiễu:  {voltage_noise_diff.std():.2f} ± {voltage_noise_diff.std():.2f} mV")
    print(f"Sai lệch điện áp EKF:    {voltage_ekf_diff.mean():.2f} ± {voltage_ekf_diff.std():.2f} mV")
    print(f"Sai lệch SOC nhiễu:      {soc_noise_diff.mean():.3f} ± {soc_noise_diff.std():.3f} %")
    print(f"Sai lệch SOC EKF:        {soc_ekf_diff.mean():.3f} ± {soc_ekf_diff.std():.3f} %")
    print(f"Cải thiện điện áp:       {voltage_noise_diff.std()/voltage_ekf_diff.std():.1f}x")
    print(f"Cải thiện SOC:           {soc_noise_diff.std()/max(soc_ekf_diff.std(), 1e-6):.1f}x")
    print(f"{'='*60}")

def plot_results(original_data, noisy_data, ekf_results):
    """Vẽ biểu đồ so sánh kết quả"""
    
    # Convert time to minutes
    time_min = original_data['Time'] / 60
    
    # Create subplots - thêm 2 subplot cho zoom
    fig = plt.figure(figsize=(20, 12))
    
    # Tạo layout 2x3
    ax1 = plt.subplot(2, 3, 1)  # SOC full
    ax2 = plt.subplot(2, 3, 2)  # Voltage full  
    ax3 = plt.subplot(2, 3, 3)  # Errors
    ax4 = plt.subplot(2, 3, 4)  # SOC zoom
    ax5 = plt.subplot(2, 3, 5)  # Voltage zoom
    ax6 = plt.subplot(2, 3, 6)  # Noise comparison zoom
    
    # SOC comparison - Full view
    ax1.plot(time_min, original_data['SOC'], 'k-', linewidth=2, label='SOC Gốc')
    ax1.plot(time_min, noisy_data['SOC_Noisy'], 'r--', linewidth=1, alpha=0.7, label='SOC Nhiễu')
    ax1.plot(time_min, ekf_results['SOC_EKF'], 'b-', linewidth=2, label='SOC EKF')
    ax1.set_xlabel('Thời gian (phút)')
    ax1.set_ylabel('SOC')
    ax1.set_title('So sánh SOC - Toàn bộ')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Voltage comparison - Full view
    ax2.plot(time_min, original_data['Terminal_Voltage'], 'k-', linewidth=2, label='Điện áp Gốc')
    ax2.plot(time_min, noisy_data['Terminal_Voltage_Noisy'], 'r--', linewidth=1, alpha=0.7, label='Điện áp Nhiễu')
    ax2.plot(time_min, ekf_results['Terminal_Voltage_EKF'], 'b-', linewidth=2, label='Điện áp EKF')
    ax2.set_xlabel('Thời gian (phút)')
    ax2.set_ylabel('Điện áp (V)')
    ax2.set_title('So sánh Điện áp - Toàn bộ')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Combined errors
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(time_min, ekf_results['SOC_Error'] * 100, 'g-', linewidth=2, label='Sai số SOC (%)')
    line2 = ax3.plot(time_min, ekf_results['Uncertainty'] * 100, 'orange', linewidth=2, label='Độ tin cậy (%)')
    line3 = ax3_twin.plot(time_min, ekf_results['Voltage_Error'] * 1000, 'm-', linewidth=2, label='Sai số V (mV)')
    
    ax3.set_xlabel('Thời gian (phút)')
    ax3.set_ylabel('Sai số SOC (%)', color='g')
    ax3_twin.set_ylabel('Sai số Điện áp (mV)', color='m')
    ax3.set_title('Sai số và Độ tin cậy EKF')
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Chọn khoảng thời gian để zoom (ví dụ: từ phút 15-25)
    zoom_start, zoom_end = 15, 25
    zoom_mask = (time_min >= zoom_start) & (time_min <= zoom_end)
    
    # SOC zoom
    ax4.plot(time_min[zoom_mask], original_data['SOC'][zoom_mask], 'k-', linewidth=3, label='SOC Gốc', marker='o', markersize=3)
    ax4.plot(time_min[zoom_mask], noisy_data['SOC_Noisy'][zoom_mask], 'r--', linewidth=2, alpha=0.8, label='SOC Nhiễu', marker='s', markersize=2)
    ax4.plot(time_min[zoom_mask], ekf_results['SOC_EKF'][zoom_mask], 'b-', linewidth=3, label='SOC EKF', marker='^', markersize=2)
    ax4.set_xlabel('Thời gian (phút)')
    ax4.set_ylabel('SOC')
    ax4.set_title(f'SOC PHÓNG TO ({zoom_start}-{zoom_end} phút)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Voltage zoom
    ax5.plot(time_min[zoom_mask], original_data['Terminal_Voltage'][zoom_mask], 'k-', linewidth=3, label='Điện áp Gốc', marker='o', markersize=3)
    ax5.plot(time_min[zoom_mask], noisy_data['Terminal_Voltage_Noisy'][zoom_mask], 'r--', linewidth=2, alpha=0.8, label='Điện áp Nhiễu', marker='s', markersize=2)
    ax5.plot(time_min[zoom_mask], ekf_results['Terminal_Voltage_EKF'][zoom_mask], 'b-', linewidth=3, label='Điện áp EKF', marker='^', markersize=2)
    ax5.set_xlabel('Thời gian (phút)')
    ax5.set_ylabel('Điện áp (V)')
    ax5.set_title(f'ĐIỆN ÁP PHÓNG TO ({zoom_start}-{zoom_end} phút)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Noise comparison zoom - hiển thị độ lệch so với gốc
    voltage_noise_zoom = noisy_data['Terminal_Voltage_Noisy'][zoom_mask] - original_data['Terminal_Voltage'][zoom_mask]
    voltage_ekf_diff_zoom = ekf_results['Terminal_Voltage_EKF'][zoom_mask] - original_data['Terminal_Voltage'][zoom_mask]
    soc_noise_zoom = (noisy_data['SOC_Noisy'][zoom_mask] - original_data['SOC'][zoom_mask]) * 100
    soc_ekf_diff_zoom = (ekf_results['SOC_EKF'][zoom_mask] - original_data['SOC'][zoom_mask]) * 100
    
    ax6_twin = ax6.twinx()
    line1 = ax6.plot(time_min[zoom_mask], voltage_noise_zoom * 1000, 'r-', linewidth=2, alpha=0.7, label='Nhiễu V (mV)', marker='s', markersize=2)
    line2 = ax6.plot(time_min[zoom_mask], voltage_ekf_diff_zoom * 1000, 'b-', linewidth=3, label='EKF V (mV)', marker='^', markersize=2)
    line3 = ax6_twin.plot(time_min[zoom_mask], soc_noise_zoom, 'orange', linewidth=2, alpha=0.7, label='Nhiễu SOC (%)', marker='o', markersize=2)
    line4 = ax6_twin.plot(time_min[zoom_mask], soc_ekf_diff_zoom, 'g-', linewidth=3, label='EKF SOC (%)', marker='d', markersize=2)
    
    ax6.set_xlabel('Thời gian (phút)')
    ax6.set_ylabel('Sai lệch Điện áp (mV)', color='b')
    ax6_twin.set_ylabel('Sai lệch SOC (%)', color='g')
    ax6.set_title(f'SO SÁNH SAI LỆCH ({zoom_start}-{zoom_end} phút)')
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax6_twin.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Combine legends
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc='upper right')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_statistics(ekf_results):
    """In thống kê kết quả"""
    soc_errors = ekf_results['SOC_Error'] * 100
    voltage_errors = ekf_results['Voltage_Error'] * 1000
    
    print("\n" + "="*50)
    print("THỐNG KÊ KẾT QUẢ EKF")
    print("="*50)
    print(f"Sai số SOC trung bình:     {soc_errors.mean():.3f}%")
    print(f"Sai số SOC tối đa:         {soc_errors.max():.3f}%")
    print(f"Độ lệch chuẩn SOC:         {soc_errors.std():.3f}%")
    print(f"Sai số điện áp trung bình: {voltage_errors.mean():.2f} mV")
    print(f"Sai số điện áp tối đa:     {voltage_errors.max():.2f} mV")
    print(f"Độ lệch chuẩn điện áp:     {voltage_errors.std():.2f} mV")
    print("="*50)

def main():
    """Hàm chính"""
    print("BẮT ĐẦU PHÂN TÍCH EKF CHO PIN LITHIUM-ION")
    print("="*60)
    
    # Battery parameters từ lookup tables
    battery_params = {
        'capacity_Ah': 27.625,
        'capacity_As': 27.625 * 3600,
        'SOC_LUT': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'Em_LUT': np.array([3.567061, 3.612307, 3.650412, 3.682024, 3.714915,
                           3.800226, 3.885540, 3.979134, 4.080636, 4.192853]),
        'R0_LUT': np.array([0.008515, 0.008648, 0.008607, 0.008408, 0.008214,
                           0.008256, 0.008298, 0.008386, 0.008519, 0.008605]),
        'R1_LUT': np.array([0.002526, 0.002320, 0.002303, 0.002104, 0.001771,
                           0.001492, 0.001759, 0.002014, 0.001880, 0.001497]),
        'C1_LUT': np.array([15837.142163, 22418.135957, 34743.921371, 38015.301808, 28239.358269,
                           20112.208637, 22733.962312, 29791.306032, 31906.903822, 20040.110951])
    }
    
    # Bước 1: Đọc dữ liệu gốc
    print("Bước 1: Đọc dữ liệu pin...")
    original_data = load_battery_data('Book3.xlsx')
    print(f"Đã đọc {len(original_data)} điểm dữ liệu")
    print(f"Thời gian: {original_data['Time'].iloc[0]:.1f}s đến {original_data['Time'].iloc[-1]:.1f}s")
    print(f"SOC: {original_data['SOC'].min():.3f} đến {original_data['SOC'].max():.3f}")
    
    # Bước 2: Thêm nhiễu
    print("\nBước 2: Thêm nhiễu vào dữ liệu...")
    noisy_data = add_noise(original_data, 
                          voltage_noise=0.01,  # ±10mV
                          soc_noise=0.005,     # ±0.5%
                          current_noise=0.1)   # ±0.1A
    print("Đã thêm nhiễu: ±10mV (điện áp), ±0.5% (SOC), ±0.1A (dòng)")
    
    # Bước 3: Áp dụng EKF
    print("\nBước 3: Áp dụng EKF để lọc nhiễu...")
    ekf_results = apply_ekf_filter(noisy_data, battery_params)
    print("Đã hoàn thành EKF filtering")
    
    # Bước 4: Hiển thị kết quả
    print("\nBước 4: Hiển thị kết quả...")
    print_statistics(ekf_results)
    
    # Bước 5: Vẽ biểu đồ
    print("\nBước 5: Vẽ biểu đồ so sánh...")
    plot_results(original_data, noisy_data, ekf_results)
    
    # Bước 6: Vẽ biểu đồ phóng to chi tiết
    print("\nBước 6: Vẽ biểu đồ phóng to chi tiết...")
    
    # Cho phép người dùng chọn khoảng phóng to
    total_time_min = original_data['Time'].iloc[-1] / 60
    
    try:
        user_input = input(f"\nNhập khoảng thời gian muốn phóng to (phút) [15-25]: ")
        if user_input.strip():
            if '-' in user_input:
                zoom_start, zoom_end = map(float, user_input.split('-'))
            else:
                zoom_start = float(user_input)
                zoom_end = zoom_start + 10
        else:
            zoom_start, zoom_end = 15, 25
    except:
        zoom_start, zoom_end = 15, 25
        
    # Đảm bảo khoảng zoom hợp lệ
    zoom_start = max(0, min(zoom_start, total_time_min - 5))
    zoom_end = min(total_time_min, max(zoom_end, zoom_start + 5))
    
    print(f"Phóng to khoảng: {zoom_start:.1f} - {zoom_end:.1f} phút")
    plot_detailed_zoom(original_data, noisy_data, ekf_results, zoom_start, zoom_end)
    
    # Lưu kết quả
    print(f"\nBước 7: Lưu kết quả vào file...")
    results_df = pd.DataFrame({
        'Time_min': original_data['Time'] / 60,
        'SOC_Original': original_data['SOC'],
        'SOC_Noisy': noisy_data['SOC_Noisy'],
        'SOC_EKF': ekf_results['SOC_EKF'],
        'Voltage_Original': original_data['Terminal_Voltage'],
        'Voltage_Noisy': noisy_data['Terminal_Voltage_Noisy'],
        'Voltage_EKF': ekf_results['Terminal_Voltage_EKF'],
        'SOC_Error_Percent': ekf_results['SOC_Error'] * 100,
        'Voltage_Error_mV': ekf_results['Voltage_Error'] * 1000,
        'SOC_Noise_Diff_Percent': (noisy_data['SOC_Noisy'] - original_data['SOC']) * 100,
        'Voltage_Noise_Diff_mV': (noisy_data['Terminal_Voltage_Noisy'] - original_data['Terminal_Voltage']) * 1000,
        'SOC_EKF_Diff_Percent': (ekf_results['SOC_EKF'] - original_data['SOC']) * 100,
        'Voltage_EKF_Diff_mV': (ekf_results['Terminal_Voltage_EKF'] - original_data['Terminal_Voltage']) * 1000
    })
    
    results_df.to_excel('battery_ekf_results.xlsx', index=False)
    print("Đã lưu kết quả vào 'battery_ekf_results.xlsx'")
    
    print(f"\n{'='*60}")
    print("✅ HOÀN THÀNH PHÂN TÍCH CHI TIẾT!")
    print("📊 Đã tạo 2 biểu đồ: Tổng quan + Phóng to chi tiết")
    print("💾 Đã lưu kết quả đầy đủ vào file Excel")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()