import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from rftool.radar import Albersheim, Shnidman
from scipy import constants
import decimal
decimal.getcontext().prec = 100

Xband_freq = 9400e6
horisontal_beam_width = 0.8 #degrees
vertical_beam_width = 15.0 #degrees
antenna_gain = 33 #dB
rotation_rate = 20 #rpm
peak_power = 25e3 #W
pulse_width = 0.15e-6 #s
pulse_repetition_frequency = 4e3 #Hz
receiver_noise_figure = 5 #dB
receiver_bandwidth = 15e6 #Hz
transmitter_loss = 1.5 #dB
receiver_loss = 2.5 #dB
signal_processing_loss = 8.0 #dB
p_fa = 1e-6 #probability of false alarm
n_p = 1
T_d = n_p/pulse_repetition_frequency #dwell time
roc = 10 #ROC in meters
total_loss = transmitter_loss + receiver_loss + signal_processing_loss #dB
duty_cycle = pulse_width*pulse_repetition_frequency
p_d = np.arange(0.1, 0.99, 0.01)

n_p_td = pulse_repetition_frequency* horisontal_beam_width/(6*rotation_rate)  #number of pulses per dwell time

#function to convert from dB to linear
def db2lin(x):
    return 10**(x/10)

#function to convert from linear (W) to dB
def lin2db(x):
    return 10*np.log10(x)
#function to convert from linear (V) to dB
def lin2db_amp(x):
    return 20*np.log10(x)
#caluculate the p_d as a function of range for  with swerling0 10 m^2 target
#and plot the result
#with swerling0




def Shindman_equation(p_d, p_fa, swerling, N):
    C = 0
    alpha = 0

    if N > 40:
        alpha = np.divide(1,4)
    #print(p_fa)

    eta = np.sqrt(-0.8*np.log(4*p_fa*(1-p_fa))) + np.sign(p_d - 0.5)*(np.sqrt(-0.8*np.log(4*p_d*(1-p_d))))
   
    X_inf = eta * (eta + 2*np.sqrt(N/2 + (alpha - 1/4)))
    
    def K(SW):
        if SW == 1:
            return 1
        elif SW == 2:
            return N
        elif SW == 3:
            return 2
        elif SW == 4:
            return 2*N

    if swerling == 0:
        C = 1
    else:
        C1 = (((17.7006*p_d - 18.4496)*p_d + 14.5339)*p_d - 3.525)/K(swerling)
        
        C2 = 1/K(swerling) * (np.exp(27.31*p_d - 25.14) + (p_d-0.8)*(0.7*np.log((1e-5)/p_fa) + (2*N -  20)/80))
        C_db = 0

        if p_d <= 0.872 and p_d >= 0.1:
            C_db = C1
        elif p_d > 0.872 and p_d <= 0.99:
            C_db = C1 + C2
        C = np.power(10, C_db/10)
    X1 = np.divide(C*X_inf, N)
    SNR_dB = 10*np.log10(X1)
    return SNR_dB




def range_det(Pt, Gt,Gr, lam, roc, n_p, SNR, F, B, Ls, T0=290):
    return np.power((Pt*Gt*Gr*np.power(lam,2)*roc*n_p)/(np.power(4*np.pi, 3) * SNR*constants.Boltzmann *T0*F*B*Ls),1/4)

# def SNR_array(Pt, Gt,Gr, lam, roc, n_p, F, B, Ls, T0=290):
#     return (Pt*Gt*Gr*np.power(lam,2)*roc*n_p)/(np.power(4*np.pi, 3)*R*constants.Boltzmann*T0*F*B*Ls)

def Albersheim_arr(p_d_arr, p_fa=1e-6, N=1):
    return Albersheim(p_fa, p_d_arr, N)

def Shnidman_arr(p_d_arr, swir, p_fa=1e-6, N=1): #only works for swerling 0

    return Shnidman(p_fa, p_d_arr, N, swir)

def Shnidman_arr2(p_d_arr, swir, p_fa=1e-6, N=1):
    arr = np.zeros(len(p_d_arr))
    for i in range(len(p_d_arr)):
        arr[i] = Shindman_equation(p_d_arr[i], p_fa, swir, N)
    return arr
if __name__ == "__main__":
    R = np.arange(10, 10000, 1)
    single_pulse_shni = Shnidman_arr2(p_d, 0, p_fa, 1)
    snr_arr_alb_coherent = Albersheim_arr(p_d, p_fa, n_p_td)
    snr_arr_shn_coherent = Shnidman_arr2(p_d, 0, p_fa, n_p_td)
    snr_arr_shn1_coherent = Shnidman_arr2(p_d, 1, p_fa, n_p_td)
    snr_arr_alb_noncoherent = Albersheim_arr(p_d, p_fa, n_p)
    snr_arr_shn_noncoherent = Shnidman_arr2(p_d, 0, p_fa, n_p)
    snr_arr_shn1_noncoherent = Shnidman_arr2(p_d, 1, p_fa, n_p)   
   
    range_det_single_pulse_shnid = range_det(peak_power, db2lin(antenna_gain), db2lin(antenna_gain), constants.c/Xband_freq, roc, 1, db2lin(single_pulse_shni), db2lin(receiver_noise_figure),receiver_bandwidth, db2lin(total_loss))
    range_det_alb_arr_coherent = range_det(peak_power, db2lin(antenna_gain), db2lin(antenna_gain), constants.c/Xband_freq, roc, n_p, db2lin(snr_arr_alb_coherent), db2lin(receiver_noise_figure),receiver_bandwidth, db2lin(total_loss))
    range_det_shn_arr_coherent = range_det(peak_power, db2lin(antenna_gain), db2lin(antenna_gain), constants.c/Xband_freq, roc, n_p, db2lin(snr_arr_shn_coherent), db2lin(receiver_noise_figure),receiver_bandwidth, db2lin(total_loss))
    range_det_shn1_arr_coherent = range_det(peak_power, db2lin(antenna_gain), db2lin(antenna_gain), constants.c/Xband_freq, roc, n_p, db2lin(snr_arr_shn1_coherent), db2lin(receiver_noise_figure),receiver_bandwidth, db2lin(total_loss))
    range_det_alb_arr_noncoherent = range_det(peak_power, db2lin(antenna_gain), db2lin(antenna_gain), constants.c/Xband_freq, roc, n_p_td, db2lin(snr_arr_alb_noncoherent), db2lin(receiver_noise_figure),receiver_bandwidth, db2lin(total_loss))
    range_det_shn_arr_noncoherent = range_det(peak_power, db2lin(antenna_gain), db2lin(antenna_gain), constants.c/Xband_freq, roc, n_p_td, db2lin(snr_arr_shn_noncoherent), db2lin(receiver_noise_figure),receiver_bandwidth, db2lin(total_loss))
    range_det_shn1_arr_noncoherent = range_det(peak_power, db2lin(antenna_gain), db2lin(antenna_gain), constants.c/Xband_freq, roc, n_p_td, db2lin(snr_arr_shn1_noncoherent), db2lin(receiver_noise_figure),receiver_bandwidth, db2lin(total_loss))
    
    #SNR_arr = SNR_array(peak_power, db2lin(antenna_gain), db2lin(antenna_gain), constants.c/Xband_freq, sigma, 1, db2lin(receiver_noise_figure), receiver_bandwidth, db2lin(total_loss))
    plt.plot(range_det_alb_arr_coherent, p_d,  label='Albersheim coherent')
    plt.plot(range_det_single_pulse_shnid, p_d, label='single pulse shnidman')
    plt.plot(range_det_shn_arr_coherent, p_d, label='Shnidman swerling 0 coherent', )
    plt.plot(range_det_shn1_arr_coherent, p_d, label='Shnidman swerling 1 coherent')
    plt.plot(range_det_alb_arr_noncoherent, p_d,linestyle='dashed', label='Albersheim noncoherent')
    plt.plot(range_det_shn_arr_noncoherent, p_d,linestyle='dashed', label='Shnidman swerling 0 noncoherent')
    plt.plot(range_det_shn1_arr_noncoherent, p_d,linestyle='dashed', label='Shnidman swerling 1 noncoherent')
    plt.xlabel('Range [m]')
    plt.ylabel('p_d')
    plt.title(f'Range vs p_d with swerling 0, 1 and Albersheim') 
    plt.legend()
    plt.show()
    #print(np.power(constants.c/Xband_freq, 2))
    #plot_result(p_fa, peak_power, pulse_width, pulse_repetition_frequency, receiver_noise_figure, receiver_bandwidth, transmitter_loss, receiver_loss, signal_processing_loss)