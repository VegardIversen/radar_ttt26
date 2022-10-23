import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from rftool.radar import Albersheim, Shnidman
from scipy import constants


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


def swerling0(p_fa, peak_power, pulse_width, pulse_repetition_frequency, receiver_noise_figure, receiver_bandwidth, transmitter_loss, receiver_loss, signal_processing_loss):
    return 10*np.log10(peak_power*pulse_width*pulse_repetition_frequency*(10**(receiver_noise_figure/10))/(receiver_bandwidth*(10**(transmitter_loss/10))*(10**(receiver_loss/10))*(10**(signal_processing_loss/10))*p_fa))



#plot the result
def plot_result(p_fa, peak_power, pulse_width, pulse_repetition_frequency, receiver_noise_figure, receiver_bandwidth, transmitter_loss, receiver_loss, signal_processing_loss):
    range = np.arange(0, 10000, 1)
    p_d = swerling0(p_fa, peak_power, pulse_width, pulse_repetition_frequency, receiver_noise_figure, receiver_bandwidth, transmitter_loss, receiver_loss, signal_processing_loss)
    plt.plot(range, p_d)
    plt.show()

def range_det(Pt, Gt,Gr, lam, roc, n_p, SNR, F, B, Ls, T0=290):
    return np.power((Pt*Gt*Gr*np.power(lam,2)*roc*n_p)/(np.power(4*np.pi, 3) * SNR*constants.Boltzmann *T0*F*B*Ls),1/4)

# def SNR_array(Pt, Gt,Gr, lam, roc, n_p, F, B, Ls, T0=290):
#     return (Pt*Gt*Gr*np.power(lam,2)*roc*n_p)/(np.power(4*np.pi, 3)*R*constants.Boltzmann*T0*F*B*Ls)

def Albersheim_arr(p_d_arr, p_fa=1e-6, N=1):
    return Albersheim(p_fa, p_d_arr, N)

def Shnidman_arr(p_d_arr, swir, p_fa=1e-6, N=1):
    return Shnidman(p_fa, p_d_arr, N, swir)

if __name__ == "__main__":
    R = np.arange(10, 10000, 1)
    snr_arr_alb = Albersheim_arr(p_d, p_fa, n_p)
    snr_arr_shn = Shnidman_arr(p_d, 0, p_fa, n_p)

    #plt.plot(lin2db(snr_arr_shn), p_d, label='Shnidman')

    #print(snr_arr_alb)
    #SNR_Alb = Albersheim(p_fa, 0.9, 1)
    #SNR_Shn = Shnidman(p_fa, 0.9, 1, 0)
    range_det_alb_arr = range_det(peak_power, db2lin(antenna_gain), db2lin(antenna_gain), constants.c/Xband_freq, roc, n_p, db2lin(snr_arr_alb), db2lin(receiver_noise_figure),receiver_bandwidth, db2lin(total_loss))
    range_det_shn_arr = range_det(peak_power, db2lin(antenna_gain), db2lin(antenna_gain), constants.c/Xband_freq, roc, n_p, db2lin(snr_arr_shn), db2lin(receiver_noise_figure),receiver_bandwidth, db2lin(total_loss))
    #SNR_arr = SNR_array(peak_power, db2lin(antenna_gain), db2lin(antenna_gain), constants.c/Xband_freq, sigma, 1, db2lin(receiver_noise_figure), receiver_bandwidth, db2lin(total_loss))
    #plt.plot(lin2db(snr_arr_alb), p_d, label='Albersheim')
    plt.plot(range_det_alb_arr, p_d, label='Albersheim')
    plt.plot(range_det_shn_arr, p_d, label='Shnidman')
    plt.xlabel('Range [m]')
    plt.ylabel('p_d')
    plt.legend()
    plt.show()
    #print(np.power(constants.c/Xband_freq, 2))
    #plot_result(p_fa, peak_power, pulse_width, pulse_repetition_frequency, receiver_noise_figure, receiver_bandwidth, transmitter_loss, receiver_loss, signal_processing_loss)