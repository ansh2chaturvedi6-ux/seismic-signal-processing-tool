import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, wiener

# ── load signals saved on Day 1 ──────────────────────────
clean = np.load("data/clean_signal.npy")
noisy = np.load("data/noisy_signal.npy")
t     = np.load("data/time_axis.npy")
fs    = 500  # sampling frequency

# ── SNR helper ───────────────────────────────────────────
def compute_snr(clean, processed):
    signal_power = np.mean(clean ** 2)
    noise_power  = np.mean((processed - clean) ** 2)
    if noise_power == 0:
        return float('inf')
    return round(10 * np.log10(signal_power / noise_power), 2)

# ── Filter 1: Bandpass ────────────────────────────────────
def bandpass_filter(data, lowcut=10, highcut=120, fs=500, order=4):
    nyquist = fs / 2.0
    low     = lowcut  / nyquist
    high    = highcut / nyquist
    b, a    = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# ── Filter 2: Notch ───────────────────────────────────────
def notch_filter(data, notch_freq=50, fs=500, quality_factor=30):
    nyquist = fs / 2.0
    freq    = notch_freq / nyquist
    b, a    = iirnotch(freq, quality_factor)
    return filtfilt(b, a, data)

# ── Filter 3: Wiener ──────────────────────────────────────
def wiener_filter(data, window_size=11):
    return wiener(data, mysize=window_size)

# ── Apply all & compare ───────────────────────────────────
if __name__ == "__main__":
    bp  = bandpass_filter(noisy)
    nf  = notch_filter(noisy)
    wf  = wiener_filter(noisy)

    print("── SNR Results (dB) ──")
    print(f"  Noisy (no filter) : {compute_snr(clean, noisy)} dB")
    print(f"  Bandpass filter   : {compute_snr(clean, bp)} dB")
    print(f"  Notch filter      : {compute_snr(clean, nf)} dB")
    print(f"  Wiener filter     : {compute_snr(clean, wf)} dB")

    # ── Plot all 4 ───────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Filter Comparison on Seismic Signal", fontsize=14)

    axes[0].plot(t, noisy, color="#D85A30", linewidth=0.8)
    axes[0].set_title(f"Noisy input  |  SNR = {compute_snr(clean, noisy)} dB")

    axes[1].plot(t, bp, color="#1D9E75", linewidth=1.2)
    axes[1].plot(t, clean, color="gray", linewidth=0.6, alpha=0.5)
    axes[1].set_title(f"Bandpass (10–120 Hz)  |  SNR = {compute_snr(clean, bp)} dB")

    axes[2].plot(t, nf, color="#534AB7", linewidth=1.2)
    axes[2].plot(t, clean, color="gray", linewidth=0.6, alpha=0.5)
    axes[2].set_title(f"Notch (50 Hz removed)  |  SNR = {compute_snr(clean, nf)} dB")

    axes[3].plot(t, wf, color="#BA7517", linewidth=1.2)
    axes[3].plot(t, clean, color="gray", linewidth=0.6, alpha=0.5)
    axes[3].set_title(f"Wiener filter  |  SNR = {compute_snr(clean, wf)} dB")

    for ax in axes:
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

    axes[3].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("data/filter_comparison.png", dpi=150)
    print("Plot saved to data/filter_comparison.png")
    plt.show()