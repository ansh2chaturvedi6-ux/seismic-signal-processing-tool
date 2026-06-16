import numpy as np
import matplotlib.pyplot as plt
import os

def generate_ricker_wavelet(duration=10, fs=500, f0=25, noise_level=0.4, seed=42):
	np.random.seed(seed)
	t = np.linspace(0, duration, duration * fs)
	tau = t - duration / 2
	ricker = (1 - 2 * (np.pi * f0 * tau) ** 2) * np.exp(-(np.pi * f0 * tau) ** 2)

	signal = np.zeros_like(t)
	reflection_times = [1.5, 3.0, 5.5, 7.2, 8.8]
	amplitudes = [1.0, 0.7, -0.5, 0.4, -0.3]

	for rt, amp in zip(reflection_times, amplitudes):
		idx = int(rt * fs)
		end = min(idx + len(ricker), len(signal))
		signal[idx:end] += amp * ricker[:end - idx]

	noise = noise_level * np.random.randn(len(t))
	noisy = signal + noise
	return t, signal, noisy

def compute_snr(clean, noisy):
	signal_power = np.mean(clean ** 2)
	noise_power = np.mean((noisy - clean) ** 2)
	snr = 10 * np.log10(signal_power / noise_power)
	return round(snr, 2)

if __name__ == "__main__":
	print("Generating seismic signal...")
	t, clean, noisy = generate_ricker_wavelet()
	snr = compute_snr(clean, noisy)

	print(f"Duration     : {t[-1]:.1f} seconds")
	print(f"Samples      : {len(t)}")
	print(f"Starting SNR : {snr} dB")

	plt.figure(figsize=(14, 5))
	plt.subplot(2, 1, 1)
	plt.plot(t, clean, color="green", linewidth=1.5, label="Clean signal")
	plt.title("Clean Seismic Signal")
	plt.ylabel("Amplitude")
	plt.legend()
	plt.grid(True, alpha=0.3)

	plt.subplot(2, 1, 2)
	plt.plot(t, noisy, color="red", linewidth=0.8, label="Noisy signal")
	plt.title("Noisy Seismic Signal")
	plt.xlabel("Time (s)")
	plt.ylabel("Amplitude")
	plt.legend()
	plt.grid(True, alpha=0.3)

	plt.tight_layout()
	os.makedirs("data", exist_ok=True)
	plt.savefig("data/signal_preview.png", dpi=150)
	print("Plot saved to data/signal_preview.png")
	plt.show()

	np.save("data/clean_signal.npy", clean)
	np.save("data/noisy_signal.npy", noisy)
	np.save("data/time_axis.npy", t)
	print("Data saved!")

