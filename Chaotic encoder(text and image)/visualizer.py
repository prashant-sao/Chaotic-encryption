# visualizer.py
# helper plotting functions for embedding matplotlib plots into Tkinter popups.

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import numpy as np

def show_sender_waveforms(parent, t, trajectory, message_bits, encoded_bits, lyap_results=None, title="Sender Waveforms"):
    popup = tk.Toplevel(parent)
    popup.title(title)
    popup.geometry("1100x800")

    rows = 4 if lyap_results is not None else 3
    fig = Figure(figsize=(10.5, 7.5), dpi=110)

    ax0 = fig.add_subplot(rows, 1, 1)
    ax0.plot(t, trajectory[:,0], linewidth=1.2)
    ax0.set_ylabel("x(t)"); ax0.set_title("Chaotic waveform (x-component)"); ax0.grid(True)

    # message waveform mapped to +/-1
    ax1 = fig.add_subplot(rows, 1, 2, sharex=ax0)
    wave_msg = _bits_to_wave(message_bits, len(t))
    ax1.step(t, wave_msg, where='post', linewidth=1.0)
    ax1.set_ylabel("Message"); ax1.set_title("Message waveform"); ax1.set_ylim(-1.6, 1.6); ax1.grid(True)

    ax2 = fig.add_subplot(rows, 1, 3, sharex=ax0)
    wave_enc = _bits_to_wave(encoded_bits, len(t))
    ax2.step(t, wave_enc, where='post', linewidth=1.0)
    ax2.set_ylabel("Encoded"); ax2.set_title("Encoded waveform"); ax2.set_ylim(-1.6,1.6); ax2.grid(True)
    ax2.set_xlabel("Time")

    if lyap_results is not None:
        ax3 = fig.add_subplot(rows, 1, 4, sharex=ax0)
        t_lyap = lyap_results['t']
        ax3.plot(t_lyap, lyap_results['ln_sep'], label='ln(sep)')
        ax3.set_ylabel("ln(sep)")
        ax3b = ax3.twinx()
        ax3b.plot(t_lyap, lyap_results['ftle'], linestyle='--', label='FTLE')
        ax3b.set_ylabel("FTLE")
        ax3.set_title("Lyapunov: ln(sep) and FTLE")
        ax3.grid(True)

    fig.tight_layout(pad=1.5)
    canvas = FigureCanvasTkAgg(fig, master=popup)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def show_receiver_waveforms(parent, encoded_bits, decoded_bits):
    popup = tk.Toplevel(parent)
    popup.title("Receiver — Received & Decoded")
    popup.geometry("900x600")
    fig = Figure(figsize=(8.5, 6), dpi=110)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)

    nbits = len(encoded_bits)
    t = np.linspace(0, nbits, max(1, nbits * 4))
    ax1.step(t, _bits_to_wave(encoded_bits, len(t)), where='post', linewidth=1.2)
    ax1.set_ylabel("Received"); ax1.set_ylim(-1.6,1.6); ax1.grid(True); ax1.set_title("Received encoded waveform")

    if decoded_bits is None:
        decoded_bits = ''
    nbits_d = len(decoded_bits)
    t2 = np.linspace(0, nbits_d, max(1, nbits_d * 4))
    ax2.step(t2, _bits_to_wave(decoded_bits, len(t2)), where='post', linewidth=1.2)
    ax2.set_ylabel("Decoded"); ax2.set_ylim(-1.6,1.6); ax2.grid(True); ax2.set_xlabel("Time")

    fig.tight_layout(pad=1.5)
    canvas = FigureCanvasTkAgg(fig, master=popup)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def show_image(parent, img_arr, title="Image"):
    from PIL import Image, ImageTk
    im = Image.fromarray(img_arr)
    imtk = ImageTk.PhotoImage(im)
    popup = tk.Toplevel(parent)
    popup.title(title)
    lbl = tk.Label(popup, image=imtk)
    lbl.image = imtk
    lbl.pack()

# internal helper
def _bits_to_wave(bits, length, up=1.0, down=-1.0):
    if not bits or length <= 0:
        return np.zeros(length)
    vals = np.array([up if b == '1' else down for b in bits])
    repeat = int(np.ceil(length / len(vals)))
    return np.repeat(vals, repeat)[:length]
