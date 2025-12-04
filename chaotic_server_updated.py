# Modified chaotic cryptography GUI with Lyapunov separation logging & plotting
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import socket
import pickle

# -----------------------
# Chaotic systems + encoder (with Lyapunov computation)
# -----------------------
class ChaoticSystem:
    def __init__(self, initial_conditions, params):
        self.initial_conditions = np.array(initial_conditions, dtype=float)
        self.params = params
        self.trajectory = None

    def simulate(self, t_span, dt=0.01):
        t = np.arange(0, t_span, dt)
        self.trajectory = odeint(self.equations, self.initial_conditions, t)
        return t, self.trajectory

    def equations(self, state, t):
        raise NotImplementedError

    def compute_lyapunov(self, t, trajectory, eps=1e-8, perturb_component=0):
        """
        Compute separation between trajectory and a nearby perturbed trajectory,
        then compute finite-time Lyapunov exponent FTLE(t) = ln(sep(t)/eps) / t.
        Returns:
            sep: array of Euclidean separations |x'(t)-x(t)|
            ln_sep: natural log of sep (with small floor)
            ftle: finite-time Lyapunov exponent array (with t>0)
        """
        # prepare perturbed initial condition
        pert_ic = self.initial_conditions.copy()
        pert_ic[perturb_component] += eps

        # simulate perturbed trajectory using the same time vector (use odeint)
        pert_traj = odeint(self.equations, pert_ic, t)

        # separation (Euclidean norm)
        diff = pert_traj - trajectory
        sep = np.linalg.norm(diff, axis=1)

        # avoid zeros in log calculation
        tiny = 1e-30
        safe_sep = np.maximum(sep, tiny)

        # natural log of separation
        ln_sep = np.log(safe_sep)

        # FTLE: ln(sep/eps) / t  (for t==0 we set FTLE to same as first non-zero or 0)
        ftle = np.zeros_like(ln_sep)
        # compute only for t>0 to avoid div by zero - simple forward fill for t==0
        positive_t = t > 0
        ftle[positive_t] = (np.log(safe_sep[positive_t] / eps)) / t[positive_t]
        # handle first sample (t==0) - set to ftle at next positive index or 0
        if not np.any(positive_t):
            ftle[:] = 0.0
        else:
            first_pos_idx = np.argmax(positive_t)
            ftle[0] = ftle[first_pos_idx]

        return sep, ln_sep, ftle


class ChuaSystem(ChaoticSystem):
    def __init__(self, initial_conditions=[0.7, 0.0, 0.0],
                 params={'alpha': 15.6, 'beta': 28.0, 'a': -1.143, 'b': -0.714}):
        super().__init__(initial_conditions, params)

    def equations(self, state, t):
        x, y, z = state
        alpha = self.params['alpha']; beta = self.params['beta']; a = self.params['a']; b = self.params['b']
        h = b * x + 0.5 * (a - b) * (abs(x + 1) - abs(x - 1))
        dx = alpha * (y - x - h)
        dy = x - y + z
        dz = -beta * y
        return [dx, dy, dz]


class LorenzSystem(ChaoticSystem):
    def __init__(self, initial_conditions=[1.0, 1.0, 1.0],
                 params={'sigma': 10.0, 'rho': 28.0, 'beta': 8.0/3.0}):
        super().__init__(initial_conditions, params)

    def equations(self, state, t):
        x, y, z = state
        sigma = self.params['sigma']; rho = self.params['rho']; beta = self.params['beta']
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]


class RosslerSystem(ChaoticSystem):
    def __init__(self, initial_conditions=[1.0, 1.0, 1.0],
                 params={'a': 0.2, 'b': 0.2, 'c': 5.7}):
        super().__init__(initial_conditions, params)

    def equations(self, state, t):
        x, y, z = state
        a = self.params['a']; b = self.params['b']; c = self.params['c']
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        return [dx, dy, dz]


class ChaoticEncoder:
    def __init__(self, system):
        self.system = system

    def text_to_bits(self, text):
        return ''.join(format(ord(c), '08b') for c in text)

    def bits_to_text(self, bits):
        chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
        return ''.join(chr(int(c, 2)) for c in chars if len(c) == 8)

    def generate_keystream(self, length, component=0):
        if self.system.trajectory is None:
            raise ValueError("System must be simulated first")
        x_vals = self.system.trajectory[:, 0]
        y_vals = self.system.trajectory[:, 1]
        z_vals = self.system.trajectory[:, 2]
        keystream = []
        L = len(x_vals)
        for i in range(length):
            idx = (i * 13) % L
            x = x_vals[idx]; y = y_vals[idx]; z = z_vals[idx]
            mixed = (x * 1.23456789 + y * 9.87654321 + z * 3.14159265)
            fractional = abs(mixed) - int(abs(mixed))
            decimal_str = f"{fractional:.15f}"
            digit_sum = sum(int(d) for d in decimal_str if d.isdigit())
            bit = '1' if digit_sum % 2 == 1 else '0'
            keystream.append(bit)
        return ''.join(keystream)

    def encode(self, message, t_span=500, dt=0.005, component=0, lyap_eps=1e-8, lyap_component=0):
        """
        Returns:
            encoded_bits, message_bits, keystream, t, trajectory, lyap_results
        where lyap_results = dict(sep, ln_sep, ftle, eps, component)
        """
        message_bits = self.text_to_bits(message)
        t, trajectory = self.system.simulate(t_span, dt)
        # compute lyapunov separation and FTLE
        sep, ln_sep, ftle = self.system.compute_lyapunov(t, trajectory, eps=lyap_eps, perturb_component=lyap_component)

        keystream = self.generate_keystream(len(message_bits), component)
        encoded_bits = ''.join(str(int(m) ^ int(k)) for m, k in zip(message_bits, keystream))

        lyap_results = {
            'sep': sep,
            'ln_sep': ln_sep,
            'ftle': ftle,
            'eps': lyap_eps,
            'perturb_component': lyap_component,
            't': t
        }
        return encoded_bits, message_bits, keystream, t, trajectory, lyap_results

    def decode(self, encoded_bits, keystream):
        decoded_bits = ''.join(str(int(e) ^ int(k)) for e, k in zip(encoded_bits, keystream))
        decoded_message = self.bits_to_text(decoded_bits)
        return decoded_message, decoded_bits


def create_system(system_name, initial_conditions):
    if system_name == "Lorenz":
        return LorenzSystem(initial_conditions)
    elif system_name == "Chua":
        return ChuaSystem(initial_conditions)
    elif system_name == "Rössler":
        return RosslerSystem(initial_conditions)
    else:
        raise ValueError(f"Unknown system: {system_name}")


# -----------------------
# GUI
# -----------------------
class SenderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sender - Chaotic Message Encoder")
        self.root.geometry("1000x700")

        self.server_socket = None
        self.client_socket = None
        self.is_server_running = False

        # last artifacts for visualization
        self.last_t = None
        self.last_trajectory = None
        self.last_message_bits = None
        self.last_encoded_bits = None
        self.last_keystream = None
        self.last_lyap = None

        self.setup_ui()

    def setup_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        self.root.columnconfigure(0, weight=1); self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)

        ttk.Label(main, text="🔐 SENDER - Chaotic Message Encoder", font=('Arial', 15, 'bold')).grid(row=0, column=0, pady=(6,8))

        # top frames: connection and system
        top_frame = ttk.Frame(main); top_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        conn = ttk.LabelFrame(top_frame, text="Connection", padding=6); conn.grid(row=0, column=0, sticky=tk.W, padx=4)
        ttk.Label(conn, text="Port:").grid(row=0, column=0, padx=4)
        self.port_entry = ttk.Entry(conn, width=8); self.port_entry.insert(0, "5555"); self.port_entry.grid(row=0, column=1, padx=4)
        self.start_server_btn = ttk.Button(conn, text="🟢 Start Server", command=self.start_server); self.start_server_btn.grid(row=0, column=2, padx=4)
        self.stop_server_btn = ttk.Button(conn, text="🔴 Stop Server", command=self.stop_server, state=tk.DISABLED); self.stop_server_btn.grid(row=0, column=3, padx=4)

        sysf = ttk.LabelFrame(top_frame, text="Chaotic System", padding=6); sysf.grid(row=0, column=1, sticky=tk.E, padx=8)
        ttk.Label(sysf, text="System:").grid(row=0, column=0, padx=4)
        self.system_var = tk.StringVar(value="Lorenz")
        ttk.Combobox(sysf, textvariable=self.system_var, values=["Lorenz","Chua","Rössler"], state="readonly", width=10).grid(row=0, column=1, padx=4)
        ttk.Label(sysf, text="Init X:").grid(row=0, column=2, padx=4); self.init_x = ttk.Entry(sysf, width=8); self.init_x.insert(0,"1.0"); self.init_x.grid(row=0, column=3)
        ttk.Label(sysf, text="Init Y:").grid(row=0, column=4, padx=4); self.init_y = ttk.Entry(sysf, width=8); self.init_y.insert(0,"1.0"); self.init_y.grid(row=0, column=5)
        ttk.Label(sysf, text="Init Z:").grid(row=0, column=6, padx=4); self.init_z = ttk.Entry(sysf, width=8); self.init_z.insert(0,"1.0"); self.init_z.grid(row=0, column=7)

        # Message
        msg_frame = ttk.LabelFrame(main, text="Message to Send", padding=6); msg_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(8,6))
        self.message_input = scrolledtext.ScrolledText(msg_frame, height=4, wrap=tk.WORD); self.message_input.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.message_input.insert(1.0, "Secret message from chaos!")

        # Lyapunov options
        lyap_frame = ttk.Frame(main); lyap_frame.grid(row=3, column=0, sticky=tk.W, pady=(6,0))
        ttk.Label(lyap_frame, text="Lyap eps:").grid(row=0, column=0, padx=4)
        self.lyap_eps_entry = ttk.Entry(lyap_frame, width=8); self.lyap_eps_entry.insert(0, "1e-8"); self.lyap_eps_entry.grid(row=0, column=1, padx=4)
        ttk.Label(lyap_frame, text="Perturb comp:").grid(row=0, column=2, padx=4)
        self.lyap_comp_entry = ttk.Entry(lyap_frame, width=4); self.lyap_comp_entry.insert(0, "0"); self.lyap_comp_entry.grid(row=0, column=3, padx=4)

        # buttons (send + visualize)
        btn_frame = ttk.Frame(main); btn_frame.grid(row=4, column=0, sticky=tk.W, pady=(6,8))
        self.send_btn = ttk.Button(btn_frame, text="📤 Encode & Send Message", command=self.send_message, state=tk.DISABLED)
        self.send_btn.grid(row=0, column=0, padx=6)
        self.visualize_btn = ttk.Button(btn_frame, text="🔍 Visualize Waveforms", command=self.open_visualize_sender)
        self.visualize_btn.grid(row=0, column=1, padx=6)

        # Log (reduced)
        log_frame = ttk.LabelFrame(main, text="Activity Log", padding=6); log_frame.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.S), pady=(6,4))
        log_frame.columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.status_var = tk.StringVar(value="Ready - Start server to begin")
        ttk.Label(main, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(6,4))

    def log(self, msg):
        self.log_text.insert(tk.END, f"{msg}\n"); self.log_text.see(tk.END)

    def start_server(self):
        try:
            port = int(self.port_entry.get())
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', port))
            self.server_socket.listen(1)
            self.is_server_running = True
            self.start_server_btn.config(state=tk.DISABLED); self.stop_server_btn.config(state=tk.NORMAL)
            self.log(f"✓ Server started on port {port}")
            self.status_var.set(f"Server running on port {port}")
            threading.Thread(target=self.accept_connections, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start server: {e}"); self.log(f"✗ Error starting server: {e}")

    def accept_connections(self):
        try:
            self.client_socket, addr = self.server_socket.accept()
            self.log(f"✓ Receiver connected from {addr}")
            self.status_var.set(f"Connected to receiver at {addr}")
            self.send_btn.config(state=tk.NORMAL)
        except Exception as e:
            if self.is_server_running:
                self.log(f"✗ Connection error: {e}")

    def stop_server(self):
        self.is_server_running = False
        if self.client_socket: self.client_socket.close(); self.client_socket = None
        if self.server_socket: self.server_socket.close(); self.server_socket = None
        self.start_server_btn.config(state=tk.NORMAL); self.stop_server_btn.config(state=tk.DISABLED); self.send_btn.config(state=tk.DISABLED)
        self.log("✓ Server stopped"); self.status_var.set("Server stopped")

    def _bits_to_wave(self, bits, length, up=1.0, down=-1.0):
        if len(bits) == 0 or length <= 0:
            return np.zeros(length)
        vals = np.array([up if b == '1' else down for b in bits])
        repeat = int(np.ceil(length / len(vals)))
        return np.repeat(vals, repeat)[:length]

    def plot_waveforms_in_popup(self, t, trajectory, message_bits, encoded_bits, lyap_results=None, title="Sender Waveforms"):
        # create popup
        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.geometry("1200x800")
        # build figure with 4 rows if lyap_results provided
        rows = 4 if lyap_results is not None else 3
        fig = Figure(figsize=(11.0, 8.0), dpi=110)
        axes = [fig.add_subplot(rows, 1, i+1) for i in range(rows)]

        # chaos x(t)
        axes[0].plot(t, trajectory[:,0], linewidth=1.2)
        axes[0].set_ylabel("x(t)"); axes[0].set_title("Chaotic waveform (x-component)"); axes[0].grid(True)

        # message waveform
        wave_msg = self._bits_to_wave(message_bits, len(t))
        axes[1].step(t, wave_msg, where='post', linewidth=1.0)
        axes[1].set_ylabel("Message\n(1:+1 / 0:-1)"); axes[1].set_title("Original message waveform (bits mapped)"); axes[1].set_ylim(-1.6, 1.6); axes[1].grid(True)

        # encoded waveform
        wave_enc = self._bits_to_wave(encoded_bits, len(t))
        axes[2].step(t, wave_enc, where='post', linewidth=1.0)
        axes[2].set_ylabel("Encoded\n(1:+1 / 0:-1)"); axes[2].set_title("Encoded message waveform (after XOR)"); axes[2].set_ylim(-1.6, 1.6)
        axes[2].set_xlabel("Time"); axes[2].grid(True)

        # lyapunov plots (ln separation + FTLE)
        if lyap_results is not None:
            ln_sep = lyap_results['ln_sep']; ftle = lyap_results['ftle']; t_lyap = lyap_results['t']
            ax4 = axes[3]
            ax4.plot(t_lyap, ln_sep, linewidth=1.0, label="ln(sep)")
            # FTLE plotted on secondary axis (same subplot)
            ax4b = ax4.twinx()
            ax4b.plot(t_lyap, ftle, linewidth=1.0, linestyle='--', label="FTLE")
            ax4.set_ylabel("ln(sep)"); ax4b.set_ylabel("FTLE (1/t)")
            ax4.set_title("Lyapunov: ln(separation) and finite-time Lyapunov exponent (FTLE)")
            ax4.grid(True)

        fig.tight_layout(pad=2.0)
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def open_visualize_sender(self):
        """Open visualization popup for sender. Will simulate if no last artifacts exist."""
        try:
            # If we've encoded previously, use those artifacts
            if self.last_t is not None and self.last_trajectory is not None and self.last_message_bits is not None and self.last_encoded_bits is not None:
                t = self.last_t; trajectory = self.last_trajectory; msg_bits = self.last_message_bits; enc_bits = self.last_encoded_bits; lyap = self.last_lyap
            else:
                # simulate from current GUI values without sending
                message = self.message_input.get(1.0, tk.END).strip()
                if not message:
                    messagebox.showwarning("Warning", "Please enter a message to visualize!"); return
                x = float(self.init_x.get()); y = float(self.init_y.get()); z = float(self.init_z.get())
                lyap_eps = float(self.lyap_eps_entry.get())
                lyap_comp = int(self.lyap_comp_entry.get())
                system = create_system(self.system_var.get(), [x, y, z])
                encoder = ChaoticEncoder(system)
                enc_bits, msg_bits, keystream, t, trajectory, lyap = encoder.encode(message, lyap_eps=lyap_eps, lyap_component=lyap_comp)
                # store for re-open
                self.last_t = t; self.last_trajectory = trajectory; self.last_message_bits = msg_bits; self.last_encoded_bits = enc_bits; self.last_keystream = keystream; self.last_lyap = lyap

            # open popup with plots (include lyapunov if available)
            self.plot_waveforms_in_popup(t, trajectory, msg_bits, enc_bits, lyap_results=self.last_lyap, title="Sender — Chaos / Message / Encoded / Lyapunov")
        except Exception as e:
            messagebox.showerror("Visualization Error", f"Could not visualize: {e}")
            self.log(f"✗ Visualization error: {e}")

    def _log_lyapunov_summary(self, lyap_results):
        """
        Log a compact Lyapunov summary into the activity log:
        - eps (initial perturbation)
        - final separation
        - max FTLE, mean FTLE (over reasonable window)
        - small sample of ln(sep) values
        """
        if lyap_results is None:
            return
        sep = lyap_results['sep']; ln_sep = lyap_results['ln_sep']; ftle = lyap_results['ftle']; eps = lyap_results['eps']
        t = lyap_results['t']
        # summary stats
        final_sep = sep[-1]
        max_ftle = np.max(ftle)
        mean_ftle = np.mean(ftle[int(len(ftle)*0.1):]) if len(ftle) > 10 else np.mean(ftle)
        # sample ln(sep) first/last few
        sample_ln_front = ','.join([f"{v:.3f}" for v in ln_sep[1:6]]) if len(ln_sep) > 6 else ','.join([f"{v:.3f}" for v in ln_sep])
        sample_ln_back = ','.join([f"{v:.3f}" for v in ln_sep[-5:]]) if len(ln_sep) >= 5 else ''
        self.log(f"--- Lyapunov summary ---")
        self.log(f"eps (initial perturbation): {eps:e}")
        self.log(f"perturb component index: {lyap_results.get('perturb_component', 0)}")
        self.log(f"final separation |Δx(T)|: {final_sep:.6e}")
        self.log(f"max FTLE: {max_ftle:.6f}; mean FTLE (after 10% burn): {mean_ftle:.6f}")
        self.log(f"sample ln(sep) start: [{sample_ln_front}] ... end: [{sample_ln_back}]")
        self.log(f"(Interpretation: positive FTLE => exponential divergence => high sensitivity)")

    def send_message(self):
        if not self.client_socket:
            messagebox.showwarning("Warning", "No receiver connected!"); return
        message = self.message_input.get(1.0, tk.END).strip()
        if not message:
            messagebox.showwarning("Warning", "Please enter a message!"); return
        try:
            x = float(self.init_x.get()); y = float(self.init_y.get()); z = float(self.init_z.get())
            lyap_eps = float(self.lyap_eps_entry.get())
            lyap_comp = int(self.lyap_comp_entry.get())
            system = create_system(self.system_var.get(), [x, y, z])
            encoder = ChaoticEncoder(system)
            encoded_bits, message_bits, keystream, t, trajectory, lyap_results = encoder.encode(message, lyap_eps=lyap_eps, lyap_component=lyap_comp)
            # store artifacts (useful for later visualize)
            self.last_t = t; self.last_trajectory = trajectory; self.last_message_bits = message_bits; self.last_encoded_bits = encoded_bits; self.last_keystream = keystream; self.last_lyap = lyap_results

            # prepare and send packet (include small sample for debug)
            pkt = {'encoded_message': encoded_bits, 'system_type': self.system_var.get(), 'message_length': len(message), 'keystream_sample': keystream[:160]}
            data = pickle.dumps(pkt)
            self.client_socket.sendall(len(data).to_bytes(4, 'big')); self.client_socket.sendall(data)

            self.log(f"✓ Sent encoded message ({len(encoded_bits)} bits)")
            self.log(f"  System: {self.system_var.get()}")
            self.log(f"  Initial conds: [{x}, {y}, {z}]")
            self.log(f"  Original message: '{message}'")
            # Lyapunov summary logging
            self._log_lyapunov_summary(lyap_results)

            self.status_var.set("Message sent successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send message: {e}"); self.log(f"✗ Error: {e}")


# -----------------------
# Receiver GUI (unchanged except small cosmetic)
# -----------------------
class ReceiverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Receiver - Chaotic Message Decoder")
        self.root.geometry("900x620")
        self.client_socket = None
        self.received_data = None
        self.last_received_encoded = None
        self.last_decoded_bits = None
        self.setup_ui()

    def setup_ui(self):
        main = ttk.Frame(self.root, padding=8); main.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        self.root.columnconfigure(0, weight=1); self.root.rowconfigure(0, weight=1); main.columnconfigure(0, weight=1)

        ttk.Label(main, text="🔓 RECEIVER - Chaotic Message Decoder", font=('Arial', 15, 'bold')).grid(row=0, column=0, pady=(6,8))

        conn = ttk.LabelFrame(main, text="Connection", padding=6); conn.grid(row=1, column=0, sticky=(tk.W, tk.E))
        ttk.Label(conn, text="Host:").grid(row=0, column=0, padx=4); self.host_entry = ttk.Entry(conn, width=14); self.host_entry.insert(0,"localhost"); self.host_entry.grid(row=0, column=1)
        ttk.Label(conn, text="Port:").grid(row=0, column=2, padx=4); self.port_entry = ttk.Entry(conn, width=8); self.port_entry.insert(0,"5555"); self.port_entry.grid(row=0, column=3)
        self.connect_btn = ttk.Button(conn, text="🔌 Connect to Sender", command=self.connect_to_sender); self.connect_btn.grid(row=0, column=4, padx=6)
        self.disconnect_btn = ttk.Button(conn, text="❌ Disconnect", command=self.disconnect, state=tk.DISABLED); self.disconnect_btn.grid(row=0, column=5, padx=4)

        keyf = ttk.LabelFrame(main, text="Decryption Key (Initial Conditions)", padding=6); keyf.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(6,4))
        ttk.Label(keyf, text="System:").grid(row=0, column=0, padx=4); self.system_var = tk.StringVar(value="Lorenz")
        ttk.Combobox(keyf, textvariable=self.system_var, values=["Lorenz","Chua","Rössler"], state="readonly", width=10).grid(row=0, column=1, padx=4)
        ttk.Label(keyf, text="Init X:").grid(row=0, column=2, padx=4); self.init_x = ttk.Entry(keyf, width=8); self.init_x.insert(0,"1.0"); self.init_x.grid(row=0, column=3)
        ttk.Label(keyf, text="Init Y:").grid(row=0, column=4, padx=4); self.init_y = ttk.Entry(keyf, width=8); self.init_y.insert(0,"1.0"); self.init_y.grid(row=0, column=5)
        ttk.Label(keyf, text="Init Z:").grid(row=0, column=6, padx=4); self.init_z = ttk.Entry(keyf, width=8); self.init_z.insert(0,"1.0"); self.init_z.grid(row=0, column=7)

        act = ttk.Frame(main); act.grid(row=3, column=0, pady=(8,6))
        self.receive_btn = ttk.Button(act, text="📥 Receive Message", command=self.receive_message, state=tk.DISABLED); self.receive_btn.grid(row=0, column=0, padx=6)
        self.decode_btn = ttk.Button(act, text="🔓 Decode Message", command=self.decode_message, state=tk.DISABLED); self.decode_btn.grid(row=0, column=1, padx=6)
        self.visualize_btn = ttk.Button(act, text="🔍 Visualize Waveforms", command=self.open_visualize_receiver); self.visualize_btn.grid(row=0, column=2, padx=6)

        res_frame = ttk.LabelFrame(main, text="Decoded Message", padding=6); res_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(6,4))
        self.results_text = scrolledtext.ScrolledText(res_frame, height=5, wrap=tk.WORD, font=('Arial', 12, 'bold')); self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

        log_frame = ttk.LabelFrame(main, text="Activity Log", padding=6); log_frame.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.S), pady=(6,4))
        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, wrap=tk.WORD); self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

        self.status_var = tk.StringVar(value="Ready - Connect to sender")
        ttk.Label(main, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(6,4))

    def log(self, msg):
        self.log_text.insert(tk.END, f"{msg}\n"); self.log_text.see(tk.END)

    def connect_to_sender(self):
        try:
            host = self.host_entry.get(); port = int(self.port_entry.get())
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((host, port))
            self.log(f"✓ Connected to sender at {host}:{port}"); self.status_var.set(f"Connected to {host}:{port}")
            self.connect_btn.config(state=tk.DISABLED); self.disconnect_btn.config(state=tk.NORMAL); self.receive_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Connection failed: {e}"); self.log(f"✗ Connection error: {e}")

    def disconnect(self):
        if self.client_socket: self.client_socket.close(); self.client_socket = None
        self.connect_btn.config(state=tk.NORMAL); self.disconnect_btn.config(state=tk.DISABLED); self.receive_btn.config(state=tk.DISABLED); self.decode_btn.config(state=tk.DISABLED)
        self.log("✓ Disconnected from sender"); self.status_var.set("Disconnected")

    def _bits_to_wave(self, bits, length, up=1.0, down=-1.0):
        if len(bits) == 0 or length <= 0:
            return np.zeros(length)
        vals = np.array([up if b == '1' else down for b in bits])
        repeat = int(np.ceil(length / len(vals)))
        return np.repeat(vals, repeat)[:length]

    def plot_receiver_in_popup(self, encoded_bits, decoded_bits, title="Receiver Waveforms"):
        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.geometry("1000x600")
        fig = Figure(figsize=(10, 6), dpi=110)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        # received encoded waveform
        nbits = len(encoded_bits)
        t = np.linspace(0, nbits, max(1, nbits * 4))
        wave_enc = self._bits_to_wave(encoded_bits, len(t))
        ax1.step(t, wave_enc, where='post', linewidth=1.2)
        ax1.set_ylabel("Received\n(1:+1 / 0:-1)"); ax1.set_title("Received encoded waveform"); ax1.set_ylim(-1.6,1.6); ax1.grid(True)

        # decoded waveform
        nbits_d = len(decoded_bits) if decoded_bits is not None else 0
        t2 = np.linspace(0, nbits_d, max(1, nbits_d * 4))
        wave_dec = self._bits_to_wave(decoded_bits if decoded_bits is not None else '', len(t2))
        ax2.step(t2, wave_dec, where='post', linewidth=1.2)
        ax2.set_ylabel("Decoded\n(1:+1 / 0:-1)"); ax2.set_title("Decoded/recovered waveform"); ax2.set_ylim(-1.6,1.6); ax2.grid(True)
        ax2.set_xlabel("Time")

        fig.tight_layout(pad=2.0)
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def open_visualize_receiver(self):
        """Open visualization popup for receiver. Requires a received encoded message."""
        if not self.received_data:
            messagebox.showwarning("No data", "No encoded message has been received yet to visualize.")
            return
        enc = self.received_data['encoded_message']
        # If we've decoded already and have decoded bits, use them; otherwise attempt to generate keystream and decode now
        if self.last_decoded_bits is not None:
            decoded_bits = self.last_decoded_bits
        else:
            # try to decode with current key fields (non-invasive)
            try:
                x = float(self.init_x.get()); y = float(self.init_y.get()); z = float(self.init_z.get())
                system = create_system(self.system_var.get(), [x, y, z])
                encoder = ChaoticEncoder(system)
                encoder.system.simulate(500, 0.005)
                keystream = encoder.generate_keystream(len(enc), component=0)
                # decode to bits (but don't update decoded text UI automatically)
                _, decoded_bits = encoder.decode(enc, keystream)
            except Exception as e:
                messagebox.showerror("Visualization Error", f"Could not generate decode for visualization: {e}")
                self.log(f"✗ Visualization decode error: {e}")
                return
        # open popup with encoded and decoded waves
        self.plot_receiver_in_popup(enc, decoded_bits, title="Receiver — Received & Decoded")

    def receive_message(self):
        if not self.client_socket:
            messagebox.showwarning("Warning", "Not connected to sender!"); return
        try:
            self.log("Waiting to receive message..."); self.status_var.set("Receiving...")
            length_bytes = self.client_socket.recv(4)
            if not length_bytes: raise Exception("Connection closed")
            data_length = int.from_bytes(length_bytes, 'big')
            data = b''
            while len(data) < data_length:
                chunk = self.client_socket.recv(min(4096, data_length - len(data)))
                if not chunk: raise Exception("Connection closed")
                data += chunk
            self.received_data = pickle.loads(data)
            enc = self.received_data['encoded_message']
            self.last_received_encoded = enc
            self.log(f"✓ Received encoded message ({len(enc)} bits)")
            self.log(f"  System type: {self.received_data['system_type']}")
            self.log(f"  Message length: {self.received_data['message_length']} characters")
            self.log(f"  Encoded data (first 80 bits): {enc[:80]}...")
            self.log(f"\n⚠️  To decode: Select '{self.received_data['system_type']}' system and enter correct initial conditions!")
            self.system_var.set(self.received_data['system_type'])
            self.decode_btn.config(state=tk.NORMAL)
            self.status_var.set("Message received - Ready to decode")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to receive: {e}"); self.log(f"✗ Error: {e}")

    def decode_message(self):
        if not self.received_data:
            messagebox.showwarning("Warning", "No message received yet!"); return
        try:
            x = float(self.init_x.get()); y = float(self.init_y.get()); z = float(self.init_z.get())
            system_type = self.system_var.get()
            self.log("Decoding message..."); self.log(f"  Using system: {system_type}"); self.log(f"  Using initial conditions: [{x}, {y}, {z}]")
            system = create_system(system_type, [x, y, z])
            encoder = ChaoticEncoder(system)
            encoder.system.simulate(500, 0.005)
            keystream = encoder.generate_keystream(len(self.received_data['encoded_message']), component=0)
            sender_keystream = self.received_data.get('keystream_sample', 'N/A')
            receiver_sample = keystream[:len(sender_keystream)] if sender_keystream != 'N/A' else keystream[:80]
            if sender_keystream != 'N/A' and len(sender_keystream) > 0:
                matches = sum(s == r for s, r in zip(sender_keystream, receiver_sample))
                match_percent = (matches / len(sender_keystream)) * 100
                self.log(f"  Keystream match (sample {len(sender_keystream)} bits): {matches}/{len(sender_keystream)} bits ({match_percent:.1f}%)")
                if match_percent < 95:
                    self.log("  ⚠️ Keystreams don't match: check initial conditions/system params.")
            decoded_message, decoded_bits = encoder.decode(self.received_data['encoded_message'], keystream)
            self.last_decoded_bits = decoded_bits
            self.results_text.delete(1.0, tk.END); self.results_text.insert(1.0, decoded_message)
            self.log(f"✓ Message decoded!"); self.log(f"  Decoded text: '{decoded_message}'")
            printable_ratio = sum(c.isprintable() for c in decoded_message) / len(decoded_message) if decoded_message else 0
            if printable_ratio < 0.7:
                self.log("  ⚠️ Decoded message has many non-printable characters -> likely wrong key/params.")
            else:
                self.log(f"  Message appears valid (printable: {printable_ratio*100:.1f}%)")
            self.status_var.set("Message decoded!")
        except Exception as e:
            messagebox.showerror("Error", f"Decoding failed: {e}"); self.log(f"✗ Decoding error: {e}")



# -----------------------
# Launcher
# -----------------------
def main():
    def launch_sender():
        r = tk.Tk(); SenderGUI(r); r.mainloop()
    def launch_receiver():
        r = tk.Tk(); ReceiverGUI(r); r.mainloop()

    selector = tk.Tk(); selector.title("Chaotic Cryptography - Select Mode"); selector.geometry("420x260")
    frm = ttk.Frame(selector, padding=18); frm.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
    ttk.Label(frm, text="Chaotic Cryptography System", font=('Arial', 16, 'bold')).grid(row=0, column=0, pady=(8,18))
    
    # Note: Added sticky=(tk.E, tk.W) so buttons stretch if you resize, 
    # but they will stay centered because of the column configuration.
    ttk.Button(frm, text="🔐 SENDER (Encode & Send)", command=lambda: [selector.destroy(), launch_sender()], width=34).grid(row=1, column=0, pady=8)
    ttk.Button(frm, text="🔓 RECEIVER (Receive & Decode)", command=lambda: [selector.destroy(), launch_receiver()], width=34).grid(row=2, column=0, pady=6)
    
    selector.mainloop()

if __name__ == "__main__":
    main()
