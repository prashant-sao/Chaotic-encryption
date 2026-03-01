# sender_gui.py
# Modern tabbed sender GUI (text + image + lyapunov visualization)
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading, socket, pickle, os
import numpy as np
from chaos_systems import create_system
from encoder import ChaoticEncoder
from visualizer import show_sender_waveforms, show_image

class SenderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chaotic Cryptography — Sender")
        self.root.geometry("1000x700")

        self.server_socket = None
        self.client_socket = None
        self.is_listening = False

        # artifacts
        self.last_t = None
        self.last_traj = None
        self.last_msg_bits = None
        self.last_encoded_bits = None
        self.last_keystream = None
        self.last_lyap = None

        self.last_image = None

        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=8); frm.pack(fill=tk.BOTH, expand=True)
        hb = ttk.Label(frm, text="🔐 SENDER — Chaotic Cryptography", font=('Arial', 14, 'bold'))
        hb.pack(anchor='w', pady=(0,6))

        top = ttk.Frame(frm); top.pack(fill=tk.X, pady=4)
        conn = ttk.LabelFrame(top, text="Connection", padding=6); conn.pack(side=tk.LEFT, padx=6)
        ttk.Label(conn, text="Port:").grid(row=0, column=0, padx=4)
        self.port_entry = ttk.Entry(conn, width=8); self.port_entry.insert(0, "5555"); self.port_entry.grid(row=0, column=1)
        self.start_btn = ttk.Button(conn, text="Start Server", command=self.start_server); self.start_btn.grid(row=0, column=2, padx=6)
        self.stop_btn = ttk.Button(conn, text="Stop Server", command=self.stop_server, state=tk.DISABLED); self.stop_btn.grid(row=0, column=3, padx=6)

        sysf = ttk.LabelFrame(top, text="Chaotic System", padding=6); sysf.pack(side=tk.RIGHT, padx=6)
        ttk.Label(sysf, text="System:").grid(row=0, column=0, padx=4)
        self.system_var = tk.StringVar(value="Lorenz")
        ttk.Combobox(sysf, textvariable=self.system_var, values=["Lorenz","Chua","Rössler"], state="readonly", width=10).grid(row=0, column=1, padx=4)
        ttk.Label(sysf, text="Init(X,Y,Z):").grid(row=0, column=2, padx=4)
        self.ic_x = ttk.Entry(sysf, width=6); self.ic_x.insert(0, "1.0"); self.ic_x.grid(row=0, column=3)
        self.ic_y = ttk.Entry(sysf, width=6); self.ic_y.insert(0, "1.0"); self.ic_y.grid(row=0, column=4)
        self.ic_z = ttk.Entry(sysf, width=6); self.ic_z.insert(0, "1.0"); self.ic_z.grid(row=0, column=5)

        # Notebook tabs
        self.nb = ttk.Notebook(frm)
        self.nb.pack(fill=tk.BOTH, expand=True, pady=6)

        # Text tab
        txt_tab = ttk.Frame(self.nb); self.nb.add(txt_tab, text="Text Encryption")
        ttk.Label(txt_tab, text="Message:").pack(anchor='w')
        self.msg_text = scrolledtext.ScrolledText(txt_tab, height=5); self.msg_text.pack(fill=tk.X, padx=6)
        self.msg_text.insert('1.0', "Secret message from chaos!")

        txt_btns = ttk.Frame(txt_tab); txt_btns.pack(fill=tk.X, pady=6, padx=6)
        ttk.Button(txt_btns, text="Encode & Send Text", command=self.send_text).pack(side=tk.LEFT, padx=6)
        ttk.Button(txt_btns, text="Simulate & Visualize", command=self.simulate_text_and_visualize).pack(side=tk.LEFT, padx=6)

        # Image tab
        img_tab = ttk.Frame(self.nb); self.nb.add(img_tab, text="Image Encryption")
        ttk.Label(img_tab, text="Image (RGB Colour Supported):").pack(anchor='w')
        self.img_path_var = tk.StringVar()
        path_frame = ttk.Frame(img_tab); path_frame.pack(fill=tk.X, padx=6, pady=4)
        ttk.Entry(path_frame, textvariable=self.img_path_var, width=60).pack(side=tk.LEFT, padx=(0,6))
        ttk.Button(path_frame, text="Browse...", command=self.browse_image).pack(side=tk.LEFT)
        img_btns = ttk.Frame(img_tab); img_btns.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(img_btns, text="Encrypt & Send Image", command=self.send_image).pack(side=tk.LEFT, padx=6)
        ttk.Button(img_btns, text="Simulate & Preview", command=self.simulate_image_and_preview).pack(side=tk.LEFT, padx=6)

        # Lyapunov tab (visualization)
        lyap_tab = ttk.Frame(self.nb); self.nb.add(lyap_tab, text="Lyapunov / Analysis")
        self.lyap_eps_var = tk.StringVar(value="1e-8")
        ttk.Label(lyap_tab, text="Perturb eps:").grid(row=0, column=0, padx=6, pady=6, sticky='w')
        ttk.Entry(lyap_tab, textvariable=self.lyap_eps_var, width=12).grid(row=0, column=1, sticky='w')
        ttk.Button(lyap_tab, text="Simulate System (long) & Plot Lyapunov", command=self.simulate_system_and_plot).grid(row=0, column=2, padx=8)

        # Activity log
        logf = ttk.LabelFrame(frm, text="Activity Log", padding=6); logf.pack(fill=tk.BOTH, expand=False, padx=6, pady=(6,0))
        self.log_text = scrolledtext.ScrolledText(logf, height=8); self.log_text.pack(fill=tk.BOTH)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(frm, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X, pady=(6,0))

    def log(self, msg):
        self.log_text.insert(tk.END, f"{msg}\n"); self.log_text.see(tk.END)

    # Connection methods
    def start_server(self):
        try:
            port = int(self.port_entry.get())
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', port))
            self.server_socket.listen(1)
            self.is_listening = True
            self.start_btn.config(state=tk.DISABLED); self.stop_btn.config(state=tk.NORMAL)
            self.log(f"Server started on port {port}")
            self.status_var.set(f"Listening on port {port}")
            threading.Thread(target=self._accept_thread, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Could not start server: {e}"); self.log(f"Error starting server: {e}")

    def _accept_thread(self):
        try:
            self.client_socket, addr = self.server_socket.accept()
            self.log(f"Receiver connected from {addr}")
            self.status_var.set(f"Connected to {addr}")
        except Exception as e:
            if self.is_listening:
                self.log(f"Accept error: {e}")

    def stop_server(self):
        self.is_listening = False
        if self.client_socket: self.client_socket.close(); self.client_socket = None
        if self.server_socket: self.server_socket.close(); self.server_socket = None
        self.start_btn.config(state=tk.NORMAL); self.stop_btn.config(state=tk.DISABLED)
        self.log("Server stopped"); self.status_var.set("Server stopped")

    # Helpers to build system + encoder
    def _build_system_encoder(self):
        x = float(self.ic_x.get()); y = float(self.ic_y.get()); z = float(self.ic_z.get())
        sys = create_system(self.system_var.get(), (x,y,z))
        enc = ChaoticEncoder(sys)
        return sys, enc

    # Text operations
    def simulate_text_and_visualize(self):
        message = self.msg_text.get("1.0", tk.END).strip()
        if not message:
            messagebox.showwarning("Warning", "Enter message first."); return
        sys, enc = self._build_system_encoder()
        enc.system.simulate(500.0, 0.005)
        enc_bits, msg_bits, ks, t, traj, lyap = enc.encode_text(message, t_span=500.0, dt=0.005, lyap_eps=float(self.lyap_eps_var.get()))
        # store
        self.last_t = t; self.last_traj = traj; self.last_msg_bits = msg_bits; self.last_encoded_bits = enc_bits; self.last_keystream = ks; self.last_lyap = lyap
        show_sender_waveforms(self.root, t, traj, msg_bits, enc_bits, lyap_results=lyap)

    def send_text(self):
        if not self.client_socket:
            messagebox.showwarning("Warning", "No receiver connected."); return
        try:
            sys, enc = self._build_system_encoder()
            message = self.msg_text.get("1.0", tk.END).strip()
            if not message:
                messagebox.showwarning("Warning", "Empty message."); return
            enc.system.simulate(500.0, 0.005)
            encoded_bits, msg_bits, ks, t, traj, lyap = enc.encode_text(message, t_span=500.0, dt=0.005, lyap_eps=float(self.lyap_eps_var.get()))
            # prepare packet
            pkt = {'type':'text', 'encoded_bits': encoded_bits, 'keystream_sample': ks[:160], 'system': self.system_var.get(), 'init': (float(self.ic_x.get()), float(self.ic_y.get()), float(self.ic_z.get())), 'message_length': len(message)}
            data = pickle.dumps(pkt)
            self.client_socket.sendall(len(data).to_bytes(4,'big')); self.client_socket.sendall(data)
            # store for visualization
            self.last_t = t; self.last_traj = traj; self.last_msg_bits = msg_bits; self.last_encoded_bits = encoded_bits; self.last_keystream = ks; self.last_lyap = lyap
            self.log(f"Sent encoded text ({len(encoded_bits)} bits).")
        except Exception as e:
            messagebox.showerror("Error", f"Send failed: {e}"); self.log(f"Send error: {e}")

    # Image operations
    def browse_image(self):
        p = filedialog.askopenfilename(title="Select image", filetypes=[("Images","*.png *.jpg *.jpeg *.bmp")])
        if p:
            self.img_path_var.set(p)

    def simulate_image_and_preview(self):
        path = self.img_path_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showwarning("Warning", "Select a valid image first."); return
        
        sys, enc = self._build_system_encoder()
        
        # Changed: Use RGB loader
        img = enc.load_image_rgb(path)
        
        enc.system.simulate(600.0, 0.002)
        
        # Changed: Unpack 'ks_list' instead of single 'ks'
        # encode_image now returns a list of 3 keystreams [ks_R, ks_G, ks_B]
        encrypted_bits, ks_list, order_list, shape, t, traj = enc.encode_image(img, t_span=600.0, dt=0.002)
        
        # Changed: Pass 'ks_list' to decode_image
        dec_img = enc.decode_image(encrypted_bits, ks_list, order_list, shape)
        
        self.last_image = {'original': img, 'encrypted_bits': encrypted_bits, 'keystream': ks_list, 'order': order_list, 'shape': shape, 't': t, 'traj': traj}
        
        show_image(self.root, img, title="Original Image (RGB)")
        show_image(self.root, dec_img, title="Decrypted/Recovered (RGB)")

    def send_image(self):
        if not self.client_socket:
            messagebox.showwarning("Warning", "No receiver connected."); return
        path = self.img_path_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showwarning("Warning", "Select a valid image first."); return
        try:
            sys, enc = self._build_system_encoder()
            
            # Changed: Use RGB loader
            img = enc.load_image_rgb(path)
            
            enc.system.simulate(600.0, 0.002)
            
            # Changed: Unpack 'ks_list'
            encrypted_bits, ks_list, order_list, shape, t, traj = enc.encode_image(img, t_span=600.0, dt=0.002)
            
            # Changed: Packet structure
            # We now send samples of ALL 3 keystreams so the receiver can verify them
            ks_samples = [k[:160] for k in ks_list]
            
            pkt = {
                'type': 'image', 
                'encrypted_bits': encrypted_bits, 
                'keystream_samples': ks_samples, # Renamed to plural and contains list
                'order': order_list, 
                'shape': shape, 
                'system': self.system_var.get(), 
                'init': (float(self.ic_x.get()), float(self.ic_y.get()), float(self.ic_z.get()))
            }
            
            data = pickle.dumps(pkt)
            self.client_socket.sendall(len(data).to_bytes(4,'big')); self.client_socket.sendall(data)
            
            self.last_image = {'original': img, 'encrypted_bits': encrypted_bits, 'keystream': ks_list, 'order': order_list, 'shape': shape, 't': t, 'traj': traj}
            self.log(f"Sent encrypted RGB image (shape={shape}, bits={len(encrypted_bits)}).")
        except Exception as e:
            messagebox.showerror("Error", f"Image send failed: {e}"); self.log(f"Image send error: {e}")

    # Lyapunov / system plotting
    def simulate_system_and_plot(self):
        sys, enc = self._build_system_encoder()
        eps = float(self.lyap_eps_var.get())
        t, traj = enc.system.simulate(800.0, 0.005)
        sep, ln_sep, ftle = enc.system.compute_lyapunov(t, traj, eps=eps, perturb_component=0)
        lyap = {'t': t, 'sep': sep, 'ln_sep': ln_sep, 'ftle': ftle}
        # use the same waveform visualiser but produce dummy message/encoded bits to show
        dummy_msg = 'A' * 16
        dummy_msg_bits = enc.text_to_bits(dummy_msg)
        dummy_enc_bits = ''.join('1' for _ in dummy_msg_bits)
        show_sender_waveforms(self.root, t, traj, dummy_msg_bits, dummy_enc_bits, lyap_results=lyap)
        self.log("Lyapunov simulation complete.")
