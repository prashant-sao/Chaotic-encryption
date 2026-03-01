# receiver_gui.py
# Receiver GUI to connect to sender, receive packets and decode text/images.
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import socket, pickle, threading, os
from chaos_systems import create_system
from encoder import ChaoticEncoder
from visualizer import show_receiver_waveforms, show_image

class ReceiverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chaotic Cryptography — Receiver")
        self.root.geometry("920x640")
        self.sock = None
        self.received_packet = None
        self.last_decoded_bits = None
        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=8); frm.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frm, text="🔓 RECEIVER — Chaotic Cryptography", font=('Arial', 14, 'bold')).pack(anchor='w')

        connf = ttk.LabelFrame(frm, text="Connection", padding=6); connf.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(connf, text="Host:").grid(row=0, column=0, padx=4); self.host_e = ttk.Entry(connf, width=16); self.host_e.insert(0,"localhost"); self.host_e.grid(row=0, column=1)
        ttk.Label(connf, text="Port:").grid(row=0, column=2, padx=4); self.port_e = ttk.Entry(connf, width=8); self.port_e.insert(0,"5555"); self.port_e.grid(row=0, column=3)
        ttk.Button(connf, text="Connect", command=self.connect).grid(row=0, column=4, padx=6)
        ttk.Button(connf, text="Disconnect", command=self.disconnect).grid(row=0, column=5, padx=6)

        keyf = ttk.LabelFrame(frm, text="Decryption Key (initial conditions)", padding=6); keyf.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(keyf, text="System:").grid(row=0, column=0, padx=4); self.sys_var = tk.StringVar(value="Lorenz")
        ttk.Combobox(keyf, textvariable=self.sys_var, values=["Lorenz","Chua","Rössler"], state="readonly", width=10).grid(row=0, column=1, padx=4)
        ttk.Label(keyf, text="Init X:").grid(row=0, column=2, padx=4); self.icx = ttk.Entry(keyf, width=8); self.icx.insert(0,"1.0"); self.icx.grid(row=0, column=3)
        ttk.Label(keyf, text="Init Y:").grid(row=0, column=4, padx=4); self.icy = ttk.Entry(keyf, width=8); self.icy.insert(0,"1.0"); self.icy.grid(row=0, column=5)
        ttk.Label(keyf, text="Init Z:").grid(row=0, column=6, padx=4); self.icz = ttk.Entry(keyf, width=8); self.icz.insert(0,"1.0"); self.icz.grid(row=0, column=7)

        actf = ttk.Frame(frm); actf.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(actf, text="Receive Packet", command=self.receive_packet).pack(side=tk.LEFT, padx=6)
        self.decode_btn = ttk.Button(actf, text="Decode Packet", command=self.decode_packet, state=tk.DISABLED); self.decode_btn.pack(side=tk.LEFT, padx=6)
        ttk.Button(actf, text="Visualize Waves", command=self.visualize_waves).pack(side=tk.LEFT, padx=6)

        resf = ttk.LabelFrame(frm, text="Decoded Result", padding=6); resf.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.result_text = scrolledtext.ScrolledText(resf, height=8); self.result_text.pack(fill=tk.BOTH, expand=True)

        logf = ttk.LabelFrame(frm, text="Activity Log", padding=6); logf.pack(fill=tk.BOTH, expand=False, padx=6)
        self.log_text = scrolledtext.ScrolledText(logf, height=8); self.log_text.pack(fill=tk.BOTH)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(frm, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X, pady=(6,0))

    def log(self, msg):
        self.log_text.insert(tk.END, f"{msg}\n"); self.log_text.see(tk.END)

    def connect(self):
        try:
            host = self.host_e.get(); port = int(self.port_e.get())
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((host, port))
            self.log(f"Connected to {host}:{port}")
            self.status_var.set(f"Connected to {host}:{port}")
        except Exception as e:
            messagebox.showerror("Error", f"Connect failed: {e}"); self.log(f"Connect error: {e}")

    def disconnect(self):
        if self.sock: self.sock.close(); self.sock = None
        self.log("Disconnected"); self.status_var.set("Disconnected")

    def receive_packet(self):
        if not self.sock:
            messagebox.showwarning("Warning", "Not connected."); return
        try:
            self.log("Receiving...")
            length_bytes = self.sock.recv(4)
            if not length_bytes:
                raise Exception("Connection closed")
            data_length = int.from_bytes(length_bytes, 'big')
            data = b''
            while len(data) < data_length:
                chunk = self.sock.recv(min(4096, data_length - len(data)))
                if not chunk:
                    raise Exception("Connection closed")
                data += chunk
            pkt = pickle.loads(data)
            self.received_packet = pkt
            self.log(f"Packet received. Type: {pkt.get('type','?')}")
            self.log(f"System hint: {pkt.get('system','N/A')}; Init hint: {pkt.get('init','N/A')}")
            self.decode_btn.config(state=tk.NORMAL)
            self.status_var.set("Packet received")
        except Exception as e:
            messagebox.showerror("Error", f"Receive failed: {e}"); self.log(f"Receive error: {e}")

    def decode_packet(self):
        if not self.received_packet:
            messagebox.showwarning("Warning", "No packet."); return
        pkt = self.received_packet
        try:
            x = float(self.icx.get()); y = float(self.icy.get()); z = float(self.icz.get())
            system_name = self.sys_var.get()
            sys = create_system(system_name, (x,y,z))
            enc = ChaoticEncoder(sys)
            
            # --- Text Decryption (Legacy/Unchanged) ---
            if pkt.get('type') == 'text':
                enc.system.simulate(500.0, 0.005)
                # Uses default weights (standard mix)
                ks = enc.generate_keystream(len(pkt['encoded_bits']))
                decoded_message, decoded_bits = enc.decode_text(pkt['encoded_bits'], ks)
                
                self.result_text.delete('1.0', tk.END); self.result_text.insert('1.0', decoded_message)
                self.last_decoded_bits = decoded_bits
                self.log(f"Decoded text. Length: {len(decoded_message)} chars")
                
                # Check single keystream sample
                sample = pkt.get('keystream_sample', '')
                if sample:
                    matches = sum(1 for a,b in zip(sample, ks[:len(sample)]) if a==b)
                    pct = matches / len(sample) * 100
                    self.log(f"Keystream match: {matches}/{len(sample)} ({pct:.1f}%)")

            # --- Image Decryption (Updated for RGB) ---
            elif pkt.get('type') == 'image':
                enc.system.simulate(600.0, 0.002)
                
                enc_bits = pkt['encrypted_bits']
                order = pkt['order']
                shape = tuple(pkt['shape']) # Expected (H, W, 3)
                
                # 1. Calculate bit length per channel
                # Shape is (Height, Width, 3), so channel pixels = H * W
                height, width = shape[0], shape[1]
                bits_per_channel = height * width * 8
                
                # 2. Generate 3 separate keystreams using the known weights
                w_R = (1.234, 2.345, 0.987)
                w_G = (0.987, 1.234, 2.345)
                w_B = (2.345, 0.987, 1.234)
                
                ks_R = enc.generate_keystream(bits_per_channel, weights=w_R)
                ks_G = enc.generate_keystream(bits_per_channel, weights=w_G)
                ks_B = enc.generate_keystream(bits_per_channel, weights=w_B)
                
                ks_list = [ks_R, ks_G, ks_B]
                
                # 3. Decode
                img = enc.decode_image(enc_bits, ks_list, order, shape)
                
                # 4. Save & Show
                out = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png")], title="Save decoded image as")
                if out:
                    from PIL import Image
                    Image.fromarray(img).save(out)
                    self.log(f"Decoded image saved to: {out}")
                
                show_image(self.root, img, title="Decoded Image")
                
                # 5. Verify Keystreams (Check all 3 samples)
                samples = pkt.get('keystream_samples', [])
                if samples and len(samples) == 3:
                    total_match = 0
                    total_len = 0
                    for i, sample in enumerate(samples):
                        k = ks_list[i]
                        match = sum(1 for a,b in zip(sample, k[:len(sample)]) if a==b)
                        total_match += match
                        total_len += len(sample)
                    
                    if total_len > 0:
                        pct = total_match / total_len * 100
                        self.log(f"Keystream Integrity (Avg): {pct:.1f}%")
            else:
                self.log("Unknown packet type.")
        except Exception as e:
            messagebox.showerror("Error", f"Decode failed: {e}"); self.log(f"Decode error: {e}")

    def visualize_waves(self):
        if not self.received_packet:
            messagebox.showwarning("Warning", "No packet to visualize."); return
        pkt = self.received_packet
        if pkt.get('type') == 'text':
            enc_bits = pkt['encoded_bits']
            # try to simulate current key and produce decode for visualization
            try:
                x = float(self.icx.get()); y = float(self.icy.get()); z = float(self.icz.get())
                sys = create_system(self.sys_var.get(), (x,y,z))
                enc = ChaoticEncoder(sys)
                enc.system.simulate(500.0, 0.005)
                ks = enc.generate_keystream(len(enc_bits))
                _, decoded_bits = enc.decode_text(enc_bits, ks)
                show_receiver_waveforms(self.root, enc_bits, decoded_bits)
            except Exception as e:
                messagebox.showerror("Error", f"Visualize failed: {e}"); self.log(f"Visualize error: {e}")
        elif pkt.get('type') == 'image':
            # just show the encrypted data length and hint
            messagebox.showinfo("Info", "Use Decode Packet (with correct initial conditions) to view the decoded image.")
