import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.integrate import odeint
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import socket
import json
import pickle

class ChaoticSystem:
    """Base class for chaotic systems"""
    
    def __init__(self, initial_conditions, params):
        self.initial_conditions = np.array(initial_conditions)
        self.params = params
        self.trajectory = None
    
    def simulate(self, t_span, dt=0.01):
        """Simulate the chaotic system"""
        t = np.arange(0, t_span, dt)
        self.trajectory = odeint(self.equations, self.initial_conditions, t)
        return t, self.trajectory
    
    def equations(self, state, t):
        """Override in subclasses"""
        raise NotImplementedError

class ChuaSystem(ChaoticSystem):
    """Chua's circuit"""
    
    def __init__(self, initial_conditions=[0.7, 0.0, 0.0], 
                 params={'alpha': 15.6, 'beta': 28.0, 'a': -1.143, 'b': -0.714}):
        super().__init__(initial_conditions, params)
    
    def equations(self, state, t):
        x, y, z = state
        alpha = self.params['alpha']
        beta = self.params['beta']
        a = self.params['a']
        b = self.params['b']
        
        h = b * x + 0.5 * (a - b) * (abs(x + 1) - abs(x - 1))
        
        dx = alpha * (y - x - h)
        dy = x - y + z
        dz = -beta * y
        
        return [dx, dy, dz]

class LorenzSystem(ChaoticSystem):
    """Lorenz attractor"""
    
    def __init__(self, initial_conditions=[1.0, 1.0, 1.0], 
                 params={'sigma': 10.0, 'rho': 28.0, 'beta': 8.0/3.0}):
        super().__init__(initial_conditions, params)
    
    def equations(self, state, t):
        x, y, z = state
        sigma = self.params['sigma']
        rho = self.params['rho']
        beta = self.params['beta']
        
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        
        return [dx, dy, dz]

class RosslerSystem(ChaoticSystem):
    """Rössler attractor"""
    
    def __init__(self, initial_conditions=[1.0, 1.0, 1.0], 
                 params={'a': 0.2, 'b': 0.2, 'c': 5.7}):
        super().__init__(initial_conditions, params)
    
    def equations(self, state, t):
        x, y, z = state
        a = self.params['a']
        b = self.params['b']
        c = self.params['c']
        
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        
        return [dx, dy, dz]

class ChaoticEncoder:
    """Encode and decode messages using chaotic systems"""
    
    def __init__(self, system):
        self.system = system
    
    def text_to_bits(self, text):
        """Convert text to binary string"""
        return ''.join(format(ord(c), '08b') for c in text)
    
    def bits_to_text(self, bits):
        """Convert binary string to text"""
        chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
        return ''.join(chr(int(c, 2)) for c in chars if len(c) == 8)
    
    def generate_keystream(self, length, component=0):
        """Generate pseudo-random keystream from chaotic trajectory with high sensitivity"""
        if self.system.trajectory is None:
            raise ValueError("System must be simulated first")
        
        # Get all three components
        x_vals = self.system.trajectory[:, 0]
        y_vals = self.system.trajectory[:, 1]
        z_vals = self.system.trajectory[:, 2]
        
        keystream = []
        
        # Generate each bit using precise floating point values
        for i in range(length):
            # Use different parts of trajectory for each bit
            idx = (i * 13) % len(x_vals)  # Prime number for better distribution
            
            # Get raw values (NOT normalized)
            x = x_vals[idx]
            y = y_vals[idx]
            z = z_vals[idx]
            
            # Combine using a chaotic mixing function
            # This is HIGHLY sensitive to initial conditions
            mixed = (x * 1.23456789 + y * 9.87654321 + z * 3.14159265)
            
            # Use the fractional part and multiple decimal places
            # Even 0.0001 difference in initial conditions will change this
            fractional = abs(mixed) - int(abs(mixed))
            
            # Generate bit based on precise decimal representation
            # Convert to string to get exact decimal digits
            decimal_str = f"{fractional:.15f}"  # 15 decimal places
            
            # Sum all decimal digits
            digit_sum = sum(int(d) for d in decimal_str if d.isdigit())
            
            # Bit is determined by whether sum is odd or even
            bit = '1' if digit_sum % 2 == 1 else '0'
            keystream.append(bit)
        
        return ''.join(keystream)
    
    def encode(self, message, t_span=500, dt=0.005, component=0):
        """Encode a message using chaotic system"""
        message_bits = self.text_to_bits(message)
        t, trajectory = self.system.simulate(t_span, dt)
        keystream = self.generate_keystream(len(message_bits), component)
        encoded_bits = ''.join(str(int(m) ^ int(k)) for m, k in zip(message_bits, keystream))
        
        return encoded_bits, message_bits, keystream
    
    def decode(self, encoded_bits, keystream):
        """Decode an encoded message using the keystream"""
        decoded_bits = ''.join(str(int(e) ^ int(k)) for e, k in zip(encoded_bits, keystream))
        decoded_message = self.bits_to_text(decoded_bits)
        
        return decoded_message

def create_system(system_name, initial_conditions):
    """Factory function to create chaotic systems"""
    if system_name == "Lorenz":
        return LorenzSystem(initial_conditions)
    elif system_name == "Chua":
        return ChuaSystem(initial_conditions)
    elif system_name == "Rössler":
        return RosslerSystem(initial_conditions)
    else:
        raise ValueError(f"Unknown system: {system_name}")

class SenderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sender - Chaotic Cryptography")
        self.root.geometry("900x700")
        
        self.server_socket = None
        self.client_socket = None
        self.is_server_running = False
        
        self.setup_ui()
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🔐 SENDER - Chaotic Message Encoder", 
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=10)
        
        # Connection frame
        conn_frame = ttk.LabelFrame(main_frame, text="Connection Settings", padding="10")
        conn_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(conn_frame, text="Port:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.port_entry = ttk.Entry(conn_frame, width=10)
        self.port_entry.insert(0, "5555")
        self.port_entry.grid(row=0, column=1, padx=5)
        
        self.start_server_btn = ttk.Button(conn_frame, text="🟢 Start Server", 
                                           command=self.start_server)
        self.start_server_btn.grid(row=0, column=2, padx=5)
        
        self.stop_server_btn = ttk.Button(conn_frame, text="🔴 Stop Server", 
                                          command=self.stop_server, state=tk.DISABLED)
        self.stop_server_btn.grid(row=0, column=3, padx=5)
        
        # System configuration
        system_frame = ttk.LabelFrame(main_frame, text="Chaotic System Configuration", padding="10")
        system_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(system_frame, text="System:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.system_var = tk.StringVar(value="Lorenz")
        system_combo = ttk.Combobox(system_frame, textvariable=self.system_var, 
                                     values=["Lorenz", "Chua", "Rössler"], state="readonly", width=12)
        system_combo.grid(row=0, column=1, padx=5)
        
        ttk.Label(system_frame, text="Initial X:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.init_x = ttk.Entry(system_frame, width=10)
        self.init_x.insert(0, "1.0")
        self.init_x.grid(row=0, column=3, padx=5)
        
        ttk.Label(system_frame, text="Initial Y:").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.init_y = ttk.Entry(system_frame, width=10)
        self.init_y.insert(0, "1.0")
        self.init_y.grid(row=0, column=5, padx=5)
        
        ttk.Label(system_frame, text="Initial Z:").grid(row=0, column=6, sticky=tk.W, padx=5)
        self.init_z = ttk.Entry(system_frame, width=10)
        self.init_z.insert(0, "1.0")
        self.init_z.grid(row=0, column=7, padx=5)
        
        # Message input
        input_frame = ttk.LabelFrame(main_frame, text="Message to Send", padding="10")
        input_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        input_frame.columnconfigure(0, weight=1)
        
        self.message_input = scrolledtext.ScrolledText(input_frame, height=4, wrap=tk.WORD)
        self.message_input.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        self.message_input.insert(1.0, "Secret message from chaos!")
        
        # Send button
        send_frame = ttk.Frame(main_frame)
        send_frame.grid(row=4, column=0, pady=10)
        
        self.send_btn = ttk.Button(send_frame, text="📤 Encode & Send Message", 
                                    command=self.send_message, state=tk.DISABLED)
        self.send_btn.grid(row=0, column=0, padx=5)
        
        # Log
        log_frame = ttk.LabelFrame(main_frame, text="Activity Log", padding="10")
        log_frame.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Start server to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=5)
    
    def log(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
    
    def start_server(self):
        """Start the server socket"""
        try:
            port = int(self.port_entry.get())
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', port))
            self.server_socket.listen(1)
            
            self.is_server_running = True
            self.start_server_btn.config(state=tk.DISABLED)
            self.stop_server_btn.config(state=tk.NORMAL)
            
            self.log(f"✓ Server started on port {port}")
            self.log("Waiting for receiver to connect...")
            self.status_var.set(f"Server running on port {port}")
            
            # Accept connections in a separate thread
            thread = threading.Thread(target=self.accept_connections, daemon=True)
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start server: {str(e)}")
            self.log(f"✗ Error starting server: {str(e)}")
    
    def accept_connections(self):
        """Accept incoming connections"""
        try:
            self.client_socket, addr = self.server_socket.accept()
            self.log(f"✓ Receiver connected from {addr}")
            self.status_var.set(f"Connected to receiver at {addr}")
            self.send_btn.config(state=tk.NORMAL)
        except Exception as e:
            if self.is_server_running:
                self.log(f"✗ Connection error: {str(e)}")
    
    def stop_server(self):
        """Stop the server"""
        self.is_server_running = False
        
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
        
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
        
        self.start_server_btn.config(state=tk.NORMAL)
        self.stop_server_btn.config(state=tk.DISABLED)
        self.send_btn.config(state=tk.DISABLED)
        
        self.log("✓ Server stopped")
        self.status_var.set("Server stopped")
    
    def send_message(self):
        """Encode and send the message"""
        if not self.client_socket:
            messagebox.showwarning("Warning", "No receiver connected!")
            return
        
        message = self.message_input.get(1.0, tk.END).strip()
        if not message:
            messagebox.showwarning("Warning", "Please enter a message!")
            return
        
        try:
            # Get initial conditions
            x = float(self.init_x.get())
            y = float(self.init_y.get())
            z = float(self.init_z.get())
            initial = [x, y, z]
            
            # Create system and encode
            self.log("Encoding message...")
            system = create_system(self.system_var.get(), initial)
            encoder = ChaoticEncoder(system)
            encoded_bits, _, _ = encoder.encode(message)
            
            # Prepare data packet
            data_packet = {
                'encoded_message': encoded_bits,
                'system_type': self.system_var.get(),
                'message_length': len(message)
            }
            
            # Send data
            serialized = pickle.dumps(data_packet)
            self.client_socket.sendall(len(serialized).to_bytes(4, 'big'))
            self.client_socket.sendall(serialized)
            
            self.log(f"✓ Sent encoded message ({len(encoded_bits)} bits)")
            self.log(f"  System: {self.system_var.get()}")
            self.log(f"  Initial conditions: [{x}, {y}, {z}]")
            self.log(f"  Original message: '{message}'")
            self.status_var.set("Message sent successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send message: {str(e)}")
            self.log(f"✗ Error: {str(e)}")

class ReceiverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Receiver - Chaotic Cryptography")
        self.root.geometry("900x700")
        
        self.client_socket = None
        self.received_data = None
        
        self.setup_ui()
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🔓 RECEIVER - Chaotic Message Decoder", 
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=10)
        
        # Connection frame
        conn_frame = ttk.LabelFrame(main_frame, text="Connection Settings", padding="10")
        conn_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(conn_frame, text="Host:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.host_entry = ttk.Entry(conn_frame, width=15)
        self.host_entry.insert(0, "localhost")
        self.host_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(conn_frame, text="Port:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.port_entry = ttk.Entry(conn_frame, width=10)
        self.port_entry.insert(0, "5555")
        self.port_entry.grid(row=0, column=3, padx=5)
        
        self.connect_btn = ttk.Button(conn_frame, text="🔌 Connect to Sender", 
                                      command=self.connect_to_sender)
        self.connect_btn.grid(row=0, column=4, padx=5)
        
        self.disconnect_btn = ttk.Button(conn_frame, text="❌ Disconnect", 
                                         command=self.disconnect, state=tk.DISABLED)
        self.disconnect_btn.grid(row=0, column=5, padx=5)
        
        # Decryption key frame
        key_frame = ttk.LabelFrame(main_frame, text="Decryption Key (Initial Conditions)", padding="10")
        key_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(key_frame, text="System:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.system_var = tk.StringVar(value="Lorenz")
        system_combo = ttk.Combobox(key_frame, textvariable=self.system_var, 
                                     values=["Lorenz", "Chua", "Rössler"], state="readonly", width=12)
        system_combo.grid(row=0, column=1, padx=5)
        
        ttk.Label(key_frame, text="Initial X:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.init_x = ttk.Entry(key_frame, width=12)
        self.init_x.insert(0, "1.0")
        self.init_x.grid(row=0, column=3, padx=5)
        
        ttk.Label(key_frame, text="Initial Y:").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.init_y = ttk.Entry(key_frame, width=12)
        self.init_y.insert(0, "1.0")
        self.init_y.grid(row=0, column=5, padx=5)
        
        ttk.Label(key_frame, text="Initial Z:").grid(row=0, column=6, sticky=tk.W, padx=5)
        self.init_z = ttk.Entry(key_frame, width=12)
        self.init_z.insert(0, "1.0")
        self.init_z.grid(row=0, column=7, padx=5)
        
        # Receive and decode buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=3, column=0, pady=10)
        
        self.receive_btn = ttk.Button(action_frame, text="📥 Receive Message", 
                                      command=self.receive_message, state=tk.DISABLED)
        self.receive_btn.grid(row=0, column=0, padx=5)
        
        self.decode_btn = ttk.Button(action_frame, text="🔓 Decode Message", 
                                     command=self.decode_message, state=tk.DISABLED)
        self.decode_btn.grid(row=0, column=1, padx=5)
        
        # Results
        results_frame = ttk.LabelFrame(main_frame, text="Decoded Message", padding="10")
        results_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)
        results_frame.columnconfigure(0, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=5, wrap=tk.WORD,
                                                      font=('Arial', 12, 'bold'))
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Log
        log_frame = ttk.LabelFrame(main_frame, text="Activity Log", padding="10")
        log_frame.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Connect to sender")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=5)
    
    def log(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
    
    def connect_to_sender(self):
        """Connect to the sender"""
        try:
            host = self.host_entry.get()
            port = int(self.port_entry.get())
            
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((host, port))
            
            self.log(f"✓ Connected to sender at {host}:{port}")
            self.status_var.set(f"Connected to {host}:{port}")
            
            self.connect_btn.config(state=tk.DISABLED)
            self.disconnect_btn.config(state=tk.NORMAL)
            self.receive_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Connection failed: {str(e)}")
            self.log(f"✗ Connection error: {str(e)}")
    
    def disconnect(self):
        """Disconnect from sender"""
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
        
        self.connect_btn.config(state=tk.NORMAL)
        self.disconnect_btn.config(state=tk.DISABLED)
        self.receive_btn.config(state=tk.DISABLED)
        self.decode_btn.config(state=tk.DISABLED)
        
        self.log("✓ Disconnected from sender")
        self.status_var.set("Disconnected")
    
    def receive_message(self):
        """Receive encoded message"""
        if not self.client_socket:
            messagebox.showwarning("Warning", "Not connected to sender!")
            return
        
        try:
            self.log("Waiting to receive message...")
            self.status_var.set("Receiving...")
            
            # Receive data length
            length_bytes = self.client_socket.recv(4)
            if not length_bytes:
                raise Exception("Connection closed")
            
            data_length = int.from_bytes(length_bytes, 'big')
            
            # Receive data
            data = b''
            while len(data) < data_length:
                chunk = self.client_socket.recv(min(4096, data_length - len(data)))
                if not chunk:
                    raise Exception("Connection closed")
                data += chunk
            
            self.received_data = pickle.loads(data)
            
            self.log(f"✓ Received encoded message ({len(self.received_data['encoded_message'])} bits)")
            self.log(f"  System type: {self.received_data['system_type']}")
            self.log(f"  Message length: {self.received_data['message_length']} characters")
            self.log(f"  Encoded data (first 80 bits): {self.received_data['encoded_message'][:80]}...")
            self.log(f"\n⚠️  To decode: Select '{self.received_data['system_type']}' system and enter correct initial conditions!")
            
            # Auto-select the system type
            self.system_var.set(self.received_data['system_type'])
            
            self.decode_btn.config(state=tk.NORMAL)
            self.status_var.set("Message received - Ready to decode")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to receive: {str(e)}")
            self.log(f"✗ Error: {str(e)}")
    
    def decode_message(self):
        """Decode the received message"""
        if not self.received_data:
            messagebox.showwarning("Warning", "No message received yet!")
            return
        
        try:
            # Get initial conditions from GUI
            x = float(self.init_x.get())
            y = float(self.init_y.get())
            z = float(self.init_z.get())
            initial = [x, y, z]
            
            # Get system type from GUI
            system_type = self.system_var.get()
            
            self.log("Decoding message...")
            self.log(f"  Using system: {system_type}")
            self.log(f"  Using initial conditions: [{x}, {y}, {z}]")
            
            # Create system with receiver's initial conditions
            system = create_system(system_type, initial)
            encoder = ChaoticEncoder(system)
            
            # Generate keystream with the SAME parameters as sender (longer simulation)
            encoder.system.simulate(500, 0.005)
            keystream = encoder.generate_keystream(len(self.received_data['encoded_message']), component=0)
            
            # Debug: Show keystream comparison
            sender_keystream = self.received_data.get('keystream_sample', 'N/A')
            receiver_keystream = keystream[:80]

            
            # Calculate keystream match percentage
            if sender_keystream != 'N/A':
                matches = sum(s == r for s, r in zip(sender_keystream, receiver_keystream))
                match_percent = (matches / len(sender_keystream)) * 100
                self.log(f"  Keystream match: {matches}/{len(sender_keystream)} bits ({match_percent:.1f}%)")
                
                if match_percent < 95:
                    self.log(f"\n  ⚠️  WARNING: Keystreams don't match! Wrong initial conditions!")
            
            # Decode
            decoded_message = encoder.decode(self.received_data['encoded_message'], keystream)
            
            # Display result
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, decoded_message)
            
            self.log(f"✓ Message decoded!")
            self.log(f"  Decoded text: '{decoded_message}'")
            
            # Verify if it makes sense (basic check)
            printable_ratio = sum(c.isprintable() for c in decoded_message) / len(decoded_message) if decoded_message else 0
            if printable_ratio < 0.7:
                self.log(f"\n⚠️  WARNING: Decoded message contains many non-printable characters!")
                self.log(f"  This likely means WRONG initial conditions or system type.")
                self.log(f"  Printable characters: {printable_ratio*100:.1f}%")
            else:
                self.log(f"  Message appears valid (printable: {printable_ratio*100:.1f}%)")
            
            self.status_var.set("Message decoded!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Decoding failed: {str(e)}")
            self.log(f"✗ Decoding error: {str(e)}")
            self.log("  Hint: Check if initial conditions and system type match the sender's values")

def main():
    """Launch selector GUI"""
    def launch_sender():
        root = tk.Tk()
        SenderGUI(root)
        root.mainloop()
    
    def launch_receiver():
        root = tk.Tk()
        ReceiverGUI(root)
        root.mainloop()
    
    # Selector window
    selector = tk.Tk()
    selector.title("Chaotic Cryptography - Select Mode")
    selector.geometry("400x250")
    
    main_frame = ttk.Frame(selector, padding="20")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    selector.columnconfigure(0, weight=1)
    selector.rowconfigure(0, weight=1)
    
    ttk.Label(main_frame, text="Chaotic Cryptography System", 
              font=('Arial', 16, 'bold')).grid(row=0, column=0, pady=20)
    
    ttk.Label(main_frame, text="Select your role:", 
              font=('Arial', 12)).grid(row=1, column=0, pady=10)
    
    ttk.Button(main_frame, text="🔐 SENDER (Encode & Send)", 
               command=lambda: [selector.destroy(), launch_sender()],
               width=30).grid(row=2, column=0, pady=10)
    
    ttk.Button(main_frame, text="🔓 RECEIVER (Receive & Decode)", 
               command=lambda: [selector.destroy(), launch_receiver()],
               width=30).grid(row=3, column=0, pady=10)
    
    selector.mainloop()

if __name__ == "__main__":
    main()