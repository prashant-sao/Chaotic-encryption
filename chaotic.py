import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.integrate import odeint
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading

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
        """Generate pseudo-random keystream from chaotic trajectory"""
        if self.system.trajectory is None:
            raise ValueError("System must be simulated first")
        
        values = self.system.trajectory[:, component]
        normalized = (values - values.min()) / (values.max() - values.min())
        threshold = 0.5
        keystream = ''.join('1' if v > threshold else '0' for v in normalized[:length])
        
        return keystream
    
    def encode(self, message, t_span=100, dt=0.01, component=0):
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

class ChaoticCryptoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chaotic Systems Cryptography")
        self.root.geometry("1200x800")
        
        self.encoded_bits = None
        self.keystream = None
        self.current_system = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Chaotic Systems Message Encoder/Decoder", 
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=10)
        
        # System selection frame
        system_frame = ttk.LabelFrame(main_frame, text="System Configuration", padding="10")
        system_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(system_frame, text="Chaotic System:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.system_var = tk.StringVar(value="Lorenz")
        system_combo = ttk.Combobox(system_frame, textvariable=self.system_var, 
                                     values=["Lorenz", "Chua", "Rössler"], state="readonly", width=15)
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
        
        # Message input frame
        input_frame = ttk.LabelFrame(main_frame, text="Message Input", padding="10")
        input_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        input_frame.columnconfigure(0, weight=1)
        
        self.message_input = scrolledtext.ScrolledText(input_frame, height=4, wrap=tk.WORD)
        self.message_input.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        self.message_input.insert(1.0, "Hello from chaos theory!")
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, pady=10)
        
        self.encode_btn = ttk.Button(button_frame, text="🔒 Encode Message", 
                                      command=self.encode_message)
        self.encode_btn.grid(row=0, column=0, padx=5)
        
        self.decode_btn = ttk.Button(button_frame, text="🔓 Decode Message", 
                                      command=self.decode_message, state=tk.DISABLED)
        self.decode_btn.grid(row=0, column=1, padx=5)
        
        self.visualize_btn = ttk.Button(button_frame, text="📊 Visualize Attractor", 
                                         command=self.visualize_attractor)
        self.visualize_btn.grid(row=0, column=2, padx=5)
        
        self.clear_btn = ttk.Button(button_frame, text="🗑️ Clear", 
                                     command=self.clear_all)
        self.clear_btn.grid(row=0, column=3, padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, wrap=tk.WORD)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=5)
    
    def get_system(self):
        """Create the selected chaotic system"""
        try:
            x = float(self.init_x.get())
            y = float(self.init_y.get())
            z = float(self.init_z.get())
            initial = [x, y, z]
        except ValueError:
            messagebox.showerror("Error", "Invalid initial conditions. Using defaults.")
            initial = [1.0, 1.0, 1.0]
        
        system_name = self.system_var.get()
        
        if system_name == "Lorenz":
            return LorenzSystem(initial)
        elif system_name == "Chua":
            return ChuaSystem(initial)
        elif system_name == "Rössler":
            return RosslerSystem(initial)
    
    def encode_message(self):
        """Encode the message"""
        message = self.message_input.get(1.0, tk.END).strip()
        
        if not message:
            messagebox.showwarning("Warning", "Please enter a message to encode.")
            return
        
        self.status_var.set("Encoding...")
        self.encode_btn.config(state=tk.DISABLED)
        self.results_text.delete(1.0, tk.END)
        
        def encode_thread():
            try:
                system = self.get_system()
                self.current_system = system
                encoder = ChaoticEncoder(system)
                
                self.encoded_bits, original_bits, self.keystream = encoder.encode(
                    message, t_span=100, dt=0.01, component=0
                )
                
                # Display results
                results = f"Original Message: {message}\n"
                results += f"Message Length: {len(message)} characters\n"
                results += f"Binary Length: {len(original_bits)} bits\n\n"
                results += f"System: {self.system_var.get()}\n"
                results += f"Initial Conditions: [{self.init_x.get()}, {self.init_y.get()}, {self.init_z.get()}]\n\n"
                results += f"Original Binary (first 80 bits):\n{original_bits[:80]}...\n\n"
                results += f"Keystream (first 80 bits):\n{self.keystream[:80]}...\n\n"
                results += f"Encoded Binary (first 80 bits):\n{self.encoded_bits[:80]}...\n\n"
                results += f"✓ Message successfully encoded!\n"
                results += f"Total encoded bits: {len(self.encoded_bits)}"
                
                self.results_text.insert(1.0, results)
                self.decode_btn.config(state=tk.NORMAL)
                self.status_var.set("Encoding complete!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Encoding failed: {str(e)}")
                self.status_var.set("Encoding failed!")
            finally:
                self.encode_btn.config(state=tk.NORMAL)
        
        thread = threading.Thread(target=encode_thread)
        thread.start()
    
    def decode_message(self):
        """Decode the message"""
        if self.encoded_bits is None or self.keystream is None:
            messagebox.showwarning("Warning", "Please encode a message first.")
            return
        
        self.status_var.set("Decoding...")
        
        try:
            encoder = ChaoticEncoder(self.current_system)
            decoded_message = encoder.decode(self.encoded_bits, self.keystream)
            
            self.results_text.insert(tk.END, f"\n\n{'=' * 60}\n")
            self.results_text.insert(tk.END, f"DECODED MESSAGE:\n")
            self.results_text.insert(tk.END, f"{'=' * 60}\n")
            self.results_text.insert(tk.END, f"{decoded_message}\n")
            self.results_text.insert(tk.END, f"{'=' * 60}\n")
            self.results_text.insert(tk.END, f"✓ Decoding successful!\n")
            
            self.status_var.set("Decoding complete!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Decoding failed: {str(e)}")
            self.status_var.set("Decoding failed!")
    
    def visualize_attractor(self):
        """Visualize the 3D attractor"""
        self.status_var.set("Generating visualization...")
        
        def visualize_thread():
            try:
                system = self.get_system()
                system.simulate(50, 0.01)
                
                # Create new window for visualization
                viz_window = tk.Toplevel(self.root)
                viz_window.title(f"{self.system_var.get()} Attractor")
                viz_window.geometry("800x600")
                
                fig = Figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                
                traj = system.trajectory
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=0.5, color='blue')
                ax.set_xlabel('X', fontsize=10)
                ax.set_ylabel('Y', fontsize=10)
                ax.set_zlabel('Z', fontsize=10)
                ax.set_title(f'{self.system_var.get()} Attractor', fontsize=12, fontweight='bold')
                
                canvas = FigureCanvasTkAgg(fig, master=viz_window)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                self.status_var.set("Visualization complete!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Visualization failed: {str(e)}")
                self.status_var.set("Visualization failed!")
        
        thread = threading.Thread(target=visualize_thread)
        thread.start()
    
    def clear_all(self):
        """Clear all fields"""
        self.results_text.delete(1.0, tk.END)
        self.encoded_bits = None
        self.keystream = None
        self.decode_btn.config(state=tk.DISABLED)
        self.status_var.set("Cleared!")

def main():
    root = tk.Tk()
    app = ChaoticCryptoGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()