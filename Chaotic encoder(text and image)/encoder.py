import numpy as np
from PIL import Image
import io

class ChaoticEncoder:
    def __init__(self, system):
        self.system = system

    # ---------- Text helpers ----------
    def text_to_bits(self, text):
        return ''.join(format(ord(c), '08b') for c in text)

    def bits_to_text(self, bits):
        chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
        return ''.join(chr(int(c, 2)) for c in chars if len(c) == 8)

    def generate_keystream(self, length, component=0):
        if self.system.trajectory is None:
            raise ValueError("System must be simulated first (call simulate()).")
        vals = self.system.trajectory[:, component]
        L = len(vals)
        if L == 0:
            raise ValueError("Empty trajectory.")
        # Mix components for better randomness (if available)
        comp_mix = np.abs(self.system.trajectory[:,0] * 1.234 + self.system.trajectory[:,1] * 2.345 + self.system.trajectory[:,2] * 0.987)
        # Build keystream of bits by sampling using modular stepping
        keystream = []
        step = 13
        for i in range(length):
            idx = (i * step) % L
            val = comp_mix[idx]
            frac = abs(val) - int(abs(val))
            # Map fractional to one bit by parity of digits sum (deterministic)
            ds = sum(int(ch) for ch in f"{frac:.15f}" if ch.isdigit())
            bit = '1' if (ds % 2) == 1 else '0'
            keystream.append(bit)
        return ''.join(keystream)

    def encode_text(self, message, t_span=500.0, dt=0.005, lyap_eps=None, lyap_component=0):
        message_bits = self.text_to_bits(message)
        t, traj = self.system.simulate(t_span, dt)
        if lyap_eps is not None:
            sep, ln_sep, ftle = self.system.compute_lyapunov(t, traj, eps=lyap_eps, perturb_component=lyap_component)
            lyap_results = {'sep': sep, 'ln_sep': ln_sep, 'ftle': ftle, 't': t, 'eps': lyap_eps}
        else:
            lyap_results = None
        ks = self.generate_keystream(len(message_bits))
        encoded_bits = ''.join(str(int(m) ^ int(k)) for m, k in zip(message_bits, ks))
        return encoded_bits, message_bits, ks, t, traj, lyap_results

    def decode_text(self, encoded_bits, keystream):
        decoded_bits = ''.join(str(int(e) ^ int(k)) for e, k in zip(encoded_bits, keystream))
        decoded_message = self.bits_to_text(decoded_bits)
        return decoded_message, decoded_bits

    # ---------- Image helpers ----------
    @staticmethod
    def load_image_grayscale(path_or_bytes):
        # Accept either file path or raw bytes
        if isinstance(path_or_bytes, (bytes, bytearray)):
            img = Image.open(io.BytesIO(path_or_bytes)).convert('L')
        else:
            img = Image.open(path_or_bytes).convert('L')
        arr = np.array(img, dtype=np.uint8)
        return arr

    @staticmethod
    def image_to_bits(img_arr):
        flat = img_arr.flatten()
        bits = ''.join(format(px, '08b') for px in flat)
        return bits, img_arr.shape

    @staticmethod
    def bits_to_image(bits, shape):
        pixels = [int(bits[i:i+8], 2) for i in range(0, len(bits), 8)]
        arr = np.array(pixels, dtype=np.uint8).reshape(shape)
        return arr

    def permutation_indices(self, N):
        # Create a chaotic ordering using one trajectory component (absolute and argsort)
        if self.system.trajectory is None:
            raise ValueError("System must be simulated first.")
        vals = np.abs(self.system.trajectory[:, 0])
        if len(vals) < N:
            # repeat or tile the component until length >= N
            reps = int(np.ceil(N / len(vals)))
            vals = np.tile(vals, reps)[:N]
        else:
            vals = vals[:N]
        order = np.argsort(vals)
        return order

    def encode_image(self, img_arr, t_span=600.0, dt=0.002):
        # Flatten and convert to permutation + XOR diffusion
        bits, shape = self.image_to_bits(img_arr)
        num_pixels = img_arr.size

        t, traj = self.system.simulate(t_span, dt)

        order = self.permutation_indices(num_pixels)
        flat = img_arr.flatten()
        permuted = flat[order]

        perm_bits = ''.join(format(px, '08b') for px in permuted)
        ks = self.generate_keystream(len(perm_bits))
        encrypted_bits = ''.join(str(int(a) ^ int(b)) for a, b in zip(perm_bits, ks))

        # For transport, convert order to list
        return encrypted_bits, ks, order.tolist(), shape, t, traj

    def decode_image(self, encrypted_bits, keystream, order_list, shape):
        # reverse XOR
        perm_bits = ''.join(str(int(a) ^ int(b)) for a, b in zip(encrypted_bits, keystream))
        # bits -> pixels
        perm_pixels = [int(perm_bits[i:i+8], 2) for i in range(0, len(perm_bits), 8)]
        perm_pixels = np.array(perm_pixels, dtype=np.uint8)
        order = np.array(order_list, dtype=int)
        original = np.zeros_like(perm_pixels)
        original[order] = perm_pixels
        img = original.reshape(shape)
        return img
