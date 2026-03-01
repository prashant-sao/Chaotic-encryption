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

    def generate_keystream(self, length, component=0, weights=(1.234, 2.345, 0.987)):
        """
        Generates a binary keystream of 'length' bits.
        'weights' allows specifying different linear combinations of (x,y,z) 
        for RGB channels (e.g., use different weights for Red, Green, Blue).
        """
        if self.system.trajectory is None:
            raise ValueError("System must be simulated first (call simulate()).")
        
        vals = self.system.trajectory[:, component]
        L = len(vals)
        if L == 0:
            raise ValueError("Empty trajectory.")
            
        # Mix components using the provided weights
        # Default weights match the original implementation
        comp_mix = np.abs(self.system.trajectory[:,0] * weights[0] + 
                          self.system.trajectory[:,1] * weights[1] + 
                          self.system.trajectory[:,2] * weights[2])
                          
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
    def load_image_rgb(path_or_bytes):
        # Accept either file path or raw bytes
        if isinstance(path_or_bytes, (bytes, bytearray)):
            img = Image.open(io.BytesIO(path_or_bytes)).convert('RGB')
        else:
            img = Image.open(path_or_bytes).convert('RGB')
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
        # 1. Simulate Chaos
        t, traj = self.system.simulate(t_span, dt)
        
        # 2. Validate Input is RGB
        if len(img_arr.shape) != 3 or img_arr.shape[2] != 3:
            raise ValueError("Image must be RGB (3 channels).")
            
        height, width, _ = img_arr.shape
        shape = img_arr.shape # (H, W, 3)
        
        # 3. Split Channels
        # Flatten each channel individually to prepare for stream encryption
        flat_R = img_arr[:,:,0].flatten()
        flat_G = img_arr[:,:,1].flatten()
        flat_B = img_arr[:,:,2].flatten()
        
        # 4. Convert to bits
        # Helper returns bits and shape (we ignore shape here)
        bits_R, _ = self.image_to_bits(flat_R)
        bits_G, _ = self.image_to_bits(flat_G)
        bits_B, _ = self.image_to_bits(flat_B)
        
        # 5. Generate 3 Unique Keystreams
        # Using cyclic permutations of the weights for R, G, B
        w_R = (1.234, 2.345, 0.987)
        w_G = (0.987, 1.234, 2.345)
        w_B = (2.345, 0.987, 1.234)
        
        ks_R = self.generate_keystream(len(bits_R), weights=w_R)
        ks_G = self.generate_keystream(len(bits_G), weights=w_G)
        ks_B = self.generate_keystream(len(bits_B), weights=w_B)
        
        # 6. Encrypt (XOR) - Diffusion Phase
        def xor_bits(b, k): return ''.join(str(int(x)^int(y)) for x,y in zip(b, k))
        
        enc_bits_R = xor_bits(bits_R, ks_R)
        enc_bits_G = xor_bits(bits_G, ks_G)
        enc_bits_B = xor_bits(bits_B, ks_B)
        
        # 7. Reconstruct "Encrypted but not Shuffled" Image
        # We need to turn bits back into pixel values to stack them correctly for the global shuffle
        # We use a temporary shape (H, W) for the single channel reconstruction
        enc_R_arr = self.bits_to_image(enc_bits_R, (height, width))
        enc_G_arr = self.bits_to_image(enc_bits_G, (height, width))
        enc_B_arr = self.bits_to_image(enc_bits_B, (height, width))
        
        # Stack back to (H, W, 3) and then flatten globally
        # This ensures the R, G, B pixels are interleaved naturally (R,G,B, R,G,B...)
        enc_img_combined = np.dstack((enc_R_arr, enc_G_arr, enc_B_arr))
        flat_enc_pixels = enc_img_combined.flatten()
        
        # 8. Global Shuffle (Permutation)
        # Use the chaotic trajectory to generate a permutation for the WHOLE image size
        order = self.permutation_indices(flat_enc_pixels.size)
        permuted_pixels = flat_enc_pixels[order]
        
        # 9. Final Conversion to Bits for Transmission
        final_encrypted_bits = ''.join(format(px, '08b') for px in permuted_pixels)
        
        # Return all artifacts (ks is now a list of 3)
        return final_encrypted_bits, [ks_R, ks_G, ks_B], order.tolist(), shape, t, traj

    def decode_image(self, encrypted_bits, keystreams, order_list, shape):
        """
        Decodes an RGB image. 
        Note: 'keystreams' must be a list of 3 strings [ks_R, ks_G, ks_B].
        """
        # 1. Convert transmitted bits back to PERMUTED pixels (integers)
        # This gives us the scrambled, flattened pixel array
        perm_pixels = [int(encrypted_bits[i:i+8], 2) for i in range(0, len(encrypted_bits), 8)]
        perm_pixels = np.array(perm_pixels, dtype=np.uint8)

        # 2. Un-shuffle (Reverse Permutation)
        # We restore the pixels to their original positions, but they are still XOR-encrypted
        order = np.array(order_list, dtype=int)
        ordered_pixels = np.zeros_like(perm_pixels)
        ordered_pixels[order] = perm_pixels

        # Reshape to (H, W, 3) temporarily so we can split the channels
        ordered_img = ordered_pixels.reshape(shape)

        # 3. Split Channels
        flat_enc_R = ordered_img[:,:,0].flatten()
        flat_enc_G = ordered_img[:,:,1].flatten()
        flat_enc_B = ordered_img[:,:,2].flatten()

        # 4. Encrypted Pixels -> Encrypted Bits
        # We need bits to perform the XOR operation
        bits_enc_R, _ = self.image_to_bits(flat_enc_R)
        bits_enc_G, _ = self.image_to_bits(flat_enc_G)
        bits_enc_B, _ = self.image_to_bits(flat_enc_B)

        # 5. Decrypt (XOR) - Reverse Diffusion
        # We expect 'keystreams' to contain [ks_R, ks_G, ks_B]
        def xor_bits(b, k): return ''.join(str(int(x)^int(y)) for x,y in zip(b, k))

        dec_bits_R = xor_bits(bits_enc_R, keystreams[0])
        dec_bits_G = xor_bits(bits_enc_G, keystreams[1])
        dec_bits_B = xor_bits(bits_enc_B, keystreams[2])

        # 6. Reconstruct Final RGB Image
        height, width = shape[0], shape[1]
        dec_R_arr = self.bits_to_image(dec_bits_R, (height, width))
        dec_G_arr = self.bits_to_image(dec_bits_G, (height, width))
        dec_B_arr = self.bits_to_image(dec_bits_B, (height, width))

        dec_img = np.dstack((dec_R_arr, dec_G_arr, dec_B_arr))
        return dec_img
