from PIL import Image
import numpy as np

def chaos_encode_image(image_path):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    flat = arr.flatten()

    np.random.seed(987654321)
    keystream = np.random.randint(0, 256, size=len(flat), dtype=np.uint8)

    encoded = np.bitwise_xor(flat, keystream)

    return encoded.tobytes(), img.size, flat.shape  # Include original shape

def chaos_decode_image(encoded_bytes, image_size, original_shape, out_path):
    arr = np.frombuffer(encoded_bytes, dtype=np.uint8)

    np.random.seed(987654321)
    keystream = np.random.randint(0, 256, size=len(arr), dtype=np.uint8)

    decoded = np.bitwise_xor(arr, keystream)
    decoded = decoded.reshape(original_shape)  # Use original shape

    img = Image.fromarray(decoded.astype(np.uint8), 'RGB')
    img.save(out_path)