import tkinter as tk
from tkinter import ttk
import sender_gui, receiver_gui

def launch_sender():
    win = tk.Tk()
    sender_gui.SenderGUI(win)
    win.mainloop()

def launch_receiver():
    win = tk.Tk()
    receiver_gui.ReceiverGUI(win)
    win.mainloop()

def main_selector():
    root = tk.Tk(); root.title("Chaotic Cryptography — Launcher"); root.geometry("420x260")
    frm = ttk.Frame(root, padding=16); frm.pack(fill=tk.BOTH, expand=True)
    ttk.Label(frm, text="Chaotic Cryptography", font=('Arial',16,'bold')).pack(pady=(6,12))
    ttk.Button(frm, text="🔐 Sender (Encode & Send)", command=lambda: [root.destroy(), launch_sender()], width=36).pack(pady=8)
    ttk.Button(frm, text="🔓 Receiver (Receive & Decode)", command=lambda: [root.destroy(), launch_receiver()], width=36).pack(pady=8)
    root.mainloop()

if __name__ == "__main__":
    main_selector()
