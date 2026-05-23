# Imports
import numpy as np
import cv2
import serial
import time
import struct
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Setting up communication b/w Laptop and Microcontroller (Arduino Mega)
SERIAL_PORT = 'COM11'
BAUD = 115200

# Defining the Region of Interest To process only the part of the beam and not the whole frame
ROI_X = 50
ROI_Y = 290
ROI_W = 500
ROI_H = 30

X = ROI_W // 2
ball_visible = False

# Servo Horizontal Ref. and Limits
DC = 93
MAX_ANGLE = 6
REF_POS = ROI_X + (ROI_W // 2)

# PID Params.
pid_pos_filtered = float(REF_POS)
pid_err_prev = 0.0
pid_integral = 0.0
servo_angle = DC
FIXED_DT = 0.033

# Real time Tracker
REST_TIMEOUT = 3.0
MIN_ITER_DURATION = 1.0

class IterationTracker:
    def __init__(self):
        self.all_iterations = []
        self.iter_number = 0
        self.reset()
        
    def reset(self):
        self.active = False
        self.start_time = None
        self.times = []
        self.positions = []
        self.rest_timer = 0.0
        self.initial_pos = None
        self.crossed_centre = False
        self.peak_overshoot = 0.0
        self.settled = False
        self.settle_time = None

    def begin(self, pos, t):
        self.reset()
        self.active = True
        self.start_time = t
        self.initial_pos = pos
        self.times.append(0.0)
        self.positions.append(pos)
        
    def update(self, pos, t, in_deadband):
        if not self.active:
            return
        rel_t = t - self.start_time
        self.times.append(rel_t)
        self.positions.append(pos)
        err = pos - REF_POS
        
        if not self.crossed_centre and abs(err) < 20:
            self.crossed_centre = True
            
        if self.crossed_centre and not self.settled:
            if abs(err) > self.peak_overshoot:
                self.peak_overshoot = abs(err)
                
        if in_deadband:
            self.rest_timer += FIXED_DT
            if self.settle_time is None:
                self.settle_time = rel_t
        else:
            self.rest_timer = 0.0
            self.settle_time = None
            self.settled = False
            
        if self.rest_timer >= REST_TIMEOUT and not self.settled:
            self.settled = True
            
    def end(self, reason="unknown"):
        if not self.active:
            return None
        self.active = False
        duration = self.times[-1] if self.times else 0.0
        if duration < MIN_ITER_DURATION or len(self.positions) < 10:
            return None
        self.iter_number += 1
        
        result = {
            "number": self.iter_number,
            "reason": reason,
            "duration": duration,
            "times": list(self.times),
            "positions": list(self.positions),
            "metrics": self._compute_metrics()
        }
        self.all_iterations.append(result)
        self._print_metrics(result)
        return result

    def _compute_metrics(self):
        pos = np.array(self.positions)
        t = np.array(self.times)
        errs = pos - REF_POS
        tail = max(1, len(errs) // 5)
        
        ss_error = float(np.mean(errs[-tail:]))
        initial_err = float(errs[0]) if len(errs) > 0 else 0.0
        overshoot_pct = (self.peak_overshoot / abs(initial_err) * 100.0) if abs(initial_err) > 1 else 0.0
        
        target_63 = initial_err * (1.0 - 0.632)
        time_const = None
        for i, e in enumerate(errs):
            if (initial_err > 0 and e < target_63) or (initial_err < 0 and e > target_63):
                time_const = float(t[i])
                break
                
        settle_t = self.settle_time if self.settle_time is not None else float(t[-1])
        return {
            "initial_error_px": round(initial_err, 1),
            "ss_error_px": round(ss_error, 2),
            "overshoot_pct": round(overshoot_pct, 1),
            "time_constant_s": round(time_const, 3) if time_const else None,
            "settling_time_s": round(settle_t, 3),
        }

    def _print_metrics(self, r):
        m = r["metrics"]
        print(f"\n{'='*50}")
        print(f"Iteration #{r['number']} ({r['reason']})")
        print(f"Duration      : {r['duration']:.2f}s")
        print(f"Initial error : {m['initial_error_px']} px")
        print(f"Steady-state  : {m['ss_error_px']} px")
        print(f"Overshoot     : {m['overshoot_pct']:.1f} %")
        print(f"Time constant : {m['time_constant_s']}s")
        print(f"Settling time : {m['settling_time_s']}s")
        print(f"{'='*50}\n")

tracker = IterationTracker()

params = {
    "Kp": 0.06,
    "Ki": 0.04,
    "Kd": 0.028,
    "Alpha": 0.600,
    "Deadband": 5,
    "H_min": 0,
    "S_min": 0,
    "V_min": 120,
    "H_max": 179,
    "S_max": 80,
    "V_max": 255,
    "Min_r": 20,
    "Max_r": 30,
}

FIELDS = [
    ("Kp", "Kp", 0.000, 0.500, False, 1000),
    ("Ki", "Ki", 0.000, 0.200, False, 1000),
    ("Kd", "Kd", 0.000, 2.000, False, 1000),
    ("Alpha", "Alpha", 0.050, 1.000, False, 1000),
    ("Deadband", "Deadband", 0, 20, True, 1),
    ("H min", "H_min", 0, 179, True, 1),
    ("S min", "S_min", 0, 255, True, 1),
    ("V min", "V_min", 0, 255, True, 1),
    ("H max", "H_max", 0, 179, True, 1),
    ("S max", "S_max", 0, 255, True, 1),
    ("V max", "V_max", 0, 255, True, 1),
    ("Min radius", "Min_r", 1, 60, True, 1),
    ("Max radius", "Max_r", 10, 200, True, 1),
]

tk_vars = {}
tk_sliders = {}
tk_entries = {}
_lock = {}

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def apply_value(key, actual, source):
    _, _, lo, hi, is_int, scale = next(f for f in FIELDS if f[1] == key)
    actual = clamp(actual, lo, hi)
    params[key] = actual
    if source != "tk_slider" and key in tk_vars:
        tk_vars[key].set(actual * scale if not is_int else actual)
    if source != "tk_entry" and key in tk_entries:
        tk_entries[key].delete(0, tk.END)
        tk_entries[key].insert(0, f"{actual:.4f}" if not is_int else str(int(actual)))
        tk_entries[key].config(bg="#2d2d2d")

def on_tk_slider(key, val):
    if _lock.get(key): return
    _lock[key] = True
    _, _, lo, hi, is_int, scale = next(f for f in FIELDS if f[1] == key)
    actual = float(val) / scale if not is_int else int(float(val))
    apply_value(key, actual, "tk_slider")
    _lock[key] = False

def on_tk_entry(key, widget):
    if _lock.get(key): return
    _lock[key] = True
    _, _, lo, hi, is_int, scale = next(f for f in FIELDS if f[1] == key)
    try:
        val = int(widget.get()) if is_int else float(widget.get())
        apply_value(key, val, "tk_entry")
        widget.config(bg="#2d2d2d")
    except ValueError:
        widget.config(bg="#6b2d2d")
    _lock[key] = False

# Connecting to Hardware
print("[System] Connecting to Arduino...")
try:
    Arduino = serial.Serial(SERIAL_PORT, baudrate=BAUD, timeout=0)
    time.sleep(2)
    Arduino.reset_input_buffer()
    Arduino.reset_output_buffer()
    print("[Serial] Connected.")
except Exception as e:
    print(f"[Warning] Arduino not connected: {e}")
    Arduino = None

print("[System] Starting camera...")
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

# 3 column windows for tuning and metrics
root = tk.Tk()
root.title("Ball & Beam Control Panel")
root.configure(bg="#1e1e1e")

style = ttk.Style()
style.theme_use("clam")
style.configure("TScale", background="#1e1e1e", troughcolor="#3a3a3a", sliderlength=14)

# COL 0: Parameter tuner
col0 = tk.Frame(root, bg="#1e1e1e")
col0.grid(row=0, column=0, sticky="ns", padx=(8, 4), pady=8)

tk.Label(col0, text="Parameter Tuner", bg="#1e1e1e", fg="#ffffff", font=("Helvetica", 11, "bold")).grid(row=0, column=0, columnspan=4, pady=(4, 2))
tk.Label(col0, text="Enter value press Enter to apply", bg="#1e1e1e", fg="#555555", font=("Helvetica", 8)).grid(row=1, column=0, columnspan=4, pady=(0, 6))

for ci, h in enumerate(["Param", "Slider", "Value", "Range"]):
    tk.Label(col0, text=h, bg="#1e1e1e", fg="#666666", font=("Helvetica", 8)).grid(row=2, column=ci, padx=4)

SECTIONS = {"Kp": "PID Gains", "H_min": "HSV Detection", "Min_r": "Ball Size Filter"}
prow = 3

for label, key, lo, hi, is_int, scale in FIELDS:
    if key in SECTIONS:
        tk.Label(col0, text=f"--- {SECTIONS[key]} ---", bg="#1e1e1e", fg="#888888", font=("Helvetica", 8, "italic")).grid(row=prow, column=0, columnspan=4, sticky="w", padx=8, pady=(6, 1))
        prow += 1
        
    _lock[key] = False
    init_val = params[key]
    sv = init_val * scale if not is_int else init_val
    
    tk.Label(col0, text=label, bg="#1e1e1e", fg="#cccccc", font=("Helvetica", 9), width=10, anchor="w").grid(row=prow, column=0, padx=(8, 2), pady=1, sticky="w")
    
    var = tk.DoubleVar(value=sv)
    tk_vars[key] = var
    
    sl = ttk.Scale(col0, from_=lo*scale if not is_int else lo, to=hi*scale if not is_int else hi, orient="horizontal", variable=var, length=160, command=lambda v, k=key: on_tk_slider(k, v))
    sl.grid(row=prow, column=1, padx=4, pady=1)
    tk_sliders[key] = sl
    
    ent = tk.Entry(col0, width=8, font=("Courier", 9), bg="#2d2d2d", fg="#ffffff", insertbackground="white", relief="flat", bd=3)
    ent.insert(0, f"{init_val:.4f}" if not is_int else str(int(init_val)))
    ent.grid(row=prow, column=2, padx=4, pady=1)
    ent.bind("<Return>", lambda e, k=key, w=ent: on_tk_entry(k, w))
    ent.bind("<FocusOut>", lambda e, k=key, w=ent: on_tk_entry(k, w))
    tk_entries[key] = ent
    
    rtxt = f"{lo}-{hi}" if is_int else f"{lo:.2f}-{hi:.2f}"
    tk.Label(col0, text=rtxt, bg="#1e1e1e", fg="#444444", font=("Helvetica", 7), width=8).grid(row=prow, column=3, padx=2)
    prow += 1

tk.Button(col0, text="Force New Iteration", bg="#2d4a6b", fg="white", font=("Helvetica", 9, "bold"), command=lambda: [tracker.end("manual") if tracker.active else tracker.reset()]).grid(row=prow, column=0, columnspan=4, pady=(10, 4), padx=8, sticky="we")

# COL 1: Camera Feed & Tracking Masks
col1 = tk.Frame(root, bg="#1e1e1e")
col1.grid(row=0, column=1, sticky="ns", padx=4, pady=8)

tk.Label(col1, text="Camera Feed", bg="#1e1e1e", fg="#aaaaaa", font=("Helvetica", 9)).pack(pady=(0, 2))
cam_label = tk.Label(col1, bg="#000000")
cam_label.pack()

tk.Label(col1, text="ROI Tracking", bg="#1e1e1e", fg="#aaaaaa", font=("Helvetica", 9)).pack(pady=(6, 2))
roi_label = tk.Label(col1, bg="#000000")
roi_label.pack()

tk.Label(col1, text="HSV Mask", bg="#1e1e1e", fg="#aaaaaa", font=("Helvetica", 9)).pack(pady=(6, 2))
mask_label = tk.Label(col1, bg="#000000")
mask_label.pack()

tele = tk.Frame(col1, bg="#1e1e1e")
tele.pack(fill=tk.X, pady=(8, 0))
lbl_err = tk.Label(tele, text="Err: 0 px", bg="#1e1e1e", fg="#ff4444", font=("Courier", 10, "bold"))
lbl_err.pack(side=tk.LEFT, padx=6)
lbl_ang = tk.Label(tele, text="Servo: 93", bg="#1e1e1e", fg="#00aaff", font=("Courier", 10, "bold"))
lbl_ang.pack(side=tk.LEFT)
lbl_iter = tk.Label(tele, text="", bg="#1e1e1e", fg="#888888", font=("Courier", 9))
lbl_iter.pack(side=tk.RIGHT, padx=6)

def show_frame_in_label(label, bgr_img, max_w, max_h):
    h, w = bgr_img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        bgr_img = cv2.resize(bgr_img, (int(w*scale), int(h*scale)))
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    photo = ImageTk.PhotoImage(pil)
    label.config(image=photo)
    label.image = photo 

def show_mask_in_label(label, gray_img, max_w, max_h):
    h, w = gray_img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        gray_img = cv2.resize(gray_img, (int(w*scale), int(h*scale)))
    pil = Image.fromarray(gray_img)
    photo = ImageTk.PhotoImage(pil)
    label.config(image=photo)
    label.image = photo

# COL 2: Performance Plot & Metrics
col2 = tk.Frame(root, bg="#1e1e1e")
col2.grid(row=0, column=2, sticky="nsew", padx=(4, 8), pady=8)
root.columnconfigure(2, weight=1)

mf = tk.Frame(col2, bg="#252525")
mf.pack(fill=tk.X, pady=(0, 6))
tk.Label(mf, text="Last Iteration Metrics", bg="#252525", fg="#aaaaaa", font=("Helvetica", 9, "bold")).grid(row=0, column=0, columnspan=12, pady=(4, 2), padx=8)

metric_labels = {}
metric_defs = [
    ("Iter #", "number"),
    ("Init err", "initial_error_px"),
    ("SS error", "ss_error_px"),
    ("Overshoot", "overshoot_pct"),
    ("Time const", "time_constant_s"),
    ("Settle time", "settling_time_s")
]

for ci, (disp, key) in enumerate(metric_defs):
    tk.Label(mf, text=disp+":", bg="#252525", fg="#666666", font=("Helvetica", 8)).grid(row=1, column=ci*2, padx=(8, 1), pady=3, sticky="e")
    lbl = tk.Label(mf, text=" ", bg="#252525", fg="#ffffff", font=("Courier", 8, "bold"), width=7, anchor="w")
    lbl.grid(row=1, column=ci*2+1, padx=(1, 6), pady=3, sticky="w")
    metric_labels[key] = lbl

def update_metrics_display(result):
    m = result["metrics"]
    metric_labels["number"].config(text=str(result["number"]))
    metric_labels["initial_error_px"].config(text=f"{m['initial_error_px']}px")
    metric_labels["ss_error_px"].config(text=f"{m['ss_error_px']}px")
    metric_labels["overshoot_pct"].config(text=f"{m['overshoot_pct']:.1f}%", fg="#ff6666" if m["overshoot_pct"] > 20 else "#66ff66")
    metric_labels["time_constant_s"].config(text=f"{m['time_constant_s']}s" if m["time_constant_s"] else "N/A")
    metric_labels["settling_time_s"].config(text=f"{m['settling_time_s']}s")

fig = Figure(figsize=(5, 3.5), dpi=96, facecolor="#1e1e1e")
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.14)
ax.set_facecolor("#141414")
ax.tick_params(colors="#666666", labelsize=7)
ax.set_xlabel("Time (s)", color="#666666", fontsize=8)
ax.set_ylabel("Displacement (px)", color="#666666", fontsize=8)
ax.set_title("Ball Position Error vs Time", color="#cccccc", fontsize=9)
ax.axhline(0, color="#444444", linestyle="--", linewidth=1)

db_hi, = ax.plot([0, 1], [params["Deadband"], params["Deadband"]], color="#336633", linestyle=":", linewidth=0.8)
db_lo, = ax.plot([0, 1], [-params["Deadband"], -params["Deadband"]], color="#336633", linestyle=":", linewidth=0.8)

for sp in ax.spines.values():
    sp.set_edgecolor("#333333")
    
line_cur, = ax.plot([], [], color="#00aaff", linewidth=1.8, label="Current")
line_prev, = ax.plot([], [], color="#555555", linewidth=1.0, linestyle="--", label="Previous")
ax.legend(loc="upper right", facecolor="#252525", edgecolor="none", labelcolor="white", fontsize=7)

graph_canvas = FigureCanvasTkAgg(fig, master=col2)
graph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def update_graph():
    db_hi.set_ydata([params["Deadband"], params["Deadband"]])
    db_lo.set_ydata([-params["Deadband"], -params["Deadband"]])
    
    if tracker.active and len(tracker.times) > 1:
        t_a = np.array(tracker.times)
        err_a = np.array(tracker.positions) - REF_POS
        line_cur.set_data(t_a, err_a)
        db_hi.set_xdata([0, t_a[-1]+0.5])
        db_lo.set_xdata([0, t_a[-1]+0.5])
        ax.set_xlim(0, max(5.0, t_a[-1]+0.5))
        ax.set_ylim(-max(50, np.max(np.abs(err_a))*1.2), max(50, np.max(np.abs(err_a))*1.2))
        
    elif not tracker.active and tracker.all_iterations:
        last = tracker.all_iterations[-1]
        t_a = np.array(last["times"])
        err_a = np.array(last["positions"]) - REF_POS
        line_cur.set_data([], [])
        line_prev.set_data(t_a, err_a)
        db_hi.set_xdata([0, t_a[-1]+0.5])
        db_lo.set_xdata([0, t_a[-1]+0.5])
        ax.set_xlim(0, max(5.0, t_a[-1]+0.5))
        ax.set_ylim(-max(50, np.max(np.abs(err_a))*1.2), max(50, np.max(np.abs(err_a))*1.2))
        
    graph_canvas.draw_idle()
    root.after(100, update_graph)

def on_closing():
    print("[System] Shutting down...")
    if tracker.active:
        tracker.end("shutdown")
    cap.release()
    if Arduino and Arduino.is_open:
        Arduino.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

was_visible = False

def process_frame():
    global x, ball_visible, servo_angle, was_visible
    global pid_pos_filtered, pid_err_prev, pid_integral
    
    ret, frame = cap.read()
    if not ret:
        root.after(10, process_frame)
        return
        
    roi = frame[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W].copy()
    if roi.size == 0:
        root.after(10, process_frame)
        return
        
    lower = np.array([params["H_min"], params["S_min"], params["V_min"]])
    upper = np.array([params["H_max"], params["S_max"], params["V_max"]])
    
    blurred = cv2.GaussianBlur(roi, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_visible = False
    x_roi = ROI_W // 2
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        center, radius = cv2.minEnclosingCircle(c)
        if params["Min_r"] < radius < params["Max_r"]:
            M = cv2.moments(c)
            if M["m00"] != 0:
                x_roi = int(M["m10"] / M["m00"])
                ball_visible = True
                cv2.circle(roi, (x_roi, int(center[1])), 5, (0, 0, 255), -1)
                cv2.circle(roi, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 2)
                cv2.putText(roi, f"r={int(radius)}", (x_roi+6, int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    x = x_roi + ROI_X
    now = time.time()
    in_deadband = abs(x - REF_POS) <= params["Deadband"]
    
    # Iteration Tracking Framework
    if ball_visible and not was_visible:
        if tracker.active:
            tracker.end("ball lost and returned")
        tracker.begin(x, now)
    elif not ball_visible and was_visible:
        if tracker.active:
            result = tracker.end("ball left frame")
            if result:
                update_metrics_display(result)
                line_prev.set_data(np.array(result["times"]), np.array(result["positions"]) - REF_POS)
    elif ball_visible and tracker.active:
        tracker.update(x, now, in_deadband)
        if tracker.rest_timer > REST_TIMEOUT and tracker.settled:
            result = tracker.end("settled at rest")
            if result:
                update_metrics_display(result)
    elif ball_visible and not tracker.active:
        if abs(x - REF_POS) > params["Deadband"] * 2:
            tracker.begin(x, now)
            
    was_visible = ball_visible
    
    # PID Math Pipeline
    tx_x = x
    if abs(tx_x - int(pid_pos_filtered)) > 50:
        pid_pos_filtered = float(tx_x)
        pid_err_prev = 0.0
    else:
        pid_pos_filtered = (params["Alpha"] * tx_x + (1.0 - params["Alpha"]) * pid_pos_filtered)
        
    err = REF_POS - pid_pos_filtered
    dt = FIXED_DT
    
    if abs(err) <= params["Deadband"]:
        active_err = 0.0
        D = 0.0
    else:
        active_err = err
        derivative = (err - pid_err_prev) / dt
        pid_err_prev = err
        D = derivative * params["Kd"]
        
    P = active_err * params["Kp"]
    pid_integral += active_err * dt
    int_limit = (MAX_ANGLE * 0.4) / max(params["Ki"], 0.001)
    pid_integral = max(-int_limit, min(int_limit, pid_integral))
    I = params["Ki"] * pid_integral
    
    output = P + I + D
    servo_angle = DC + int(output)
    servo_angle = max(DC - MAX_ANGLE, min(DC + MAX_ANGLE, servo_angle))
    
    # Serial Comm
    if Arduino:
        try:
            if Arduino.in_waiting > 0:
                data = Arduino.read(Arduino.in_waiting)
                if b'!' in data:
                    Arduino.write(struct.pack('<B', int(servo_angle)))
        except serial.SerialException as e:
            print(f"[Serial error] {e}")
            
    # Graphical Annotation overlays on full frame
    frame_disp = frame.copy()
    cv2.rectangle(frame_disp, (ROI_X, ROI_Y), (ROI_X+ROI_W, ROI_Y+ROI_H), (0, 255, 0), 2)
    beam_cx = ROI_X + (ROI_W // 2)
    cv2.line(frame_disp, (beam_cx, ROI_Y), (beam_cx, ROI_Y+ROI_H), (255, 255, 0), 1)
    
    sc = (0, 255, 0) if ball_visible else (0, 0, 255)
    stxt = "BALL OK" if ball_visible else "NO BALL"
    cv2.putText(frame_disp, stxt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, sc, 2)
    cv2.putText(frame_disp, f"Kp={params['Kp']:.4f} Ki={params['Ki']:.4f} Kd={params['Kd']:.4f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)
    cv2.putText(frame_disp, f"Alpha={params['Alpha']:.3f} DB={int(params['Deadband'])}px MaxR={int(params['Max_r'])}", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    
    # Render into TKinter frames
    show_frame_in_label(cam_label, frame_disp, max_w=480, max_h=240)
    show_frame_in_label(roi_label, roi, max_w=480, max_h=80)
    show_mask_in_label(mask_label, mask, max_w=480, max_h=80)
    
    # Telemetry Updates
    lbl_err.config(text=f"Err: {x - REF_POS: +4d} px")
    lbl_ang.config(text=f"Servo: {int(servo_angle):+3d}")
    lbl_iter.config(
        text=f"Iter #{tracker.iter_number+1} t={tracker.times[-1]:.1f}s" if tracker.active else f"Iter #{tracker.iter_number} done",
        fg="#00aaff" if tracker.active else "#888888"
    )
    root.after(10, process_frame)

# Running the system
print("[System] Running close window to quit")
root.after(10, process_frame)
root.after(100, update_graph)
root.mainloop()
