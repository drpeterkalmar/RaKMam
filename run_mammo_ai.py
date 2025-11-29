import os
import sys
import torch
import pydicom
import numpy as np
import cv2
import functools
import types
import gc
import csv
from transformers import AutoModel
from pydicom.pixel_data_handlers.util import apply_voi_lut

# --- SETUP ---
try:
    import pylibjpeg
    pydicom.config.pixel_data_handlers = ['pydicom.pixel_data_handlers.pylibjpeg_handler'] + pydicom.config.pixel_data_handlers
except ImportError:
    pass

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'ianpan_mammoscreen')

# Globale Speicher
activations = {}
gradients = {}

def forward_hook_fn(name, module, input, output):
    global activations
    activations[name] = output

def backward_hook_fn(name, module, grad_in, grad_out):
    global gradients
    gradients[name] = grad_out[0]

def register_hooks_everywhere(model):
    hooks_count = 0
    for name, module in model.named_modules():
        if "conv_head" in name and isinstance(module, torch.nn.Conv2d):
            module.register_forward_hook(functools.partial(forward_hook_fn, name))
            module.register_full_backward_hook(functools.partial(backward_hook_fn, name))
            hooks_count += 1
    if hooks_count == 0:
        last_module = None
        last_name = None
        for name, module in model.named_modules():
             if isinstance(module, torch.nn.Conv2d):
                 last_name = name; last_module = module
        if last_module:
            last_module.register_forward_hook(functools.partial(forward_hook_fn, last_name))
            last_module.register_full_backward_hook(functools.partial(backward_hook_fn, last_name))

def patch_efficientnet_forward(model):
    def new_forward(self, x):
        param = next(self.parameters(), None)
        if param is not None and x.device != param.device:
            x = x.to(param.device)
        return self.original_forward(x)

    for module in model.modules():
        if module.__class__.__name__ == 'EfficientNet':
            if not hasattr(module, 'original_forward'):
                module.original_forward = module.forward
                module.forward = types.MethodType(new_forward, module)

# --- INTELLIGENTER DATEI-CHECK ---
def is_potential_dicom(path):
    """
    Entscheidet, ob wir VERSUCHEN sollten, die Datei zu laden.
    """
    filename = os.path.basename(path).lower()
    
    # 1. Ausschlussliste (Klar kein Dicom)
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.csv', '.txt', '.py', '.json', '.md', '.exe', '.bat')):
        return False

    # 2. Bekannte DICOM-Endungen (inkl. .dicom!)
    if filename.endswith(('.dcm', '.dicom', '.ima')):
        return True

    # 3. Datei ohne Endung oder unbekannt? -> Magic Bytes checken
    try:
        if os.path.getsize(path) < 132: return False
        with open(path, 'rb') as f:
            f.seek(128)
            header = f.read(4)
            return header == b'DICM'
    except:
        return False

def load_dicom_as_numpy(path):
    # Hier fangen wir jetzt Fehler ab und geben sie zurück
    try:
        # force=True ist wichtig für Dicoms ohne Preamble (oft bei .dicom Files ohne Header)
        try:
            ds = pydicom.dcmread(path, force=True)
        except Exception as e:
            return None, None, f"dcmread failed: {str(e)}"

        # Check: Ist überhaupt ein Bild drin?
        if not hasattr(ds, "pixel_array"):
             return None, None, "Keine Pixel-Daten (pixel_array fehlt)"

        if 'VOILUTSequence' in ds or 'WindowCenter' in ds:
            try: img = apply_voi_lut(ds.pixel_array, ds)
            except: img = ds.pixel_array
        else:
            img = ds.pixel_array
        
        # Tomosynthese / 3D Check
        if img.ndim == 3:
            if img.shape[0] > 4: 
                print(f"   (Info: Erstelle MIP aus 3D-Stapel...)")
                img = np.max(img, axis=0)
            elif img.shape[2] == 3: 
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if hasattr(ds, 'PhotometricInterpretation') and ds.PhotometricInterpretation == 'MONOCHROME1':
            img = img.max() - img
            
        img = img.astype(np.float32)
        if (img.max() - img.min()) != 0:
            img = (img - img.min()) / (img.max() - img.min()) * 255.0
        img = img.astype(np.uint8)
        
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
        # PatientID Fallback
        pid = ds.PatientID if hasattr(ds, 'PatientID') else "Unknown_ID"
        
        return img, pid, None # Kein Fehler
        
    except Exception as e:
        return None, None, f"Processing Error: {str(e)}"

def generate_side_by_side(score_tensor, original_img):
    global activations, gradients
    valid_layer = None
    for name in activations:
        if name in gradients and gradients[name] is not None:
            valid_layer = name
            break
    if valid_layer is None: return None

    act = activations[valid_layer][0].detach().cpu().float()
    grads = gradients[valid_layer][0].detach().cpu().float()
    
    pooled_grads = torch.mean(grads, dim=[1, 2])
    for i in range(pooled_grads.shape[0]):
        act[i, :, :] *= pooled_grads[i]
        
    heatmap = torch.mean(act, dim=0).numpy()
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0: heatmap /= np.max(heatmap)
    
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Cleaning
    mask = original_img < 25
    heatmap[mask] = 0
    
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)
    
    separator = np.ones((original_img.shape[0], 10, 3), dtype=np.uint8) * 255
    combined = np.hstack((original_bgr, separator, overlay))
    
    return combined

def main(start_folder):
    if not torch.cuda.is_available():
        print("WARNUNG: Keine NVIDIA GPU. Laufe auf CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"HARDWARE: {torch.cuda.get_device_name(0)} (Mixed Precision)")
    
    try:
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
        model.to(device)
        model.eval() 
        patch_efficientnet_forward(model)
        register_hooks_everywhere(model)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return

    abs_folder = os.path.abspath(start_folder)
    print(f"Starte Analyse in: {abs_folder}")
    results_list = []

    for root, dirs, files in os.walk(start_folder):
        
        # --- PRE-FILTERUNG ---
        # Wir schauen uns JEDE Datei an und prüfen, ob sie wie ein DICOM aussieht
        candidates = []
        for f in files:
            full_path = os.path.join(root, f)
            if is_potential_dicom(full_path):
                candidates.append(f)
        
        if candidates:
            print(f"\nVerarbeite Ordner: {os.path.basename(root)}")
            
            for f in candidates:
                full_path = os.path.join(root, f)
                img_array, pid, error_msg = load_dicom_as_numpy(full_path)
                
                # FEHLER-REPORTING
                if img_array is None:
                    print(f"   -> ÜBERSPRUNGEN: {f}")
                    print(f"      Grund: {error_msg}")
                    continue
                
                # Normaler Ablauf
                try:
                    global activations, gradients
                    activations = {}
                    gradients = {}
                    gc.collect()
                    if device.type == 'cuda': torch.cuda.empty_cache()

                    for param in model.parameters(): param.requires_grad = True
                    input_dict = {"image": img_array}
                    
                    if device.type == 'cuda':
                        with torch.amp.autocast('cuda'):
                            output = model(input_dict)
                            score_tensor = output['cancer'] if isinstance(output, dict) and 'cancer' in output else (output['logits'] if isinstance(output, dict) else output)
                            if isinstance(score_tensor, (list, tuple)): score_tensor = score_tensor[0]
                    else:
                        output = model(input_dict)
                        score_tensor = output['cancer'] if isinstance(output, dict) and 'cancer' in output else (output['logits'] if isinstance(output, dict) else output)
                        if isinstance(score_tensor, (list, tuple)): score_tensor = score_tensor[0]

                    prob = torch.sigmoid(score_tensor.float()).item()
                    print(f"   -> {f}: {prob:.4f}")
                    
                    subfolder_name = os.path.basename(root)
                    results_list.append([subfolder_name, pid, f, f"{prob:.4f}"])

                    model.zero_grad()
                    score_tensor.backward(retain_graph=False)
                    
                    combined_img = generate_side_by_side(score_tensor, img_array)
                    
                    if combined_img is not None:
                        text = f"Cancer Score: {prob:.4f}"
                        cv2.putText(combined_img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        
                        save_name = f"{f}_result.jpg"
                        save_path = os.path.join(root, save_name)
                        cv2.imwrite(save_path, combined_img)
                        print(f"      [Bild generiert]")
                        
                    del score_tensor, output, input_dict

                except Exception as e:
                    print(f"Fehler bei {f}: {e}")
                    if device.type == 'cuda': torch.cuda.empty_cache()

    if results_list:
        csv_path = os.path.join(start_folder, "report.csv")
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(["Ordner", "Patient ID", "Dateiname", "Wahrscheinlichkeit"])
            writer.writerows(results_list)
        print(f"\nFERTIG! Bericht gespeichert unter: {csv_path}")
    else:
        print("\nKeine verarbeitbaren DICOM-Dateien gefunden.")

if __name__ == "__main__":
    target_folder = sys.argv[1] if len(sys.argv) > 1 else "."
    main(target_folder)