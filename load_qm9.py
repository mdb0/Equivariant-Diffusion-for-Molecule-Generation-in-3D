"""
This file profide one function : qm9_fetch, this allow us to load the qm9 dataset.
"""
import numpy as np
import urllib
import tarfile
import os
import torch

def qm9_prepare_molecule(lines):
    pt = {"C": 6, "H": 1, "O": 8, "N": 7, "F": 9}
    N = int(lines[0])
    atoms = [line.split() for line in lines[2:N + 2]]
    elements = [pt[a[0]] for a in atoms]
    data = np.array([[float(x.replace('*^', 'e')) for x in a[1:]] for a in atoms])
    coords = data[:, :3]
    charges = data[:, 3]
    return {"N": N, "elements": elements, "coords": coords, "charges": charges}


def qm9_fetch(num_molecules = 133885):
    """
    Télécharge et prépare le dataset QM9.
    """
    RAW_FILE = "./data/QM9/qm9.tar.bz2"
    QM9_URL = "https://ndownloader.figshare.com/files/3195389"
    
    
    if not os.path.isfile(RAW_FILE): # Télécharger l'archive si nécessaire
        print("Téléchargement du dataset QM9...")
        urllib.request.urlretrieve(QM9_URL, RAW_FILE)
        print("Téléchargement terminé")
    else:
        print(f"Archive {RAW_FILE} existante")
    
    
    molecules = []
    with tarfile.open(RAW_FILE, "r:bz2") as tar:
        for i in range(1, num_molecules + 1):
            if i % 100 == 0:
                progress = (i / num_molecules) * 100
                print(f"\rProgression: {progress:.1f}%", end="", flush=True)
            
            # Extraire et parser chaque molécule
            filename = f"dsgdb9nsd_{i:06d}.xyz"
            
            try:
                with tar.extractfile(filename) as f:
                    lines = f.read().decode("utf-8").splitlines()
                    molecule = qm9_prepare_molecule(lines)
                    molecules.append(molecule)
                    
            except (ValueError, KeyError) as e:
                print(f"\nErreur lors du traitement de la molécule {i}: {e}")
                raise e
    
    print(f"\nExtraction terminée ! {len(molecules)} molécules chargées.")
    return molecules


#########################################################################################################################
# store QM9 as torch datset for faster reloading
#########################################################################################################################
def qm9_prepare_molecule_torch(lines):
    """Parse one molecule from QM9 .xyz lines."""
    pt = {"C": 6, "H": 1, "O": 8, "N": 7, "F": 9}
    N = int(lines[0])
    atoms = [line.split() for line in lines[2:N + 2]]
    elements = torch.tensor([pt[a[0]] for a in atoms], dtype=torch.int64)
    data = torch.tensor([[float(x.replace('*^', 'e')) for x in a[1:]] for a in atoms], dtype=torch.float64)
    coords = data[:, :3]
    charges = data[:, 3]
    return coords, elements, charges


def qm9_to_torch(num_molecules=None):
    save_path="./data/QM9/qm9_torch.pt"
    raw_file = "./data/QM9/qm9.tar.bz2"
    qm9_url = "https://ndownloader.figshare.com/files/3195389"

    # 1. Download if necessary
    if not os.path.isfile(raw_file):
        print("Downloading QM9 dataset (~350 MB)...")
        urllib.request.urlretrieve(qm9_url, raw_file)
        print("Download complete.")
    else:
        print("Using existing archive:", raw_file)

    # 2. Define generator that streams directly from archive
    mols_torch = []
    with tarfile.open(raw_file, "r:bz2") as tar:
        members = sorted(
            [m for m in tar.getnames() if m.startswith("dsgdb9nsd_") and m.endswith(".xyz")]
        )
        if num_molecules:
            members = members[:num_molecules]

        for i, name in enumerate(members, 1):
            if i % 100 == 0:
                print(f"\rParsed {i}/{len(members)} molecules", end="", flush=True)
            
            f = tar.extractfile(name)
            lines = f.read().decode("utf-8").splitlines()
            mols_torch += [qm9_prepare_molecule_torch(lines)]

    torch.save(mols_torch, save_path)
    print(f"\nSaved torch list to {save_path}")



def qm9_load_torch(save_path="./data/QM9/qm9_torch.pt"):
    """Reload previously saved QM9 TensorFlow dataset."""
    return torch.load(save_path)
