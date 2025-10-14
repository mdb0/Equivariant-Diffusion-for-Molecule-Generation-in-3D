"""
This file profide one function : qm9_fetch, this allow us to load the qm9 dataset.
"""
import numpy as np
import urllib
import tarfile
import os

def qm9_prepare_molecule(lines):
    pt = {"C": 6, "H": 1, "O": 8, "N": 7, "F": 9}
    
    N = int(lines[0]) # Nombre d'atomes
    
    elements = [pt[line.split()[0]] for line in lines[2:N + 2]]
    
    coords = np.empty((N, 3), dtype=np.float64)
    charges = np.empty(N, dtype=np.float64)
    
    for i in range(N):
        values = [float(x.replace('*^', 'e')) for x in lines[i + 2].split()[1:]]
        coords[i] = values[:3]  # x, y, z
        charges[i] = values[3]   # Charge de Mulliken
    
    return {
        "N": N,
        "elements": elements,
        "coords": coords,
        "charges": charges
    }


def qm9_fetch(num_molecules = 133885):
    """
    Télécharge et prépare le dataset QM9.
    """
    RAW_FILE = "qm9.tar.bz2"
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
                    lines = [line.decode("UTF-8") for line in f.readlines()]
                    molecule = qm9_prepare_molecule(lines)
                    molecules.append(molecule)
                    
            except (ValueError, KeyError) as e:
                print(f"\nErreur lors du traitement de la molécule {i}: {e}")
                raise e
    
    print(f"\nExtraction terminée ! {len(molecules)} molécules chargées.")
    return molecules