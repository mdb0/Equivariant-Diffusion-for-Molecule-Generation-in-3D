"""
This file profide one function : qm9_fetch, this allow us to load the qm9 dataset.
"""
import numpy as np
import urllib
import tarfile
import os
import tensorflow as tf

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
# store QM9 as tf datset for faster reloading
#########################################################################################################################
def qm9_prepare_molecule_tf(lines):
    """Parse one molecule from QM9 .xyz lines."""
    pt = {"C": 6, "H": 1, "O": 8, "N": 7, "F": 9}
    N = int(lines[0])
    atoms = [line.split() for line in lines[2:N + 2]]
    elements = [pt[a[0]] for a in atoms]
    data = np.array([[float(x.replace('*^', 'e')) for x in a[1:]] for a in atoms])
    coords = data[:, :3].astype(np.float32)
    charges = data[:, 3].astype(np.float32)
    return coords, np.array(elements, dtype=np.int64), charges


def qm9_to_tfdata(num_molecules=None):
    save_path="./data/QM9/qm9_tfdata"
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
    def gen():
        with tarfile.open(raw_file, "r:bz2") as tar:
            members = sorted(
                [m for m in tar.getnames() if m.startswith("dsgdb9nsd_") and m.endswith(".xyz")]
            )
            if num_molecules:
                members = members[:num_molecules]

            for i, name in enumerate(members, 1):
                if i % 100 == 0:
                    print(f"\rParsed {i}/{len(members)} molecules", end="", flush=True)
                try:
                    f = tar.extractfile(name)
                    if f is None:
                        continue
                    lines = f.read().decode("utf-8").splitlines()
                    coords, elements, charges = qm9_prepare_molecule_tf(lines)
                    yield coords, elements, charges
                except Exception as e:
                    print(f"\nSkipping molecule {name}: {e}")
                    continue

    # 3. Build TensorFlow dataset directly from the generator
    output_signature = (
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    # Optional: Save for later use
    tf.data.Dataset.save(ds, save_path)
    print(f"\nSaved TensorFlow dataset to {save_path}")
    return ds


def qm9_load_tfdata(save_path="./data/QM9/qm9_tfdata"):
    """Reload previously saved QM9 TensorFlow dataset."""
    return tf.data.Dataset.load(save_path)
