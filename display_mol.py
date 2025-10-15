"""
This file profide two functions to display molecules, the code is bad and in part AI generated but this do the job.
- plot_molecule(molecule, title="Molécule", show_charges=False, figsize=(10, 8))
- plot_multiple_molecules(molecules, n_molecules=6, cols=3, figsize=(15, 10))
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# Données de longueurs de liaison en picomètres (pm) depuis wiredchemist.com
# Format: (element1, element2): [(bond_type, length_pm), ...]
BOND_LENGTHS = {
    # Liaisons H
    (1, 1): [('single', 74)],
    (1, 6): [('single', 109)],
    (1, 7): [('single', 101)],
    (1, 8): [('single', 96)],
    (1, 9): [('single', 92)],
    
    # Liaisons C-C
    (6, 6): [('triple', 120), ('double', 134), ('single', 154)],
    
    # Liaisons C-N
    (6, 7): [('triple', 116), ('double', 129), ('single', 147)],
    
    # Liaisons C-O
    (6, 8): [('triple', 113), ('double', 120), ('single', 143)],
    
    # Liaisons C-F
    (6, 9): [('single', 135)],
    
    # Liaisons N-N
    (7, 7): [('triple', 110), ('double', 125), ('single', 145)],
    
    # Liaisons N-O
    (7, 8): [('double', 121), ('single', 140)],
    
    # Liaisons N-F
    (7, 9): [('single', 136)],
    
    # Liaisons O-O
    (8, 8): [('double', 121), ('single', 148)],
    
    # Liaisons O-F
    (8, 9): [('single', 142)],
    
    # Liaisons F-F
    (9, 9): [('single', 142)],
}

def mol_to_dict(mol):
    return {"N" : mol[1].shape[0],
            'coords': mol[0].numpy(),
            'elements': mol[1].numpy(),
            'charges': mol[2].numpy()}


def get_bond_type(element1, element2, distance_angstrom):
    """
    Détermine le type de liaison basé sur la distance et les éléments.
    
    Args:
        element1, element2: Numéros atomiques
        distance_angstrom: Distance en Angströms
        
    Returns:
        tuple: (bond_type, color, linewidth) ou None si pas de liaison
    """
    # Convertir en pm
    distance_pm = distance_angstrom * 100
    
    # Créer la clé (ordre n'importe pas)
    key = tuple(sorted([element1, element2]))
    
    if key not in BOND_LENGTHS:
        return None
    
    # Trouver le type de liaison le plus proche
    bonds = BOND_LENGTHS[key]
    tolerance = {
        'single' : 10,
        'double' : 5,
        'triple' : 3,
    }  # pm de tolérance
    
    for bond_type, ref_length in bonds:
        if distance_pm < ref_length + tolerance[bond_type]:
            # Retourner style selon le type
            if bond_type == 'single':
                return ('simple', 'black', 2.0)
            elif bond_type == 'double':
                return ('double', 'black', 3.0)
            elif bond_type == 'triple':
                return ('triple', 'black', 4.0)
    
    return None


def plot_molecule(molecule, title="Molécule", show_charges=False, figsize=(10, 8)):
    """
    Visualise une molécule en 3D avec matplotlib.
    
    Args:
        molecule: Dictionnaire contenant les attributs de la molécule
        title: Titre du graphique
        show_charges: Si True, affiche les charges dans la légende
        figsize: Taille de la figure
    """
    # Couleurs CPK (Corey-Pauling-Koltun) standard pour les éléments
    element_colors = {
        1: '#FFFFFF',   # H - Blanc
        6: '#808080',   # C - Gris
        7: '#0000FF',   # N - Bleu
        8: '#FF0000',   # O - Rouge
        9: '#90E050',   # F - Vert clair
    }
    
    # Noms des éléments
    element_names = {
        1: 'H',
        6: 'C',
        7: 'N',
        8: 'O',
        9: 'F',
    }
    
    # Rayons de van der Waals (en Angströms, échelle relative)
    element_sizes = {
        1: 120,   # H
        6: 170,   # C
        7: 155,   # N
        8: 152,   # O
        9: 147,   # F
    }
    
    # Créer la figure 3D
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    coords = molecule['coords']
    elements = molecule['elements']
    charges = molecule['charges']
    
    # Tracer chaque atome
    for i, (coord, element, charge) in enumerate(zip(coords, elements, charges)):
        color = element_colors.get(element, '#FF1493')  # Rose par défaut si élément inconnu
        size = element_sizes.get(element, 150) * 3
        name = element_names.get(element, f'Z={element}')
        
        label = f"{name}"
        if show_charges:
            label += f" (q={charge:.2f})"
        
        ax.scatter(coord[0], coord[1], coord[2], 
                  c=color, s=size, 
                  edgecolors='black', linewidth=1.5,
                  label=label if i == 0 or elements[i] != elements[i-1] else "")
    
    # Tracer les liaisons en utilisant les données de longueur de liaison
    max_bond_distance = 2.0  # Angströms (distance max pour considérer une liaison)
    
    # Vecteur de vue (de la caméra vers la scène)
    elev = ax.elev if hasattr(ax, 'elev') else 30
    azim = ax.azim if hasattr(ax, 'azim') else -60
    
    # Convertir angles en radians
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    
    # Vecteur de vue
    view_vector = np.array([
        np.cos(elev_rad) * np.cos(azim_rad),
        np.cos(elev_rad) * np.sin(azim_rad),
        np.sin(elev_rad)
    ])

    
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            distance = np.linalg.norm(coords[i] - coords[j])
            
            if distance < max_bond_distance:
                bond_info = get_bond_type(elements[i], elements[j], distance)
                
                if bond_info:
                    bond_type, color, linewidth = bond_info
                    
                    # Vecteur de la liaison
                    bond_vector = coords[j] - coords[i]
                    bond_vector_norm = bond_vector / np.linalg.norm(bond_vector)
                    
                    # Vecteur perpendiculaire pour offset (produit vectoriel)
                    offset_vector = np.cross(view_vector, bond_vector_norm)
                    offset_norm = np.linalg.norm(offset_vector)
                    
                    if offset_norm > 0.001:  # Éviter division par zéro
                        offset_vector = offset_vector / offset_norm
                        offset_distance = 0.02  # Distance entre lignes parallèles
                        
                        if bond_type == 'simple':
                            # Une seule ligne
                            ax.plot([coords[i][0], coords[j][0]],
                                   [coords[i][1], coords[j][1]],
                                   [coords[i][2], coords[j][2]],
                                   color=color, linewidth=2.0, alpha=0.8)
                        
                        elif bond_type == 'double':
                            # Deux lignes parallèles
                            offset = offset_vector * offset_distance
                            
                            ax.plot([coords[i][0] + offset[0], coords[j][0] + offset[0]],
                                   [coords[i][1] + offset[1], coords[j][1] + offset[1]],
                                   [coords[i][2] + offset[2], coords[j][2] + offset[2]],
                                   color=color, linewidth=2.0, alpha=0.8)
                            
                            ax.plot([coords[i][0] - offset[0], coords[j][0] - offset[0]],
                                   [coords[i][1] - offset[1], coords[j][1] - offset[1]],
                                   [coords[i][2] - offset[2], coords[j][2] - offset[2]],
                                   color=color, linewidth=2.0, alpha=0.8)
                        
                        elif bond_type == 'triple':
                            # Trois lignes parallèles
                            offset_distance = 0.03
                            offset = offset_vector * offset_distance
                            
                            # Ligne centrale
                            ax.plot([coords[i][0], coords[j][0]],
                                   [coords[i][1], coords[j][1]],
                                   [coords[i][2], coords[j][2]],
                                   color=color, linewidth=2.0, alpha=0.8)
                            
                            # Deux lignes de chaque côté
                            ax.plot([coords[i][0] + offset[0], coords[j][0] + offset[0]],
                                   [coords[i][1] + offset[1], coords[j][1] + offset[1]],
                                   [coords[i][2] + offset[2], coords[j][2] + offset[2]],
                                   color=color, linewidth=2.0, alpha=0.8)
                            
                            ax.plot([coords[i][0] - offset[0], coords[j][0] - offset[0]],
                                   [coords[i][1] - offset[1], coords[j][1] - offset[1]],
                                   [coords[i][2] - offset[2], coords[j][2] - offset[2]],
                                   color=color, linewidth=2.0, alpha=0.8)
                    else:
                        # Fallback si le vecteur est parallèle à la vue
                        ax.plot([coords[i][0], coords[j][0]],
                               [coords[i][1], coords[j][1]],
                               [coords[i][2], coords[j][2]],
                               color=color, linewidth=2.0, alpha=0.8)
    
    # Configuration du graphique
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_axis_off()
    
    # Légende (sans doublons)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # Aspect ratio égal pour une meilleure visualisation
    """
    max_range = np.array([coords[:, 0].max() - coords[:, 0].min(),
                         coords[:, 1].max() - coords[:, 1].min(),
                         coords[:, 2].max() - coords[:, 2].min()]).max() / 2.0
    
    mid_x = (coords[:, 0].max() + coords[:, 0].min()) * 0.5
    mid_y = (coords[:, 1].max() + coords[:, 1].min()) * 0.5
    mid_z = (coords[:, 2].max() + coords[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    """
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig, ax


def plot_multiple_molecules(molecules, n_molecules=6, cols=3, figsize=(15, 10)):
    rows = (n_molecules + cols - 1) // cols
    fig = plt.figure(figsize=figsize)
    
    element_colors = {1: '#FFFFFF', 6: '#808090', 7: '#0000FF', 8: '#FF0000', 9: '#90E050'}
    element_sizes = {1: 80, 6: 120, 7: 110, 8: 110, 9: 105}
    
    for idx in range(min(n_molecules, len(molecules))):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        
        molecule = molecules[idx]
        coords = molecule['coords']
        elements = molecule['elements']
        
        # Tracer les atomes
        for coord, element in zip(coords, elements):
            color = element_colors.get(element, '#FF1493')
            size = element_sizes.get(element, 100) * 3
            ax.scatter(coord[0], coord[1], coord[2], 
                      c=color, s=size,
                      edgecolors='black', linewidth=1)
        
        # Tracer les liaisons (version simplifiée pour grilles multiples)
        max_bond_distance = 2.0
        # Vecteur de vue (de la caméra vers la scène)
        elev = ax.elev if hasattr(ax, 'elev') else 30
        azim = ax.azim if hasattr(ax, 'azim') else -60
        
        # Convertir angles en radians
        elev_rad = np.radians(elev)
        azim_rad = np.radians(azim)
        
        # Vecteur de vue
        view_vector = np.array([
            np.cos(elev_rad) * np.cos(azim_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
            np.sin(elev_rad)
        ])
        
        for i in range(len(coords)):
          for j in range(i + 1, len(coords)):
            distance = np.linalg.norm(coords[i] - coords[j])
            
            if distance < max_bond_distance:
                bond_info = get_bond_type(elements[i], elements[j], distance)
                
                if bond_info:
                    bond_type, color, linewidth = bond_info
                    
                    # Vecteur de la liaison
                    bond_vector = coords[j] - coords[i]
                    bond_vector_norm = bond_vector / np.linalg.norm(bond_vector)
                    
                    # Vecteur perpendiculaire pour offset (produit vectoriel)
                    offset_vector = np.cross(view_vector, bond_vector_norm)
                    offset_norm = np.linalg.norm(offset_vector)
                    
                    if offset_norm > 0.001:  # Éviter division par zéro
                        offset_vector = offset_vector / offset_norm
                        offset_distance = 0.002  # Distance entre lignes parallèles
                        
                        if bond_type == 'simple':
                            # Une seule ligne
                            ax.plot([coords[i][0], coords[j][0]],
                                   [coords[i][1], coords[j][1]],
                                   [coords[i][2], coords[j][2]],
                                   color=color, linewidth=1.3, alpha=0.8)
                        
                        elif bond_type == 'double':
                            # Deux lignes parallèles
                            offset = offset_vector * offset_distance
                            
                            ax.plot([coords[i][0] + offset[0], coords[j][0] + offset[0]],
                                   [coords[i][1] + offset[1], coords[j][1] + offset[1]],
                                   [coords[i][2] + offset[2], coords[j][2] + offset[2]],
                                   color=color, linewidth=1.3, alpha=0.8)
                            
                            ax.plot([coords[i][0] - offset[0], coords[j][0] - offset[0]],
                                   [coords[i][1] - offset[1], coords[j][1] - offset[1]],
                                   [coords[i][2] - offset[2], coords[j][2] - offset[2]],
                                   color=color, linewidth=1.3, alpha=0.8)
                        
                        elif bond_type == 'triple':
                            # Trois lignes parallèles
                            offset_distance = 0.002
                            offset = offset_vector * offset_distance
                            
                            # Ligne centrale
                            ax.plot([coords[i][0], coords[j][0]],
                                   [coords[i][1], coords[j][1]],
                                   [coords[i][2], coords[j][2]],
                                   color=color, linewidth=1.3, alpha=0.8)
                            
                            # Deux lignes de chaque côté
                            ax.plot([coords[i][0] + offset[0], coords[j][0] + offset[0]],
                                   [coords[i][1] + offset[1], coords[j][1] + offset[1]],
                                   [coords[i][2] + offset[2], coords[j][2] + offset[2]],
                                   color=color, linewidth=1.3, alpha=0.8)
                            
                            ax.plot([coords[i][0] - offset[0], coords[j][0] - offset[0]],
                                   [coords[i][1] - offset[1], coords[j][1] - offset[1]],
                                   [coords[i][2] - offset[2], coords[j][2] - offset[2]],
                                   color=color, linewidth=1.3, alpha=0.8)
                    else:
                        # Fallback si le vecteur est parallèle à la vue
                        ax.plot([coords[i][0], coords[j][0]],
                               [coords[i][1], coords[j][1]],
                               [coords[i][2], coords[j][2]],
                               color=color, linewidth=1.3, alpha=0.8)
        
        ax.set_title(f'Molécule {idx + 1} ({molecule["N"]} atomes)', fontsize=10)
        ax.set_axis_off()
    
    plt.tight_layout()
    return fig

