"""RDKit-based 2D structure depiction for SMILES codes, used by the annotation panels.
Rendering is deliberately lazy (called only when a card is expanded) and cached per SMILES
string for the lifetime of the process, since the same compound is often matched for many
features.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from PySide6 import QtGui

STRUCTURE_SIZE = 250


@lru_cache(maxsize=512)
def render_structure_pixmap(smiles: str, size: int = STRUCTURE_SIZE) -> "QtGui.QPixmap | None":
    """Render *smiles* to a `QPixmap` of `size` x `size` px, or return None if the SMILES
    could not be parsed or RDKit is unavailable."""
    if not smiles:
        return None

    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
    except Exception:
        logging.error("Structure rendering skipped: RDKit could not be imported")
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png_bytes = drawer.GetDrawingText()
    except Exception:
        logging.exception("Failed to render structure for SMILES '%s'", smiles)
        return None

    pixmap = QtGui.QPixmap()
    if not pixmap.loadFromData(png_bytes, "PNG"):
        return None
    return pixmap
