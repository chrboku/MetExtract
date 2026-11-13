"""Lazily loads MS/MS spectral library files referenced by the "MSMS_info" sheet, keeping
them in memory for the lifetime of the loaded results file (used to draw mirror plots for
"5_Annotated_MSMS" matches, since only the match summary - not the library spectrum itself -
is stored in the results file)."""

from __future__ import annotations

import logging


def _guess_library_file_type(path):
    p = str(path).lower()
    if p.endswith(".json"):
        return "json"
    if p.endswith(".msp"):
        return "msp"
    return "mgf"


class LibraryCache:
    """Maps a library name (as it appears in the "Library" column of "5_Annotated_MSMS",
    i.e. the library file's basename without extension) to its loaded spectra list,
    loading each library file at most once, on first use."""

    def __init__(self, db_con):
        self.db_con = db_con
        self._library_paths = None
        self._loaded_libraries = {}

    def _library_paths_by_name(self):
        if self._library_paths is None:
            self._library_paths = {}
            if self.db_con is not None and self.db_con.has_table("MSMS_info"):
                for row in self.db_con.tables["MSMS_info"].to_dicts():
                    name = row.get("Library")
                    path = row.get("Path")
                    if name and path:
                        self._library_paths[name] = path
        return self._library_paths

    def get_spectrum(self, library_name, compound_id):
        """Return the `MGFLibrarySpectrum` for *library_name*/*compound_id* (the same
        "db_spectrum_index" stored as "Compound_ID"), or None if unavailable."""
        if library_name is None or compound_id is None:
            return None

        if library_name not in self._loaded_libraries:
            self._loaded_libraries[library_name] = self._load_library(library_name)

        spectra = self._loaded_libraries[library_name]
        try:
            idx = int(compound_id)
        except (TypeError, ValueError):
            return None
        if spectra and 0 <= idx < len(spectra):
            return spectra[idx]
        return None

    def _load_library(self, library_name):
        path = self._library_paths_by_name().get(library_name)
        if not path:
            return []
        try:
            from ..MSMS import mgfLibrary

            entry = {"path": path, "type": _guess_library_file_type(path), "precursor_mz_key": None, "polarity_key": None}
            return mgfLibrary.load_library_entry(entry)
        except Exception:
            logging.exception("Failed to lazily load MS/MS library '%s' from '%s'", library_name, path)
            return []
