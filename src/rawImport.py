from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import subprocess
import tempfile
import urllib.request
import zipfile
from lxml import etree


class UnsupportedFileError(RuntimeError):
    """Raised when importing a file is aborted because its format is unsupported."""


THERMO_RAW_FILE_PARSER_URL = "https://github.com/CompOmics/ThermoRawFileParser/releases/download/v1.4.3/ThermoRawFileParser1.4.3.zip"
THERMO_RAW_FILE_PARSER_RELATIVE_PATH = Path("ThermoRawFileParser_1.4.3") / "ThermoRawFileParser.exe"


def prepare_import_file(file_path):
    """Return one mzML/mzXML path, converting the RAW file when requested."""
    return prepare_import_files([file_path])[0]


def prepare_import_files(file_paths):
    """Prepare only the explicitly selected mzML, mzXML, and RAW files."""
    paths = [Path(file_path) for file_path in file_paths]
    raw_paths = []
    for path in paths:
        if path.suffix.lower() in {".mzml", ".mzxml"}:
            continue
        if path.suffix.lower() == ".raw":
            raw_paths.append(path)
            continue
        raise UnsupportedFileError(f"Unsupported file type: {path.name}")

    raw_paths_to_convert = [path for path in raw_paths if not path.with_suffix(".mzML").exists()]
    if not raw_paths_to_convert:
        return [str(path.with_suffix(".mzML") if path.suffix.lower() == ".raw" else path) for path in paths]

    from PySide6 import QtWidgets

    first_raw_path = raw_paths_to_convert[0]
    message_box = QtWidgets.QMessageBox(QtWidgets.QApplication.activeWindow())
    message_box.setIcon(QtWidgets.QMessageBox.Warning)
    message_box.setWindowTitle("Thermo RAW file")
    message_box.setText(f"'{first_raw_path.name}' is a Thermo RAW file, but MetExtract imports mzML or mzXML files.")
    message_box.setInformativeText(
        "Choose how this RAW file should be imported.\n\nNote: The ThermoRawFileParser will be downloaded and set up automatically if it is not already available.\nThis is an external software tool that is not part of MetExtract.\nNo personal data except for your IP address will be sent to any server during this process.\nNo guarantee is given that this software will work on your system, but it is open source and can be inspected by anyone.\nNo responsibility is taken for any damage that may occur by using this software."
    )
    message_box.addButton("Abort import", QtWidgets.QMessageBox.RejectRole)
    convert_button = message_box.addButton("Convert RAW files", QtWidgets.QMessageBox.AcceptRole)
    fix_button = message_box.addButton("Convert and fix MS/MS", QtWidgets.QMessageBox.AcceptRole)
    message_box.exec()

    selected_button = message_box.clickedButton()
    if selected_button not in (convert_button, fix_button):
        raise UnsupportedFileError(f"Unsupported RAW file import was aborted: {first_raw_path}")

    try:
        parser = _ensure_thermo_parser()
    except Exception as exc:
        QtWidgets.QMessageBox.critical(
            QtWidgets.QApplication.activeWindow(),
            "ThermoRawFileParser setup failed",
            f"The ThermoRawFileParser could not be downloaded or set up.\n\n{exc}",
        )
        raise UnsupportedFileError(f"ThermoRawFileParser setup failed: {exc}") from exc

    try:
        _convert_raw_files(raw_paths_to_convert, parser, selected_button is fix_button, QtWidgets)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(
            QtWidgets.QApplication.activeWindow(),
            "RAW conversion failed",
            f"One or more RAW files could not be converted.\n\n{exc}",
        )
        raise UnsupportedFileError(f"RAW conversion failed: {exc}") from exc

    for raw_path in raw_paths_to_convert:
        if not raw_path.with_suffix(".mzML").exists():
            raise RuntimeError(f"ThermoRawFileParser did not create an mzML file for '{raw_path}'.")
    return [str(path.with_suffix(".mzML") if path.suffix.lower() == ".raw" else path) for path in paths]


def _ensure_thermo_parser():
    from PySide6 import QtCore, QtWidgets

    repository_root = Path(__file__).resolve().parent.parent
    executable = repository_root / "ext_sw" / THERMO_RAW_FILE_PARSER_RELATIVE_PATH
    if executable.exists():
        return executable

    software_directory = executable.parent
    software_directory.mkdir(parents=True, exist_ok=True)

    progress = QtWidgets.QProgressDialog(
        "Downloading and setting up ThermoRawFileParser 1.4.3. This might take a few minutes...",
        None,
        0,
        0,
        QtWidgets.QApplication.activeWindow(),
    )
    progress.setWindowTitle("Setting up ThermoRawFileParser")
    progress.setWindowModality(QtCore.Qt.ApplicationModal)
    progress.setMinimumDuration(0)
    progress.setAutoClose(False)
    progress.show()
    QtWidgets.QApplication.processEvents()

    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "ThermoRawFileParser1.4.3.zip"

            def update_download_progress(_block_count, _block_size, _total_size):
                QtWidgets.QApplication.processEvents()

            urllib.request.urlretrieve(THERMO_RAW_FILE_PARSER_URL, archive, reporthook=update_download_progress)
            progress.setLabelText("Unpacking ThermoRawFileParser 1.4.3. This might take a few minutes...")
            QtWidgets.QApplication.processEvents()
            with zipfile.ZipFile(archive) as archive_file:
                for member in archive_file.infolist():
                    archive_file.extract(member, software_directory)
                    QtWidgets.QApplication.processEvents()
    finally:
        progress.close()

    if not executable.exists():
        raise RuntimeError(f"ThermoRawFileParser executable was not found after extraction: {executable}")
    return executable


def _convert_raw_file(raw_file, parser):
    result = subprocess.run(
        [str(parser), "-f", "1", "-a", "-e", "-x", "-i", str(raw_file), "-o", str(raw_file.parent)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"ThermoRawFileParser failed for '{raw_file}': {details}")


def _convert_raw_file_job(raw_file, parser, fix_msms):
    _convert_raw_file(raw_file, parser)
    if fix_msms:
        _fix_msms_precursor_info(raw_file.with_suffix(".mzML"))
    return raw_file


def _convert_raw_files(raw_files, parser, fix_msms, qt_widgets):
    from PySide6 import QtCore

    total_files = len(raw_files)
    if not total_files:
        return

    progress = qt_widgets.QProgressDialog(
        f"Converting RAW files: 0 of {total_files} completed...",
        None,
        0,
        total_files,
        qt_widgets.QApplication.activeWindow(),
    )
    progress.setWindowTitle("Converting Thermo RAW files")
    progress.setWindowModality(QtCore.Qt.ApplicationModal)
    progress.setMinimumDuration(0)
    progress.setAutoClose(False)
    progress.show()
    qt_widgets.QApplication.processEvents()

    errors = []
    completed_files = 0
    worker_count = min(20, total_files)
    try:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(_convert_raw_file_job, raw_file, parser, fix_msms): raw_file for raw_file in raw_files}
            for future in as_completed(futures):
                raw_file = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    errors.append(f"{raw_file.name}: {exc}")
                completed_files += 1
                progress.setValue(completed_files)
                progress.setLabelText(f"Converting RAW files: {completed_files} of {total_files} completed...")
                qt_widgets.QApplication.processEvents()
    finally:
        progress.close()

    if errors:
        raise RuntimeError("\n".join(errors))


def _fix_msms_precursor_info(mzml_file, ppm_deviation=1.0):
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(str(mzml_file), parser)
    changed = False

    for precursor in tree.iter():
        if _xml_local_name(precursor.tag) != "precursor":
            continue

        selected_ion = []
        isolation_target = []
        for cv_param in precursor.iter():
            if _xml_local_name(cv_param.tag) != "cvParam" or cv_param.get("cvRef") != "MS":
                continue
            if cv_param.get("accession") == "MS:1000744" and cv_param.get("name") == "selected ion m/z":
                selected_ion.append(cv_param)
            elif cv_param.get("accession") == "MS:1000827" and cv_param.get("name") == "isolation window target m/z":
                isolation_target.append(cv_param)

        if len(selected_ion) != 1 or len(isolation_target) != 1:
            continue

        selected_value = float(selected_ion[0].get("value"))
        target_value = float(isolation_target[0].get("value"))
        if target_value and abs(selected_value - target_value) / abs(target_value) * 1e6 >= ppm_deviation:
            selected_ion[0].set("value", str(target_value))
            changed = True

    if changed:
        tree.write(str(mzml_file), encoding="utf-8", xml_declaration=True)


def _xml_local_name(tag):
    return tag.rsplit("}", 1)[-1]
