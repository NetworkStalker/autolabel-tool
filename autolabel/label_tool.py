"""Run with python label_tool.py [project.dlabel]."""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from dataclasses import replace
from pathlib import Path
from threading import Event

import yaml
from PySide6.QtCore import QAbstractListModel, QModelIndex, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QFileDialog, QFrame, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QListView, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSplitter, QToolBar, QVBoxLayout, QWidget,
)

from label_canvas import Canvas, moved, pixmap_for
from label_dialogs import ExportDialog, MappingDialog, SourceDialog, TrainDialog, class_editor
from label_export import export_yolo
from label_predict import TEXT_MODEL, approve, batch, inspect_model
from label_project import IMAGE_EXTENSIONS, STATUSES, Project, class_names
from label_train import train_dataset


STYLE = """
QWidget { background: #151b29; color: #e9edf7; font-family: "Segoe UI"; font-size: 13px; }
QMainWindow, QMenuBar, QToolBar { background: #1c2435; }
QToolBar { padding: 6px; spacing: 6px; border: 0; }
QToolButton, QPushButton, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
    background: #263149; border: 1px solid #37445e; padding: 5px 8px; border-radius: 5px;
}
QPushButton:hover, QToolButton:hover, QMenu::item:selected { background: #3a3f65; }
QPushButton:checked, QToolButton:checked { background: #6248c8; border-color: #9c83ff; }
QPushButton:disabled, QToolButton:disabled { color: #798398; }
QListView, QTableWidget { background: #111827; border: 1px solid #303b53; border-radius: 5px; }
QListView::item { padding: 3px; }
QListView::item:selected, QTableWidget::item:selected { background: #45366d; color: white; }
QLabel[section="true"] { color: #b8a6ff; font-weight: 700; padding-top: 5px; }
QProgressBar { border: 1px solid #37445e; background: #111827; text-align: center; }
QProgressBar::chunk { background: #6248c8; }
QStatusBar { background: #252e46; }
QSplitter::handle { background: #303b53; }
"""
BADGES = dict(Unlabeled="–", Predicted="P", Labeled="L", Reviewed="R", Null="N")


class Filmstrip(QAbstractListModel):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.cache = OrderedDict()

    def set_rows(self, rows):
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def data(self, index, role):
        if not index.isValid() or index.row() >= len(self.rows):
            return None
        row = self.rows[index.row()]
        path = row["path"]
        if role == Qt.ItemDataRole.DisplayRole:
            return f"{BADGES[row['status']]}  {Path(path).name}"
        if role == Qt.ItemDataRole.ToolTipRole:
            return f"{row['status']} · {path}"
        if role == Qt.ItemDataRole.DecorationRole:
            if path not in self.cache:
                try:
                    self.cache[path] = QIcon(pixmap_for(path, thumbnail=True))
                except (OSError, ValueError):
                    self.cache[path] = QIcon()
                if len(self.cache) > 192:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(path)
            return self.cache[path]
        return None


class Job(QThread):
    result_ready = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int)

    def __init__(self, task, parent):
        super().__init__(parent)
        self.task = task
        self.cancelled = Event()

    def run(self):
        try:
            self.result_ready.emit(self.task(self.cancelled, self.progress.emit))
        except Exception as error:
            self.failed.emit(str(error))


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project = None
        self.image = None
        self.objects = []
        self.class_rows = {}
        self.class_id = None
        self.selected = ""
        self.history = defaultdict(lambda: ([], []))
        self.clipboard = None
        self.preview = {}
        self.job = None
        self.loaded = False
        self.review_mode = False
        self._changing_rows = False
        self._build()
        self.resize(1380, 850)
        self.setWindowTitle("Label Tool · DLabel")
        self.statusBar().showMessage("Create or open a project to get started.")

    def guarded(self, callback):
        def call(*_):
            try:
                callback()
            except Exception as error:
                QMessageBox.warning(self, "Label Tool", str(error))
        return call

    def action(self, menu, text, callback, toolbar=None):
        action = QAction(text, self)
        action.triggered.connect(self.guarded(callback))
        if menu:
            menu.addAction(action)
        if toolbar:
            toolbar.addAction(action)
        return action

    def shortcut(self, key, callback):
        QShortcut(QKeySequence(key), self, activated=self.guarded(
            lambda: callback() if not isinstance(self.focusWidget(), QLineEdit) else None
        ))

    def button(self, layout, text, callback):
        button = QPushButton(text)
        button.clicked.connect(self.guarded(callback))
        layout.addWidget(button)
        return button

    def section(self, layout, text):
        label = QLabel(text)
        label.setProperty("section", True)
        layout.addWidget(label)
        return label

    def _build(self):
        bar = self.menuBar()
        file_menu, edit_menu, view, data, model, export, help_menu = (
            bar.addMenu(name) for name in ("File", "Edit", "View", "Dataset", "Model", "Export", "Help")
        )
        toolbar = QToolBar("Actions")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self.action(file_menu, "New project", lambda: self.choose_project(True), toolbar)
        self.action(file_menu, "Open project", lambda: self.choose_project(False), toolbar)
        self.action(file_menu, "Add images", self.add_images, toolbar)
        self.action(file_menu, "Add image folder", self.add_folder)
        self.action(file_menu, "Refresh image files", self.refresh_image_files)
        self.action(file_menu, "Import YOLO dataset", self.import_yolo)
        self.action(file_menu, "Import video frames", self.import_video)
        toolbar.addSeparator()
        self.action(None, "‹ Prev", lambda: self.navigate(-1), toolbar)
        self.action(None, "Next ›", lambda: self.navigate(1), toolbar)
        self.action(edit_menu, "Undo  Ctrl+Z", lambda: self.undo(False), toolbar)
        self.action(edit_menu, "Redo  Ctrl+Y", lambda: self.undo(True))
        self.action(edit_menu, "Copy object  Ctrl+C", self.copy_object)
        self.action(edit_menu, "Paste  Ctrl+V", self.paste)
        self.action(edit_menu, "Copy previous labels  Ctrl+Shift+V", self.paste_previous)
        self.action(edit_menu, "Delete selected  Del", self.delete_object)
        toolbar.addSeparator()
        self.action(model, "Model Manager / Auto-label", self.configure_model, toolbar)
        self.action(model, "Your Models…", lambda: self.configure_model(your_models=True))
        self.action(model, "Train Dataset…", self.train_model)
        self.action(data, "Mark null  N", self.mark_null, toolbar)
        self.action(data, "Approve image  Enter", lambda: self.review(True), toolbar)
        self.action(data, "Reject predictions  X", lambda: self.review(False))
        self.action(data, "Review mode", self.toggle_review)
        self.action(data, "Assign split to selected images", self.set_split)
        self.action(export, "Export YOLO", self.export_dataset, toolbar)
        self.action(view, "Fit image  F", lambda: self.canvas.fit())
        self.action(view, "Actual size  0", lambda: self.canvas.actual_size())
        self.action(view, "Toggle class names  L", lambda: self.toggle_canvas("show_names"))
        self.action(view, "Toggle annotations  Ctrl+H", lambda: self.toggle_canvas("show_objects"))
        self.action(help_menu, "Keyboard shortcuts", self.show_help)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(10, 8, 10, 6)
        split = QSplitter(Qt.Orientation.Vertical)
        outer.addWidget(split, 1)

        workspace = QWidget()
        top = QHBoxLayout(workspace)
        top.setContentsMargins(0, 0, 0, 0)
        tools = QVBoxLayout()
        self.section(tools, "TOOLS")
        top.addLayout(tools)
        self.tool_buttons = {}
        for name, key, tool in (("Select", "H", "select"), ("Box", "B", "box"), ("Point", "P", "point"),
                                ("Polygon", "G", "polygon"), ("Keypoint", "K", "keypoint")):
            button = self.button(tools, f"{name}  {key}", lambda mode=tool: self.set_tool(mode))
            button.setCheckable(True)
            self.tool_buttons[tool] = button
        self.box_mode = QComboBox()
        self.box_mode.addItems(["Drag box", "Two corners"])
        self.box_mode.currentIndexChanged.connect(self.box_mode_changed)
        tools.addWidget(self.box_mode)
        self.visibility = QComboBox()
        self.visibility.addItems(["Visible", "Occluded", "Missing"])
        self.visibility.currentIndexChanged.connect(lambda index: setattr(self.canvas, "visibility", 2 - index))
        tools.addWidget(self.visibility)
        tools.addStretch()
        self.button(tools, "Fit  F", lambda: self.canvas.fit())
        self.canvas = Canvas()
        self.canvas.changed.connect(self.guarded_with_item)
        self.canvas.selected.connect(self.select_object)
        self.canvas.cursor_at.connect(lambda text: self.coordinates.setText(text))
        top.addWidget(self.canvas, 1)
        self.set_tool("box")

        panel = QFrame()
        panel.setMinimumWidth(265)
        panel.setMaximumWidth(360)
        right = QVBoxLayout(panel)
        right.setContentsMargins(4, 0, 0, 0)
        top.addWidget(panel)
        self.section(right, "CLASSES")
        self.classes = QListWidget()
        self.classes.currentItemChanged.connect(self.select_class)
        self.classes.itemDoubleClicked.connect(lambda _: self.guarded(self.edit_class)())
        right.addWidget(self.classes, 1)
        row = QHBoxLayout()
        self.button(row, "+ Class", self.add_class)
        self.button(row, "Edit", self.edit_class)
        self.button(row, "↑", lambda: self.move_class(-1))
        self.button(row, "↓", lambda: self.move_class(1))
        right.addLayout(row)
        row = QHBoxLayout()
        self.button(row, "Import", self.import_class_list)
        self.button(row, "Merge / delete", self.remove_class)
        self.button(row, "Apply class", self.reclassify)
        right.addLayout(row)
        self.object_title = self.section(right, "OBJECTS")
        self.object_list = QListWidget()
        self.object_list.currentItemChanged.connect(lambda item, _: self.select_object(
            item.data(Qt.ItemDataRole.UserRole) if item else ""
        ))
        right.addWidget(self.object_list, 1)
        row = QHBoxLayout()
        self.button(row, "Hide", lambda: self.toggle_object("hidden"))
        self.button(row, "Lock", lambda: self.toggle_object("locked"))
        self.button(row, "Delete", self.delete_object)
        right.addLayout(row)
        row = QHBoxLayout()
        self.button(row, "Accept", lambda: self.review_object(True))
        self.button(row, "Reject", lambda: self.review_object(False))
        self.button(row, "Undo joint", self.remove_joint)
        right.addLayout(row)
        self.section(right, "AUTO-LABEL")
        self.model_caption = QLabel("No model configured")
        self.model_caption.setWordWrap(True)
        right.addWidget(self.model_caption)
        row = QHBoxLayout()
        self.button(row, "Configure…", self.configure_model)
        self.button(row, "This image", self.quick_predict)
        right.addLayout(row)
        split.addWidget(workspace)

        bottom = QWidget()
        gallery = QVBoxLayout(bottom)
        gallery.setContentsMargins(0, 6, 0, 0)
        row = QHBoxLayout()
        self.section(row, "IMAGES")
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter by filename")
        row.addWidget(self.search, 1)
        self.filter_status = QComboBox()
        self.filter_status.addItems(["All statuses", *STATUSES])
        self.filter_status.currentIndexChanged.connect(lambda: self.refresh_images())
        row.addWidget(self.filter_status)
        self.filter_class = QComboBox()
        self.filter_class.addItem("All classes")
        self.filter_class.currentIndexChanged.connect(lambda: self.refresh_images())
        row.addWidget(self.filter_class)
        self.button(row, "Next unlabeled", self.next_unlabeled)
        self.auto_next = self.button(row, "Auto-next box  F8", self.update_status)
        self.auto_next.setCheckable(True)
        self.auto_next.setToolTip("After saving a new box, open the next unlabeled image in this view.")
        self.preview_button = self.button(row, "Exit preview", self.clear_preview)
        self.preview_button.hide()
        gallery.addLayout(row)
        self.film = Filmstrip()
        self.gallery = QListView()
        self.gallery.setModel(self.film)
        self.gallery.setViewMode(QListView.ViewMode.IconMode)
        self.gallery.setFlow(QListView.Flow.LeftToRight)
        self.gallery.setWrapping(False)
        self.gallery.setResizeMode(QListView.ResizeMode.Adjust)
        self.gallery.setLayoutMode(QListView.LayoutMode.Batched)
        self.gallery.setMovement(QListView.Movement.Static)
        self.gallery.setUniformItemSizes(True)
        self.gallery.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.gallery.setIconSize(QSize(140, 74))
        self.gallery.setGridSize(QSize(154, 104))
        self.gallery.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.gallery.selectionModel().currentChanged.connect(self.gallery_changed)
        gallery.addWidget(self.gallery)
        split.addWidget(bottom)
        split.setSizes([608, 160])
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        self.search_timer = QTimer(self, interval=180, singleShot=True)
        self.search_timer.timeout.connect(self.refresh_images)
        self.search.textChanged.connect(lambda: self.search_timer.start())

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(185)
        self.cancel = QPushButton("Cancel job")
        self.cancel.clicked.connect(self.cancel_job)
        self.progress.hide()
        self.cancel.hide()
        self.coordinates = QLabel()
        self.statusBar().addPermanentWidget(self.coordinates)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.cancel)
        keys = {
            "D": lambda: self.navigate(1), "A": self.previous_or_approve, "Ctrl+Right": self.next_unlabeled,
            "PgUp": lambda: self.navigate(-1), "PgDown": lambda: self.navigate(1),
            "F8": self.auto_next.click, "F1": self.show_help,
            "Ctrl+Z": lambda: self.undo(False), "Ctrl+Y": lambda: self.undo(True),
            "Ctrl+C": self.copy_object, "Ctrl+V": self.paste, "Ctrl+Shift+V": self.paste_previous,
            "Delete": self.delete_object, "F": lambda: self.canvas.fit(), "0": lambda: self.canvas.actual_size(),
            "L": lambda: self.toggle_canvas("show_names"), "Ctrl+H": lambda: self.toggle_canvas("show_objects"),
            "Return": self.enter, "Enter": self.enter, "N": self.mark_null,
            "X": lambda: self.review(False), "R": lambda: self.review(False) if self.review_mode else None,
            "E": lambda: self.set_tool("select"), "Ctrl+R": self.quick_predict,
            "Escape": self.canvas.reset_gesture,
        }
        keys.update({key: (lambda tool=tool: self.set_tool(tool)) for key, tool in
                     zip("HBPGK", ("select", "box", "point", "polygon", "keypoint"))})
        for direction, dx, dy in (("Left", -1, 0), ("Right", 1, 0), ("Up", 0, -1), ("Down", 0, 1)):
            keys[direction] = lambda x=dx, y=dy: self.nudge(x, y)
            keys["Shift+" + direction] = lambda x=dx, y=dy: self.nudge(10 * x, 10 * y)
        for key, callback in keys.items():
            self.shortcut(key, callback)
        for number in range(1, 10):
            self.shortcut(str(number), lambda hotkey=number: self.select_hotkey(hotkey))

    def show_help(self):
        QMessageBox.information(self, "Keyboard shortcuts",
            "PgUp / PgDn or A / D: previous / next · Ctrl+→: next unlabeled\n"
            "F8: toggle auto-next after drawing a box · F1: show shortcuts\n"
            "H / B / P / G / K: select / box / point / polygon / keypoint\n"
            "Shift+click: select in drawing tools · 1–9: class · Del: delete\n"
            "Arrow keys: move 1 px · Shift+arrows: 10 px\n"
            "Ctrl+Z/Y: undo/redo · Ctrl+C/V: copy/paste · Ctrl+Shift+V: previous labels\n"
            "Wheel: zoom · Space+drag or middle button: pan · F: fit · 0: actual size\n"
            "Enter: finish polygon / approve · X: reject · N: null · Ctrl+R: auto-label\n"
            "Review mode: A approve, R reject, E edit.")

    def choose_project(self, create):
        if self.job:
            return
        picker = QFileDialog.getSaveFileName if create else QFileDialog.getOpenFileName
        path, _ = picker(self, "New project" if create else "Open project", "", "Label Tool project (*.dlabel)")
        if path:
            if create and not path.lower().endswith(".dlabel"):
                path += ".dlabel"
            self.open_project(path, create)

    def open_project(self, path, create=False):
        project = Project(path, create=create)
        if self.project:
            self.project.close()
        self.project = project
        self.history.clear()
        self.clipboard = None
        self.preview = {}
        self.preview_button.hide()
        self.film.cache.clear()
        self.image = None
        self.review_mode = False
        self.auto_next.setChecked(False)
        self.class_id = None
        self.search.clear()
        self.filter_status.setCurrentIndex(0)
        self.filter_class.setCurrentIndex(0)
        self.setWindowTitle(f"{project.path.stem} · Label Tool")
        self.refresh_classes()
        self.refresh_images()
        self.refresh_model()

    def _ready(self):
        if not self.project:
            self.statusBar().showMessage("Create or open a project first.", 4000)
            return False
        if self.job or self.preview:
            self.statusBar().showMessage("Finish the running job or exit preview first.", 4000)
            return False
        return True

    def editable(self):
        return self.project and self.image and self.loaded and not self.job and not self.preview

    def start_job(self, title, task, callback, cancellable=False, total_steps=None, progress_unit="images"):
        if self.job:
            return
        self.canvas.reset_gesture()
        self.canvas.editable = False
        self.progress.setRange(0, 0)
        self.progress.show()
        self.cancel.setEnabled(True)
        self.cancel.setVisible(cancellable)
        worker = Job(task, self)
        self.job = worker
        self.job_title = title
        self.job_progress = (0, total_steps) if total_steps is not None else None
        self.job_progress_unit = progress_unit
        self.show_job_status()
        self.job_result = None
        self.job_error = None
        worker.result_ready.connect(self.receive_result)
        worker.failed.connect(self.receive_failure)
        worker.progress.connect(self.show_progress)
        worker.finished.connect(lambda: self.finish_job(worker, callback))
        worker.start()

    def receive_result(self, result):
        self.job_result = result

    def receive_failure(self, message):
        self.job_error = message

    def show_progress(self, done, total):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        if self.job_progress is not None:
            self.job_progress = (done, total)
            self.show_job_status()

    def show_job_status(self):
        message = self.job_title
        if self.job.cancelled.is_set():
            message = "Cancelling…" if self.job_progress_unit == "epochs" else "Cancelling after the current image…"
        if self.job_progress is not None:
            done, total = self.job_progress
            message += f"  {done:,}/{total:,} {self.job_progress_unit}"
        self.statusBar().showMessage(message)

    def finish_job(self, worker, callback):
        self.job = None
        self.canvas.editable = bool(self.editable())
        self.cancel.hide()
        self.progress.hide()
        worker.deleteLater()
        if self.job_error:
            self.update_status()
            QMessageBox.warning(self, "Label Tool", self.job_error)
        else:
            self.guarded(lambda: callback(self.job_result))()

    def cancel_job(self):
        if self.job:
            self.job.cancelled.set()
            self.cancel.setEnabled(False)
            self.show_job_status()

    def add_images(self):
        if not self._ready():
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Add images", "", "Images (" +
            " ".join("*" + suffix for suffix in sorted(IMAGE_EXTENSIONS)) + ")")
        if paths:
            self.import_paths(paths)

    def add_folder(self):
        if not self._ready():
            return
        path = QFileDialog.getExistingDirectory(self, "Add images from folder")
        if path:
            self.import_paths(Path(path))

    def import_yolo(self):
        if not self._ready():
            return
        label_path, _ = QFileDialog.getOpenFileName(self, "Choose YOLO class list or data.yaml", "",
                                                    "Class list (*.txt *.yaml *.yml)")
        if not label_path:
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose dataset or image folder", str(Path(label_path).parent))
        if folder:
            names = class_names(label_path)
            task = "detect"
            shape = None
            if Path(label_path).suffix.lower() in (".yml", ".yaml"):
                data = yaml.safe_load(Path(label_path).read_text(encoding="utf-8-sig"))
                if "kpt_shape" in data:
                    task, shape = "pose", data["kpt_shape"]
                    if not isinstance(shape, list) or len(shape) != 2 or not all(type(v) is int for v in shape) or shape[0] < 1 or shape[1] not in (2, 3):
                        raise ValueError("Invalid kpt_shape in dataset YAML.")
            self.import_paths(Path(folder), yolo_names=names, task=task, keypoint_shape=shape, mark_split=True)

    def import_paths(self, paths, **options):
        project_path = self.project.path

        def task(cancelled, progress):
            sources = sorted(p for p in paths.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS) if isinstance(paths, Path) else paths
            project = Project(project_path)
            try:
                return project.import_images(sources, **options)
            finally:
                project.close()

        def done(result):
            if "yolo_names" in options:
                self.history.clear()
            self.import_done(result)

        self.start_job("Importing images…", task, done)

    def import_done(self, result):
        count, errors = result
        self.refresh_classes()
        self.refresh_images()
        if errors:
            self.report(f"Imported {count} image(s). {len(errors)} error(s).", errors)
        else:
            self.statusBar().showMessage(f"Imported {count} image(s).", 5000)

    def import_video(self):
        if not self._ready():
            return
        from label_video import extract_frames
        path, _ = QFileDialog.getOpenFileName(self, "Choose local video", "", "Video (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        fps, ok = QInputDialog.getDouble(self, "Extract video frames", "Frames per second", 1.0, 0.01, 120, 2)
        if ok:
            project_path = self.project.path
            self.start_job("Extracting video frames…",
                           lambda cancel, progress: extract_frames(project_path, path, fps, cancel, progress),
                           self.import_done, cancellable=True)

    def refresh_classes(self):
        if not self.project:
            return
        rows = self.project.classes()
        self.class_rows = {row["id"]: row for row in rows}
        if self.class_id not in self.class_rows:
            self.class_id = rows[0]["id"] if rows else None
        filter_id = self.filter_class.currentData()
        self.classes.blockSignals(True)
        self.classes.clear()
        for row in rows:
            key = f"[{row['hotkey']}] " if row["hotkey"] else ""
            item = QListWidgetItem(f"{key}{row['name']}   ·   {row['count']}")
            item.setForeground(QColor(row["color"]))
            item.setData(Qt.ItemDataRole.UserRole, row["id"])
            self.classes.addItem(item)
            if row["id"] == self.class_id:
                self.classes.setCurrentItem(item)
        self.classes.blockSignals(False)
        self.filter_class.blockSignals(True)
        self.filter_class.clear()
        self.filter_class.addItem("All classes")
        for row in rows:
            self.filter_class.addItem(row["name"], row["id"])
        self.filter_class.setCurrentIndex(max(0, self.filter_class.findData(filter_id)))
        self.filter_class.blockSignals(False)
        self.canvas.active_class = self.class_id

    def select_class(self, item, _):
        if item:
            self.class_id = item.data(Qt.ItemDataRole.UserRole)
            self.canvas.active_class = self.class_id
            self.update_status()

    def select_hotkey(self, hotkey):
        for index, row in enumerate(self.class_rows.values()):
            if row["hotkey"] == hotkey:
                self.classes.setCurrentRow(index)
                break

    def add_class(self):
        if self._ready() and (values := class_editor(self)):
            name, color, _ = values
            self.class_id = self.project.add_class(name, color)
            self.refresh_classes()
            self.refresh_objects()

    def edit_class(self):
        if self._ready() and self.class_id is not None:
            row = self.class_rows[self.class_id]
            if values := class_editor(self, row):
                self.project.edit_class(self.class_id, *values)
                self.refresh_classes()
                self.refresh_objects()

    def import_class_list(self):
        if self._ready():
            path, _ = QFileDialog.getOpenFileName(self, "Import classes", "", "Classes (*.txt *.yaml *.yml)")
            if path:
                self.project.import_classes(class_names(path))
                self.refresh_classes()
                self.refresh_objects()

    def move_class(self, step):
        if self._ready() and self.class_id is not None:
            self.project.move_class(self.class_id, step)
            self.refresh_classes()
            self.refresh_objects()

    def remove_class(self):
        if not self._ready() or self.class_id is None:
            return
        alternatives = [row for row in self.class_rows.values() if row["id"] != self.class_id]
        choices = ["Delete class and its annotations", *[f"Merge into {row['name']}" for row in alternatives]]
        current = self.class_rows[self.class_id]["name"]
        choice, ok = QInputDialog.getItem(self, f"Remove {current}", "Apply to the whole project", choices, 0, False)
        if ok and QMessageBox.question(self, "Apply class change?", f"{current} → {choice}") == QMessageBox.StandardButton.Yes:
            index = choices.index(choice)
            self.project.remove_class(self.class_id, alternatives[index - 1]["id"] if index else None)
            self.history.clear()
            self.clipboard = None
            self.refresh_classes()
            self.refresh_images()

    def refresh_image_files(self):
        if self._ready():
            self.film.cache.clear()
            self.loaded = False
            self.refresh_images()

    def refresh_images(self, target=None):
        if not self.project:
            return
        if target is None and self.image:
            target = self.image["id"]
        status = self.filter_status.currentText() if self.filter_status.currentIndex() else None
        rows = self.project.images(status, self.search.text(), self.filter_class.currentData())
        if self.review_mode:
            scores = self.project.db.execute("""
                SELECT image_id, MIN(json_extract(payload, '$.score')) AS low FROM annotations
                WHERE json_extract(payload, '$.pending') = 1 GROUP BY image_id
            """).fetchall()
            by_id = {row["image_id"]: row["low"] for row in scores}
            rows.sort(key=lambda row: (by_id.get(row["id"]) is None, by_id.get(row["id"]) or 0))
        if self.preview:
            rows = [row for row in self.project.images() if row["id"] in self.preview]
        self._changing_rows = True
        self.film.set_rows(rows)
        index = next((i for i, row in enumerate(rows) if row["id"] == target), 0)
        if rows:
            model_index = self.film.index(index)
            self.gallery.setCurrentIndex(model_index)
            self.gallery.scrollTo(model_index)
        self._changing_rows = False
        self.load_image(rows[index] if rows else None)

    def gallery_changed(self, index, _):
        if not self._changing_rows and index.isValid():
            self.load_image(self.film.rows[index.row()])

    def load_image(self, row):
        same = row is not None and self.image is not None and row["id"] == self.image["id"]
        self.image = self.project.image(row["id"]) if row is not None else None
        if not same or not self.loaded:
            self.selected = ""
            self.loaded = False
            try:
                self.canvas.load(self.image["path"] if self.image else None)
                self.loaded = self.image is not None
            except (OSError, ValueError) as error:
                self.report(f"Could not open {self.image['path']}", [str(error)])
        if self.image:
            self.objects = self.preview.get(self.image["id"], self.project.objects(self.image["id"]))
        else:
            self.objects = []
        self.canvas.editable = bool(self.editable())
        self.refresh_objects()

    def navigate(self, step):
        if not self.film.rows:
            return
        row = self.gallery.currentIndex().row()
        self.gallery.setCurrentIndex(self.film.index(max(0, min(row + step, len(self.film.rows) - 1))))
        self.gallery.scrollTo(self.gallery.currentIndex())

    def next_unlabeled(self):
        if not self.project or self.preview:
            return
        current = self.image["id"] if self.image else 0
        unlabeled = self.project.images("Unlabeled")
        if not unlabeled:
            self.statusBar().showMessage("No unlabeled images.", 4000)
            return
        target = next((row["id"] for row in unlabeled if row["id"] > current), unlabeled[0]["id"])
        self.clear_filters()
        self.refresh_images(target)

    def advance_after_box(self):
        index = self.gallery.currentIndex().row()
        rows = self.film.rows
        remaining = rows[index + 1:] + rows[:index]
        target = next((row["id"] for row in remaining if row["status"] == "Unlabeled"), None)
        if target is None:
            self.statusBar().showMessage("Box saved. No unlabeled images left in this view.", 4000)
        else:
            self.refresh_images(target)

    def clear_filters(self):
        self.review_mode = False
        self.search.clear()
        self.filter_status.setCurrentIndex(0)
        self.filter_class.setCurrentIndex(0)

    def selected_images(self):
        return [self.film.rows[index.row()]["id"] for index in self.gallery.selectedIndexes()]

    def set_split(self):
        if not self._ready():
            return
        ids = self.selected_images()
        if not ids:
            return
        choice, ok = QInputDialog.getItem(self, "Assign split", f"{len(ids)} selected image(s)",
                                          ["Auto", "train", "val", "test"], 0, False)
        if ok:
            self.project.set_split(ids, None if choice == "Auto" else choice)
            self.refresh_images()

    def refresh_objects(self):
        self.object_list.blockSignals(True)
        self.object_list.clear()
        self.object_title.setText(f"OBJECTS  ·  {len(self.objects)}")
        if not any(obj.id == self.selected for obj in self.objects):
            self.selected = ""
        for obj in self.objects:
            row = self.class_rows[obj.class_id]
            score = f" · P {obj.score:.2f}" if obj.pending and obj.score is not None else ""
            joints = f" · {len(obj.keypoints)} joints" if obj.keypoints else ""
            flags = (" · hidden" if obj.id in self.canvas.hidden else "") + (" · locked" if obj.id in self.canvas.locked else "")
            item = QListWidgetItem(f"{row['name']} · {obj.kind}{score}{joints}{flags}")
            item.setForeground(QColor(row["color"]))
            item.setData(Qt.ItemDataRole.UserRole, obj.id)
            self.object_list.addItem(item)
            if obj.id == self.selected:
                self.object_list.setCurrentItem(item)
        self.object_list.blockSignals(False)
        self.canvas.set_objects(self.objects, self.class_rows, self.selected)
        self.update_status()

    def select_object(self, object_id):
        if self.selected != object_id:
            self.selected = object_id
            self.object_list.blockSignals(True)
            index = next((i for i, obj in enumerate(self.objects) if obj.id == object_id), -1)
            self.object_list.setCurrentRow(index)
            self.object_list.blockSignals(False)
            self.canvas.set_objects(self.objects, self.class_rows, self.selected)

    def update_status(self):
        if self.job:
            return
        if not self.image:
            self.statusBar().showMessage("No images in this view. Add images or change the filters.")
            return
        row = self.image
        cls = self.class_rows.get(self.class_id)
        prefix = "PREVIEW · " if self.preview else "REVIEW · " if self.review_mode else (
            "AUTO-NEXT · " if self.auto_next.isChecked() else ""
        )
        self.statusBar().showMessage(
            f"{prefix}{Path(row['path']).name}  ·  {self.gallery.currentIndex().row() + 1}/{len(self.film.rows)}"
            f"  ·  {row['width']}×{row['height']}  ·  {row['status']}  ·  {row['split'] or 'auto split'}"
            f"  ·  {self.canvas.tool.title()} / {cls['name'] if cls else 'add a class'}"
        )

    def commit(self, objects, status=None):
        if not self.editable():
            return
        image_id = self.image["id"]
        before = (self.objects, self.image["status"])
        self.project.save(image_id, objects, status)
        undo, redo = self.history[image_id]
        undo.append(before)
        del undo[:-100]
        redo.clear()
        self.reload_after_edit()

    def reload_after_edit(self):
        self.image = self.project.image(self.image["id"])
        self.objects = self.project.objects(self.image["id"])
        current = self.gallery.currentIndex()
        if current.isValid() and self.film.rows[current.row()]["id"] == self.image["id"]:
            self.film.rows[current.row()] = self.image
            self.film.dataChanged.emit(current, current)
        self.refresh_classes()
        self.refresh_objects()

    def undo(self, redo):
        if not self.editable():
            return
        undo_stack, redo_stack = self.history[self.image["id"]]
        source, target = (redo_stack, undo_stack) if redo else (undo_stack, redo_stack)
        if source:
            objects, status = source[-1]
            self.project.save(self.image["id"], objects, status)
            source.pop()
            target.append((self.objects, self.image["status"]))
            self.reload_after_edit()

    def guarded_with_item(self, item):
        self.guarded(lambda: self.save_object(item))()

    def save_object(self, item):
        if not self.editable():
            return
        objects = [item if obj.id == item.id else obj for obj in self.objects]
        new_object = not any(obj.id == item.id for obj in objects)
        if new_object:
            objects.append(item)
        self.selected = item.id
        self.commit(objects)
        if new_object and item.kind == "box" and self.auto_next.isChecked() and not self.review_mode:
            self.advance_after_box()

    def current_object(self, unlocked=False):
        if unlocked and self.selected in self.canvas.locked:
            return None
        return next((obj for obj in self.objects if obj.id == self.selected), None)

    def delete_object(self):
        if self.editable() and self.current_object(unlocked=True):
            self.commit([obj for obj in self.objects if obj.id != self.selected])

    def reclassify(self):
        if self.editable() and self.class_id is not None and (obj := self.current_object(unlocked=True)):
            if self.class_id != obj.class_id:
                self.save_object(replace(obj, class_id=self.class_id))

    def remove_joint(self):
        if self.editable() and (obj := self.current_object(unlocked=True)) and obj.keypoints:
            self.save_object(replace(obj, keypoints=obj.keypoints[:-1]))

    def toggle_object(self, attribute):
        if self.current_object():
            values = getattr(self.canvas, attribute)
            values.symmetric_difference_update([self.selected])
            self.refresh_objects()

    def toggle_canvas(self, attribute):
        setattr(self.canvas, attribute, not getattr(self.canvas, attribute))
        self.canvas.viewport().update()

    def copy_object(self):
        if self.image and not self.preview and (obj := self.current_object()):
            self.clipboard = ([obj], self.image["width"], self.image["height"])

    def paste_objects(self, source, width, height):
        if not self.editable():
            return
        sx, sy = self.image["width"] / width, self.image["height"] / height
        copied = [replace(obj.copy(),
                          points=tuple(v * (sy if i % 2 else sx) for i, v in enumerate(obj.points)),
                          keypoints=tuple((x * sx, y * sy, visible) for x, y, visible in obj.keypoints))
                  for obj in source if obj.class_id in self.class_rows]
        if copied:
            self.selected = copied[-1].id
            self.commit(self.objects + copied)

    def paste(self):
        if self.clipboard:
            self.paste_objects(*self.clipboard)

    def paste_previous(self):
        if self.editable():
            previous = self.project.db.execute("SELECT * FROM images WHERE id < ? ORDER BY id DESC LIMIT 1",
                                               (self.image["id"],)).fetchone()
            if previous:
                self.paste_objects(self.project.objects(previous["id"]), previous["width"], previous["height"])

    def nudge(self, x, y):
        if self.editable() and (obj := self.current_object(unlocked=True)):
            changed = moved(obj, x, y, self.image["width"], self.image["height"])
            if changed != obj:
                self.save_object(changed)

    def set_tool(self, tool):
        self.canvas.set_tool(tool)
        for key, button in self.tool_buttons.items():
            button.setChecked(tool == key)
        self.box_mode.setVisible(tool == "box")
        self.visibility.setVisible(tool == "keypoint")
        self.update_status()

    def box_mode_changed(self, index):
        self.canvas.two_click = bool(index)
        self.canvas.reset_gesture()

    def enter(self):
        if self.canvas.tool == "polygon" and self.canvas.vertices:
            self.canvas.finish_polygon()
        else:
            self.review(True)

    def mark_null(self):
        if not self.editable():
            return
        if self.objects and QMessageBox.question(self, "Mark null?", "Clear annotations and mark this image as background?") != QMessageBox.StandardButton.Yes:
            return
        if self.image["status"] != "Null":
            self.commit([], "Null")

    def review(self, accepted):
        if not self.editable() or not self.objects:
            return
        updated = approve(self.objects) if accepted else [obj for obj in self.objects if not obj.pending]
        if updated == self.objects and (not accepted or self.image["status"] == "Reviewed"):
            return
        self.commit(updated, "Reviewed" if updated else "Unlabeled")
        if self.review_mode:
            self.refresh_images()

    def review_object(self, accepted):
        if self.editable() and (obj := self.current_object(unlocked=True)) and obj.pending:
            updated = [replace(item, pending=False) if item.id == obj.id else item for item in self.objects
                       if accepted or item.id != obj.id]
            status = "Reviewed" if updated and not any(item.pending for item in updated) else None
            self.commit(updated, status)

    def toggle_review(self):
        if self.project and not self.preview:
            self.review_mode = not self.review_mode
            self.filter_status.setCurrentText("Predicted" if self.review_mode else "All statuses")
            self.refresh_images()

    def previous_or_approve(self):
        self.review(True) if self.review_mode else self.navigate(-1)

    def refresh_model(self):
        registered = self.project.models() if self.project else {}
        self.model_caption.setText("No model configured" if not registered else
                                   f"{Path(next(reversed(registered))).name} · {len(registered)} registered")

    def configure_model(self, your_models=False):
        if not self._ready():
            return
        registered = self.project.models()
        dialog = SourceDialog(self, registered, your_models=your_models)
        if not dialog.exec():
            return
        source, device, imgsz = dialog.values()
        if source == TEXT_MODEL:
            if not self.class_rows:
                self.report("Create a project class with + Class before using Auto-label from text.")
                return
            self.map_model(source, None, device, imgsz, registered.get(source, {}))
            return
        self.inspect_for_mapping(source, device, imgsz)

    def inspect_for_mapping(self, source, device, imgsz):
        image_path = self.image["path"] if self.image else None
        defaults = self.project.models().get(source, {})
        self.start_job("Inspecting model…",
                       lambda cancel, progress: inspect_model(source, image_path, device, imgsz),
                       lambda names: self.map_model(source, names, device, imgsz, defaults))

    def train_model(self):
        if not self._ready():
            return
        dataset, _ = QFileDialog.getOpenFileName(
            self, "Choose exported data.yaml", str(self.project.path.parent), "YOLO dataset (*.yaml *.yml)"
        )
        if not dataset:
            return
        dialog = TrainDialog(self)
        if not dialog.exec():
            return
        options = dialog.values()
        output = self.project.path.parent / "training-runs"
        self.start_job(
            "Training YOLO…",
            lambda cancel, progress: train_dataset(dataset, output, cancelled=cancel, progress=progress, **options),
            lambda result: self.training_done(result, options["imgsz"]),
            cancellable=True, total_steps=options["epochs"], progress_unit="epochs",
        )

    def training_done(self, result, imgsz):
        self.update_status()
        if not result or result["cancelled"]:
            message = "Training cancelled."
            if result:
                message += f"\nLog: {result['log']}"
            self.report(message)
            return
        source = str(result["best"].resolve())
        self.project.register_model(source, {"device": "cpu", "imgsz": imgsz})
        self.refresh_model()
        box = QMessageBox(self)
        box.setWindowTitle("Training complete")
        box.setText(f"Model saved to\n{result['best']}\n\nLog: {result['log']}")
        use_model = box.addButton("Use for Auto-label", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        if box.clickedButton() == use_model:
            self.inspect_for_mapping(source, "cpu", imgsz)

    def map_model(self, source, names, device, imgsz, defaults):
        self.update_status()
        overrides, ok = QInputDialog.getText(
            self, "Model class names", "Optional: comma-separated names in model ID order (for models without metadata)"
        ) if Path(source).suffix.lower() in (".onnx", ".engine") and len(names) > 100 else ("", True)
        if not ok:
            return
        if overrides.strip():
            names = dict(enumerate(name.strip() for name in overrides.split(",") if name.strip()))
        selected = self.selected_images()
        dialog = MappingDialog(self, names, list(self.class_rows.values()), device, imgsz, defaults, len(selected),
                               active_class=self.class_id, your_model=Path(source).is_absolute())
        if not dialog.exec():
            return
        config = dialog.values()
        if dialog.settings_only:
            self.save_model_settings(source, config)
            self.statusBar().showMessage("Model settings saved.", 4000)
            return
        scope = dialog.scope.currentIndex()
        if scope == 0:
            ids = [self.image["id"]] if self.image else []
        elif scope == 1:
            ids = selected
        else:
            ids = [row["id"] for row in self.project.images("Unlabeled" if scope == 2 else None)]
        if not ids:
            self.report("No images match this scope.")
            return
        preview = dialog.preview.isChecked()
        if not preview:
            self.save_model_settings(source, config)
        self.run_predictions(source, config, ids[:5] if preview else ids, preview)

    def save_model_settings(self, source, config):
        pending_names = list(dict.fromkeys(value for value in config["mapping"].values() if isinstance(value, str)))
        created = dict(zip(pending_names, self.project.import_classes(pending_names)))
        for key in ("mapping", "class_targets"):
            if key in config:
                config[key] = {cid: created.get(value, value) for cid, value in config[key].items()}
        self.project.register_model(source, config)
        self.refresh_classes()
        self.refresh_model()

    def quick_predict(self):
        if not self._ready() or not self.image:
            return
        models = self.project.models()
        if not models:
            self.configure_model()
            return
        source = next(reversed(models))
        config = models[source]
        config["mapping"] = {int(cid): value for cid, value in config.get("mapping", {}).items()
                             if value in self.class_rows}
        if not config["mapping"]:
            if Path(source).is_absolute():
                self.inspect_for_mapping(source, config.get("device", "cpu"), config.get("imgsz", 640))
            else:
                self.configure_model()
            return
        self.run_predictions(source, config, [self.image["id"]], False)

    def run_predictions(self, source, config, ids, preview):
        project_path = self.project.path
        self.start_job("Running auto-label…",
                       lambda cancel, progress: batch(project_path, ids, source, config, preview, cancel, progress),
                       lambda result: self.predictions_done(result, preview, ids),
                       cancellable=True, total_steps=len(ids))

    def predictions_done(self, result, preview, ids):
        if preview:
            self.preview = result["previews"]
            self.preview_button.setVisible(bool(self.preview))
        elif result["changed"]:
            for image_id in ids:
                self.history.pop(image_id, None)
        self.refresh_classes()
        self.refresh_images()
        message = (f"{'Preview' if preview else 'Auto-label'} · {result['changed']} image(s) changed · "
                   f"{result['boxes']} boxes · {result['skipped']} skipped · {len(result['errors'])} errors")
        if result["cancelled"]:
            message += " · Cancelled"
        self.report(message, result["errors"])

    def clear_preview(self):
        self.preview = {}
        self.preview_button.hide()
        self.refresh_images()

    def export_dataset(self):
        if not self._ready():
            return
        dialog = ExportDialog(self)
        if not dialog.exec():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Name a new export folder", str(
            self.project.path.parent / f"{self.project.path.stem}_yolo{dialog.version.currentText()}"
        ), "Dataset folder (*)")
        if not path:
            return
        options, project_path = dialog.values(), self.project.path

        def task(cancel, progress):
            project = Project(project_path)
            try:
                return export_yolo(project, path, progress=progress, **options)
            finally:
                project.close()

        def done(result):
            self.update_status()
            note = "" if result["train"] and result["val"] else "\nTraining needs non-empty train and validation splits."
            self.report(f"Exported to {path}\n" + "\n".join(f"{k}: {v}" for k, v in result.items()) + note)

        self.start_job("Exporting dataset…", task, done)

    def report(self, message, details=()):
        box = QMessageBox(self)
        box.setWindowTitle("Label Tool")
        box.setText(message)
        if details:
            box.setDetailedText("\n".join(details))
        box.exec()

    def closeEvent(self, event):
        if self.job:
            event.ignore()
            self.statusBar().showMessage("Wait for the job to finish before closing.")
            return
        if self.project:
            self.project.close()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", nargs="?")
    args = parser.parse_args()
    app = QApplication([])
    app.setApplicationName("Label Tool")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = Window()
    window.show()
    if args.project:
        window.guarded(lambda: window.open_project(args.project))()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

