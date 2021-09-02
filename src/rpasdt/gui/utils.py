from typing import Any, Optional

from PyQt5.QtCore import QThreadPool
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
    QWidget,
)

from rpasdt.gui.dynamic_form.forms import DynamicDialog
from rpasdt.gui.dynamic_form.models import DynamicFormConfig
from rpasdt.gui.thread_utils import Worker
from rpasdt.model.constants import APP_ICON_PATH


def create_action(
    title: str,
    parent: QWidget = None,
    handler: any = None,
    tooltip: str = None,
    shortcut: str = None,
    icon: QIcon = None,
) -> QAction:
    action = QAction(title)
    if parent:
        action.setParent(parent)
    if shortcut:
        action.setShortcut(shortcut)
    if tooltip:
        action.setStatusTip(tooltip)
    if handler:
        action.triggered.connect(handler)
    if icon:
        action.setIcon(icon)
    return action


def show_message_box(
    title: str,
    text: str,
    icon: "QMessageBox_Icon" = QMessageBox.Information,
    informative_text: str = None,
    detailed_text: str = None,
    buttons=QMessageBox.Ok | QMessageBox.Cancel,
    handler=None,
):
    msg = QMessageBox()

    if title:
        msg.setWindowTitle(title)
    if text:
        msg.setText(text)
    if icon:
        msg.setIcon(icon)
    if informative_text:
        msg.setInformativeText(informative_text)

    if detailed_text:
        msg.setDetailedText(detailed_text)
    msg.setStandardButtons(buttons)
    if handler:
        msg.buttonClicked.connect(handler)
    msg.setWindowIcon(QIcon(APP_ICON_PATH))
    return msg.exec_()


def show_alert_dialog(
    title: str,
    text: str,
    icon: "QMessageBox_Icon" = QMessageBox.Information,
    informative_text: str = None,
    detailed_text: str = None,
):
    if QApplication.instance():
        return show_message_box(
            title=title,
            text=text,
            icon=icon,
            informative_text=informative_text,
            detailed_text=detailed_text,
            buttons=QMessageBox.Ok,
        )


def show_error_dialog(title: str, error_msg: str):
    show_alert_dialog(title, text=error_msg, icon=QMessageBox.Critical)


def show_dynamic_dialog(
    object: Any,
    title: Optional[str] = None,
    config: Optional[DynamicFormConfig] = None,
    parent: Optional["QWidget"] = None,
) -> Optional[Any]:
    config = config or DynamicFormConfig()
    if title:
        config.title = title
    dialog = DynamicDialog(object=object, config=config, parent=parent)
    val = dialog.exec_()
    if val == QDialog.Accepted:
        return dialog.object
    return None


pool = QThreadPool()


def run_long_task(
    function, function_args=[], function_kwargs={}, title: str = "", callback=None
):
    from rpasdt.common.exceptions import log_error

    progress_dialog = QProgressDialog()
    # make infinity progress bar
    progress_dialog.setMinimum(0)
    progress_dialog.setMaximum(0)
    progress_dialog.setValue(0)
    progress_dialog.setWindowTitle(title)
    progress_dialog.show()
    worker = Worker(function, *function_args, **function_kwargs)
    worker.signals.result.connect(
        lambda result: progress_dialog.close() and callback and callback(result)
    )
    worker.signals.error.connect(
        lambda error: progress_dialog.close()
        and log_error(
            type=error[0],
            exc=error[1],
            exc_traceback=error[2],
            title=f"Error while doing {title}",
        )
    )
    pool.start(worker)


def show_open_file_dialog() -> Optional[str]:
    result = QFileDialog.getOpenFileName()
    return result[0] if result else ""


def show_save_file_dialog() -> Optional[str]:
    result = QFileDialog.getSaveFileName()
    return result[0] if result else ""
