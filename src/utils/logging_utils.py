"""
src/utils/logging_utils.py
Flexible logging utilities with configurable output modes.

Provides a unified logging setup with support for:
- Console output (with optional progress filtering)
- File output (with rotation)
- Combined console + file output
- Complete silence (logging disabled, with critical error buffer)
- Severity-based filtering per output channel
- Automatic suppression of verbose third-party loggers

Log file path is resolved from configs/paths.yaml (project_dirs.logs)
or can be specified explicitly.

Typical usage:

    # Training mode: only warnings+errors to console, full debug to file
    setup_logging(level=logging.DEBUG, mode="both", console_level=logging.WARNING)

    # Debug mode: everything verbose to console
    setup_logging(level=logging.DEBUG, mode="console")

    # Production: errors to file only
    setup_logging(level=logging.ERROR, mode="file")

    # Silence all output (with critical buffer)
    setup_logging(mode="silent")
"""

import io
import logging
import sys
from pathlib import Path
from typing import Optional, Union
from enum import Enum

import yaml


# ============================================================================
# LOGGING MODE
# ============================================================================

class LoggingMode(Enum):
    """
    Output modes for the logging system.

    Attributes:
        CONSOLE: Log to stderr only.
        FILE: Log to file only.
        BOTH: Log to both console and file.
        SILENT: Suppress all output (critical errors are buffered for later review).
    """
    CONSOLE = "console"
    FILE = "file"
    BOTH = "both"
    SILENT = "silent"


# ============================================================================
# CRITICAL ERROR BUFFER (for silent mode)
# ============================================================================

class CriticalErrorBuffer(io.StringIO):
    """
    In-memory buffer that captures critical messages during silent mode.

    Allows retrieval of suppressed critical errors for post-hoc review
    without any console or file output during operation.
    """

    def __init__(self):
        super().__init__()
        self._critical_count = 0

    def write(self, s: str) -> int:
        if s.strip():
            self._critical_count += 1
        return super().write(s)

    @property
    def critical_count(self) -> int:
        """Number of critical messages buffered."""
        return self._critical_count

    def get_messages(self) -> str:
        """Retrieve all buffered critical messages."""
        return self.getvalue()

    def flush_to_stderr(self) -> None:
        """Dump all buffered messages to stderr."""
        content = self.getvalue()
        if content:
            sys.stderr.write(content)
            sys.stderr.flush()


# ============================================================================
# COLORED FORMATTER (console)
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Formatter with ANSI color codes for console output.

    Color scheme:
        DEBUG    - dimmed (gray)
        INFO     - default (no color)
        WARNING  - yellow
        ERROR    - red
        CRITICAL - red background
    """

    COLOR_MAP = {
        logging.DEBUG: "\033[2m",       # dim
        logging.INFO: "\033[0m",         # reset/default
        logging.WARNING: "\033[33m",     # yellow
        logging.ERROR: "\033[31m",       # red
        logging.CRITICAL: "\033[41m",    # red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLOR_MAP.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"

    @classmethod
    def supports_color(cls) -> bool:
        """Check if the terminal supports ANSI color codes."""
        if not sys.stderr.isatty():
            return False
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                return True
            except Exception:
                return False
        return True


# ============================================================================
# PATH RESOLUTION FROM CONFIG
# ============================================================================

def _find_project_root() -> Path:
    """
    Locate the project root by searching for 'configs/paths.yaml'.

    Starts from the directory of this file and traverses upward.

    Returns:
        Absolute path to the project root.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "configs" / "paths.yaml").exists():
            return current
        current = current.parent

    fallback = Path(__file__).resolve().parent.parent
    sys.stderr.write(
        f"WARNING: Project root not found, falling back to: {fallback}\n"
    )
    return fallback


def _get_default_log_dir() -> Path:
    """
    Resolve the default log directory from configs/paths.yaml.

    Reads 'project_dirs.logs' from the paths configuration file.
    Falls back to '<project_root>/logs' if the config is unavailable
    or the key is missing.

    Returns:
        Absolute path to the log directory.
    """
    root = _find_project_root()
    paths_yaml = root / "configs" / "paths.yaml"

    if paths_yaml.exists():
        try:
            with open(paths_yaml, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            logs_path = (
                config.get("project_dirs", {})
                .get("logs", None)
            )

            if logs_path:
                log_dir = Path(logs_path)
                if not log_dir.is_absolute():
                    log_dir = root / log_dir
                return log_dir

        except Exception:
            pass

    return root / "logs"


_PROJECT_ROOT = _find_project_root()
_DEFAULT_LOG_DIR = _get_default_log_dir()


# ============================================================================
# THIRD-PARTY LOGGER SUPPRESSION
# ============================================================================

# Loggers from these namespaces are forced to WARNING or higher
# to prevent debug spam from JIT compilers, audio backends, etc.
_NOISY_THIRD_PARTY_LOGGERS = [
    "numba",
    "numba.core",
    "numba.core.byteflow",
    "numba.core.ssa",
    "numba.core.interpreter",
    "resampy",
    "librosa",
    "matplotlib",
    "matplotlib.font_manager",
    "PIL",
    "PIL.PngImagePlugin",
    "urllib3",
    "urllib3.connectionpool",
]


def _silence_noisy_loggers() -> None:
    """Force all known noisy third-party loggers to WARNING level."""
    for name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


# ============================================================================
# LOGGING SETUP
# ============================================================================

class LoggingConfig:
    """
    Master logging configuration controller.

    Handles setup of all loggers with flexible output routing
    and severity filtering per channel.

    The default log file path is resolved from configs/paths.yaml
    (project_dirs.logs) if not specified explicitly. Verbose third-party
    loggers (numba, librosa, resampy, matplotlib, etc.) are automatically
    suppressed to WARNING level.

    Attributes:
        mode: Current logging mode (console/file/both/silent).
        log_file: Path to the log file (if file mode is active).
        root_level: Root logger severity threshold.
        console_level: Console output severity threshold.
        file_level: File output severity threshold.
        _buffer: Critical error buffer for silent mode.
    """

    def __init__(
        self,
        level: Union[int, str] = logging.INFO,
        mode: Union[str, LoggingMode] = "console",
        log_file: Optional[Union[str, Path]] = None,
        console_level: Optional[Union[int, str]] = None,
        file_level: Optional[Union[int, str]] = None,
        use_colors: Optional[bool] = None,
    ):
        """
        Initialize logging configuration.

        Args:
            level: Root logging level (default: INFO).
                   Accepts logging constants or string names.
            mode: Output mode - 'console', 'file', 'both', or 'silent'.
            log_file: Path to log file. If None and file mode is active,
                      defaults to '<logs_dir>/project.log'
                      where logs_dir is read from configs/paths.yaml.
            console_level: Override log level for console output.
                           If None, uses root level.
            file_level: Override log level for file output.
                        If None, uses DEBUG to capture everything.
            use_colors: Enable/disable ANSI colors for console.
                        If None, auto-detects terminal support.

        Example:
            # Training: minimal console, detailed file
            LoggingConfig(
                level=logging.DEBUG,
                mode="both",
                console_level=logging.WARNING,
                log_file="logs/training.log"
            )

            # Debugging: everything to console
            LoggingConfig(level=logging.DEBUG, mode="console")

            # Silence everything
            LoggingConfig(mode="silent")
        """
        self.mode = LoggingMode(mode) if isinstance(mode, str) else mode
        self._level = self._resolve_level(level)
        self._console_level = (
            self._resolve_level(console_level) if console_level else self._level
        )
        self._file_level = (
            self._resolve_level(file_level) if file_level else logging.DEBUG
        )
        self._use_colors = (
            use_colors if use_colors is not None
            else ColoredFormatter.supports_color()
        )

        self._buffer: Optional[CriticalErrorBuffer] = None
        self._setup_done = False

        if self.mode in (LoggingMode.FILE, LoggingMode.BOTH):
            if log_file is None:
                self.log_file = _DEFAULT_LOG_DIR / "project.log"
            else:
                self.log_file = Path(log_file)
                if not self.log_file.is_absolute():
                    self.log_file = _PROJECT_ROOT / self.log_file
        else:
            self.log_file = None

        self.apply()

    @staticmethod
    def _resolve_level(level: Union[int, str]) -> int:
        """Convert string level names to logging constants."""
        if isinstance(level, int):
            return level
        return getattr(logging, level.upper(), logging.INFO)

    def apply(self) -> None:
        """
        Apply the logging configuration to the root logger.

        Reconfigures all handlers based on the current mode and levels.
        Safe to call multiple times — clears previous handlers first.
        Also silences verbose third-party loggers to prevent noise from
        JIT compilers (numba), audio backends (librosa, resampy), and
        plotting libraries (matplotlib).

        Suppressed loggers still emit WARNING and ERROR messages.
        """
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.DEBUG)

        _silence_noisy_loggers()

        if self.mode == LoggingMode.SILENT:
            self._setup_silent_mode()
        else:
            self._setup_active_mode()

        self._setup_done = True

    def _setup_active_mode(self) -> None:
        """Configure logging for console, file, or both modes."""
        root_logger = logging.getLogger()

        if self.mode in (LoggingMode.CONSOLE, LoggingMode.BOTH):
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(self._console_level)
            console_handler.setFormatter(self._get_console_formatter())
            root_logger.addHandler(console_handler)

        if self.mode in (LoggingMode.FILE, LoggingMode.BOTH) and self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                self.log_file, encoding="utf-8"
            )
            file_handler.setLevel(self._file_level)
            file_handler.setFormatter(self._get_file_formatter())
            root_logger.addHandler(file_handler)

    def _setup_silent_mode(self) -> None:
        """Configure logging to buffer critical messages silently."""
        self._buffer = CriticalErrorBuffer()
        buffer_handler = logging.StreamHandler(self._buffer)
        buffer_handler.setLevel(logging.CRITICAL)
        buffer_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        root_logger = logging.getLogger()
        root_logger.addHandler(buffer_handler)

    def _get_console_formatter(self) -> logging.Formatter:
        """Build the console log formatter (with optional colors)."""
        fmt = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
        datefmt = "%H:%M:%S"
        if self._use_colors:
            return ColoredFormatter(fmt, datefmt=datefmt)
        return logging.Formatter(fmt, datefmt=datefmt)

    def _get_file_formatter(self) -> logging.Formatter:
        """Build the file log formatter (plain text, full timestamp)."""
        return logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    @property
    def logs_dir(self) -> Optional[Path]:
        """
        Directory where log files are stored.

        Returns:
            Path to the log directory, or None if no file logging is configured.
        """
        if self.log_file:
            return self.log_file.parent
        return None

    def get_buffer(self) -> Optional[CriticalErrorBuffer]:
        """
        Retrieve the critical error buffer (silent mode only).

        Returns:
            CriticalErrorBuffer if in silent mode, None otherwise.
        """
        return self._buffer

    def flush_critical_errors(self) -> None:
        """Dump buffered critical errors to stderr and clear the buffer."""
        if self._buffer and self._buffer.getvalue():
            sys.stderr.write("=" * 60 + "\n")
            sys.stderr.write("SUPPRESSED CRITICAL ERRORS:\n")
            sys.stderr.write("=" * 60 + "\n")
            self._buffer.flush_to_stderr()

    def __repr__(self) -> str:
        return (
            f"LoggingConfig(mode={self.mode.value}, "
            f"level={logging.getLevelName(self._level)}, "
            f"console_level={logging.getLevelName(self._console_level)}, "
            f"file_level={logging.getLevelName(self._file_level)}, "
            f"logs_dir={self.logs_dir})"
        )


# ============================================================================
# MIXIN FOR CLASSES
# ============================================================================

class LoggingMixin:
    """
    Mixin that provides a pre-configured logger to any class.

    The logger is automatically named after the module path
    and class name for clear log attribution.

    Usage:
        class MyProcessor(LoggingMixin):
            def process(self):
                self.logger.info("Processing started...")
    """

    @property
    def logger(self) -> logging.Logger:
        """
        Get a logger named after the module and class.

        Returns:
            Logger instance with name 'module.path.ClassName'.
        """
        if not hasattr(self, '_logger'):
            name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._logger = logging.getLogger(name)
        return self._logger


# ============================================================================
# CONVENIENCE SHORTCUT
# ============================================================================

_logging_config: Optional[LoggingConfig] = None


def setup_logging(
    level: Union[int, str] = logging.INFO,
    mode: Union[str, LoggingMode] = "console",
    log_file: Optional[Union[str, Path]] = None,
    console_level: Optional[Union[int, str]] = None,
    file_level: Optional[Union[int, str]] = None,
    use_colors: Optional[bool] = None,
) -> LoggingConfig:
    """
    One-line logging setup for the entire project.

    Creates and applies a LoggingConfig. Safe to call multiple times —
    subsequent calls reconfigure the root logger.

    If log_file is not specified and file mode is active, the path
    is resolved from configs/paths.yaml (project_dirs.logs).

    Verbose third-party loggers (numba, librosa, resampy, matplotlib, etc.)
    are automatically suppressed to WARNING level to prevent debug noise.

    Args:
        level: Root logging level (default: INFO).
        mode: 'console', 'file', 'both', or 'silent'.
        log_file: Path to log file. If None, uses configs/paths.yaml
                  or falls back to '<project_root>/logs/project.log'.
        console_level: Override level for console output.
        file_level: Override level for file output.
        use_colors: Enable/disable ANSI colors in console.

    Returns:
        The active LoggingConfig instance.

    Example:
        setup_logging(level=logging.DEBUG, mode="both",
                      console_level=logging.WARNING)
    """
    global _logging_config
    _logging_config = LoggingConfig(
        level=level,
        mode=mode,
        log_file=log_file,
        console_level=console_level,
        file_level=file_level,
        use_colors=use_colors,
    )
    return _logging_config


def get_logging_config() -> Optional[LoggingConfig]:
    """Return the currently active LoggingConfig, or None if not yet set up."""
    return _logging_config


# ============================================================================
# MAIN GUARD (self-test)
# ============================================================================

if __name__ == "__main__":
    print("=== Console mode (DEBUG) ===")
    config = setup_logging(level=logging.DEBUG, mode="console")
    logger = logging.getLogger("test")
    logger.debug("Debug message — should appear")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Verify third-party suppression
    numba_logger = logging.getLogger("numba.core.byteflow")
    print(f"\nThird-party logger check:")
    print(f"  numba.core.byteflow level: {logging.getLevelName(numba_logger.level)}")
    print(f"  numba.core.byteflow effective: {logging.getLevelName(numba_logger.getEffectiveLevel())}")

    print(f"\nDefault log directory: {_DEFAULT_LOG_DIR}")

    print("\n=== Silent mode ===")
    config = setup_logging(mode="silent")
    logger.critical("This is a critical error — should be buffered!")
    logger.error("This error should NOT appear")
    logger.info("This info should NOT appear")

    buffer = config.get_buffer()
    print(f"Buffered critical count: {buffer.critical_count}")
    print("\nFlushing buffer:")
    config.flush_critical_errors()

    print("\n=== Done ===")