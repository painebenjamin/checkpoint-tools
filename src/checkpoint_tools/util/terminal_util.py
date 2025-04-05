from typing import Any, Optional

__all__ = [
    "green",
    "red",
    "yellow",
    "blue",
    "cyan",
    "magenta",
    "maybe_use_termcolor",
]

def termcolor_is_available() -> bool:
    """
    Return whether termcolor is available.

    :return: Whether termcolor is available.
    """
    try:
        import termcolor
        return True
    except ImportError:
        return False

def maybe_use_termcolor(
    message: str,
    color: Optional[str]=None,
    **kwargs: Any
) -> str:
    """
    Return the message with color if termcolor is available.

    :param message: The message to display.
    :param color: The color to use.
    :param kwargs: Additional keyword arguments.
    :return: The formatted message.
    """
    if color is not None and termcolor_is_available():
        import termcolor
        return termcolor.colored(message, color, **kwargs)
    return message

def green(
    message: str,
    light: bool=True
) -> str:
    """
    Return a green message.

    :param message: The message to display.
    :param light: Whether to use a light green color.
    :return: The formatted message.
    """
    return maybe_use_termcolor(message, "light_green" if light else "green")

def red(
    message: str,
    light: bool=True
) -> str:
    """
    Return an red message.

    :param message: The message to display.
    :param light: Whether to use a light red color.
    :return: The formatted message.
    """
    return maybe_use_termcolor(message, "light_red" if light else "red")

def yellow(
    message: str,
    light: bool=True
) -> str:
    """
    Return a yellow message.

    :param message: The message to display.
    :param light: Whether to use a light yellow color.
    :return: The formatted message.
    """
    return maybe_use_termcolor(message, "light_yellow" if light else "yellow")

def blue(
    message: str,
    light: bool=True
) -> str:
    """
    Return an blue message.

    :param message: The message to display.
    :param light: Whether to use a light blue color.
    :return: The formatted message.
    """
    return maybe_use_termcolor(message, "light_blue" if light else "blue")

def cyan(
    message: str,
    light: bool=True
) -> str:
    """
    Return a cyan message.

    :param message: The message to display.
    :param light: Whether to use a light cyan color.
    :return: The formatted message.
    """
    return maybe_use_termcolor(message, "light_cyan" if light else "cyan")

def magenta(
    message: str,
    light: bool=True
) -> str:
    """
    Return a magenta message.

    :param message: The message to display.
    :param light: Whether to use a light magenta color.
    :return: The formatted message.
    """
    return maybe_use_termcolor(message, "light_magenta" if light else "magenta")
