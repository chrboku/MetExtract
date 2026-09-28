# Pyperclip v1.3
# A cross-platform clipboard module for Python. (only handles plain text for now)
# By Al Sweigart al@coffeeghost.net

# Usage:
#   import pyperclip
#   pyperclip.copy('The text to be copied to the clipboard.')
#   spam = pyperclip.paste()

# On Mac, this module makes use of the pbcopy and pbpaste commands, which should come with the os.
# On Linux, this module makes use of the xclip command, which should come with the os. Otherwise run "sudo apt-get install xclip"


# Copyright (c) 2010, Albert Sweigart
# All rights reserved.
#
# BSD-style license:
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the pyperclip nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY Albert Sweigart "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL Albert Sweigart BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Change Log:
# 1.2 Use the platform module to help determine OS.
# 1.3 Changed ctypes.windll.user32.OpenClipboard(None) to ctypes.windll.user32.OpenClipboard(0), after some people ran into some TypeError

import os
import platform


def winGetClipboard():
    CF_UNICODETEXT = 13
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    user32.OpenClipboard(0)
    try:
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return ""
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
        locked = kernel32.GlobalLock(handle)
        try:
            data = ctypes.c_wchar_p(locked).value
        finally:
            kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()
    return data


def winSetClipboard(text):
    GMEM_MOVEABLE = 0x0002
    CF_UNICODETEXT = 13
    text = str(text)
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    # Configure proper return/argument types so 64-bit handles/pointers are not truncated.
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    user32.SetClipboardData.restype = ctypes.c_void_p
    user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]

    # Number of bytes for a null-terminated wide (UTF-16) string.
    buffer_size = (len(text) + 1) * ctypes.sizeof(ctypes.c_wchar)
    hCd = kernel32.GlobalAlloc(GMEM_MOVEABLE, buffer_size)
    pchData = kernel32.GlobalLock(hCd)
    try:
        ctypes.memmove(pchData, ctypes.create_unicode_buffer(text), buffer_size)
    finally:
        kernel32.GlobalUnlock(hCd)

    user32.OpenClipboard(0)
    try:
        user32.EmptyClipboard()
        user32.SetClipboardData(CF_UNICODETEXT, hCd)
    finally:
        user32.CloseClipboard()


def macSetClipboard(text):
    outf = os.popen("pbcopy", "w")
    outf.write(text)
    outf.close()


def macGetClipboard():
    outf = os.popen("pbpaste", "r")
    content = outf.read()
    outf.close()
    return content


def gtkGetClipboard():
    return gtk.Clipboard().wait_for_text()


def gtkSetClipboard(text):
    cb = gtk.Clipboard()
    cb.set_text(text)
    cb.store()


def qtGetClipboard():
    return str(cb.text())


def qtSetClipboard(text):
    cb.setText(text)


def xclipSetClipboard(text):
    outf = os.popen("xclip -selection c", "w")
    outf.write(text)
    outf.close()


def xclipGetClipboard():
    outf = os.popen("xclip -selection c -o", "r")
    content = outf.read()
    outf.close()
    return content


def xselSetClipboard(text):
    outf = os.popen("xsel -i", "w")
    outf.write(text)
    outf.close()


def xselGetClipboard():
    outf = os.popen("xsel -o", "r")
    content = outf.read()
    outf.close()
    return content


if os.name == "nt" or platform.system() == "Windows":
    import ctypes

    getcb = winGetClipboard
    setcb = winSetClipboard
elif os.name == "mac" or platform.system() == "Darwin":
    getcb = macGetClipboard
    setcb = macSetClipboard
elif os.name == "posix" or platform.system() == "Linux":
    xclipExists = os.system("which xclip") == 0
    if xclipExists:
        getcb = xclipGetClipboard
        setcb = xclipSetClipboard
    else:
        xselExists = os.system("which xsel") == 0
        if xselExists:
            getcb = xselGetClipboard
            setcb = xselSetClipboard
        try:
            import gtk

            getcb = gtkGetClipboard
            setcb = gtkSetClipboard
        except Exception:
            try:
                app = QApplication([])
                cb = PyQt4.QtWidgets.QApplication.clipboard()
                getcb = qtGetClipboard
                setcb = qtSetClipboard
            except Exception:
                raise Exception("Pyperclip requires the gtk or PyQt4 module installed, or the xclip command.")
copy = setcb
paste = getcb
