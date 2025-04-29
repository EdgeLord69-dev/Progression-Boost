#!/usr/bin/env python3

# Dispatch-Server
# Copyright (c) Akatsumekusa and contributors

# ---------------------------------------------------------------------
# Set the port used by the dispatch server. You can set it to any port
# of your preference, as long as you set it the same in `Server.py`,
# `Server-Shutdown.py` and your filtering vpy script.
port = 18861
# ---------------------------------------------------------------------
# Copy every line in this file to your filtering vpy script. The
# optimal place to paste this is after you've imported vapoursynth and
# all the vsfunc's, and after you've loaded the source file, but before
# any filtering has been performed.
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ---------------------------------------------------------------------

import rpyc
import time

c = rpyc.connect("localhost", port)
tid = c.root.register()
while not c.root.request_release(tid):
    time.sleep(0.1)
