"""Decoder-only LMs written by the project team (ported from the CS781 capstone).

Files are copied unchanged except for lines marked `# FIX (Claude)`.
"""

from . import decoder1, decoder3

ARCHS = {"decoder1": decoder1, "decoder3": decoder3}
