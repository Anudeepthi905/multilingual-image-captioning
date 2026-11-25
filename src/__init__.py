"""Multilingual image captioning package."""
import os
import sys

# The oneDNN (MKL) CPU kernels in the Windows TensorFlow build fail with
# "could not create a memory object" on InceptionV3 convolutions. Must be set
# before TensorFlow is imported.
if sys.platform == "win32":
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
