#!/usr/bin/env python3
"""pylidc 0.2.3 × numpy 2.x／Python 3.13 相容層。

pylidc 最後一次發版早於兩項標準庫移除，兩項都落在我們要用的路徑上：

1. **numpy ≥1.24 移除 `np.int`／`np.float`／`np.bool`**（七處）
   ——Contour.to_matrix、Annotation.boolean_mask 等，正好是共識分群與直徑計算。
2. **Python ≥3.12 移除 `configparser.SafeConfigParser`**（一處）
   ——讀 `~/.pylidcrc`，即 `to_volume()` 找 DICOM 的入口。

做法：在**自己的程式碼裡**還原這些別名，不改 site-packages。
理由：改 site-packages 不會進可複現包，別人照著跑會炸；這幾行會。
兩者都是「被移除的別名」而非行為變更（`np.int is int`、
`SafeConfigParser is ConfigParser` 在被移除前都為真），故不改變任何數值。

import 本模組即生效，且必須在 import pylidc 之前。
"""
import configparser
import warnings

import numpy as np

# Python 3.12 移除；3.2 起就只是 ConfigParser 的別名。
if not hasattr(configparser, "SafeConfigParser"):
    configparser.SafeConfigParser = configparser.ConfigParser

# numpy 1.20 起 deprecated、1.24 移除。還原為當初的定義本身。
for _name, _builtin in (("int", int), ("float", float), ("bool", bool), ("object", object)):
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")


def versions():
    """回傳進可複現包的版本字串。"""
    import pylidc
    import pydicom
    return dict(numpy=np.__version__, pylidc=pylidc.__version__,
                pydicom=pydicom.__version__)
