# -*- mode: python ; coding: utf-8 -*-

import sys

app_name = 'AnalisadorIA'

# Esta seção já está correta para copiar os arquivos para a pasta de destino.
datas_list = [
    ('best.pt', '.'),
    ('Core/*.so', 'Core'),
    #('Core/*.pyd', 'Core')
]

binaries_list = []

a = Analysis(
    ['main_app.py'],
    pathex=[],
    binaries=binaries_list,
    datas=datas_list,
    hiddenimports=[
        'numpy', 'cv2', 'PIL', 'tqdm', 'PyQt6.sip',
        'ultralytics', 'torch', 'torchvision' 
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# A seção EXE cria o executável principal.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=None
)

# A seção COLLECT junta o EXE e todas as dependências em uma única pasta.
# É isso que cria o build no modo de diretório.
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=app_name,
)