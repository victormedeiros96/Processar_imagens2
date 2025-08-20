# ProcessadorDeImagens.spec
# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('trincas.onnx', '.'),
        ('panelas.onnx', '.'),
        ('Core/*.so', 'Core'),
        ('Core/*.pyd', 'Core')
    ],
    hiddenimports=[
        'numpy',
        'cv2',
        'PIL',
        'tqdm',
        'PyQt6.sip',
        'onnxruntime',
        'torch',
        'torchvision'
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    console=False
)

pyz = PYZ(a.pure)

# Escolha o modo de build que preferir (diretório ou onefile)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ProcessadorDeImagens',
    debug=False,
    strip=False,
    upx=True,
    console=False
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name='ProcessadorDeImagens'
)