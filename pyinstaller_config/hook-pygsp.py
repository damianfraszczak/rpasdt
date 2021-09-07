from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("pygsp")
hiddenimports.extend(collect_submodules("pygsp.graphs"))
hiddenimports.extend(collect_submodules("pygsp.filters"))
hiddenimports.extend(collect_submodules("pygsp.graphs.nngraphs"))
