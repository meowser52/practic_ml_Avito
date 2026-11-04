"""Execute the supplied trusted notebook in the container, without Jupyter.

This intentionally executes local notebook code, just as a notebook kernel does.
Never point it at an untrusted downloaded notebook.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def run_notebook(path: Path) -> None:
    path = path.resolve(strict=True)
    notebook = json.loads(path.read_text())
    previous = Path.cwd()
    namespace = {'__name__': '__main__', '__file__': str(path)}
    try:
        os.chdir(path.parent)
        for index, cell in enumerate(notebook['cells'], 1):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                print(f'Notebook code cell {index}', flush=True)
                exec(compile(source, f'{path.name}:cell-{index}', 'exec'), namespace)
    finally:
        os.chdir(previous)


if __name__ == '__main__':
    run_notebook(Path(__file__).resolve().parents[1] / 'prototype.ipynb')
