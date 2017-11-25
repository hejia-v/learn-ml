# -*- coding: utf-8 -*-
import json
from collections import OrderedDict

with open('notbook.tpl', 'r', encoding='utf8') as f:
    NOTBOOK_TPL = json.load(f, object_pairs_hook=OrderedDict)
with open('notbook-md.tpl', 'r', encoding='utf8') as f:
    NOTBOOK_MD_TPL = json.load(f, object_pairs_hook=OrderedDict)
with open('notbook-code.tpl', 'r', encoding='utf8') as f:
    NOTBOOK_CODE_TPL = json.load(f, object_pairs_hook=OrderedDict)


class MDCell(object):
    def __init__(self):
        self.lines = []

    def add_line(self, line):
        self.lines.append(line)

    def output(self):
        o = {}
        o.update(NOTBOOK_MD_TPL)
        o['source'] = self.lines
        return o


class CodeCell(object):
    def __init__(self):
        self.lines = []

    def add_line(self, line):
        if line.strip().startswith('```'):
            return
        self.lines.append(line)

    def output(self):
        o = {}
        o.update(NOTBOOK_CODE_TPL)
        o['source'] = self.lines
        return o


def md_to_ipynb(filename):
    lines = []
    with open(filename, 'rt', encoding='utf8') as fd:
        lines = [f'{l.rstrip()}\n' for l in fd.readlines()]

    cells = []
    last_cell = None
    for line in lines:
        if last_cell is None:
            last_cell = MDCell()
            last_cell.add_line(line)
        elif line.strip().startswith('#'):
            last_cell = MDCell()
            cells.append(last_cell)
            last_cell.add_line(line)
        elif line.strip().startswith('```'):
            if not isinstance(last_cell, CodeCell):
                last_cell = CodeCell()
                cells.append(last_cell)
                last_cell.add_line(line)
            else:
                last_cell.add_line(line)
                last_cell = None
        else:
            last_cell.add_line(line)
    # print(cells)
    cells = [c.output() for c in cells]
    o = {}
    o.update(NOTBOOK_TPL)
    o['cells'] = cells

    outfile = filename.rpartition('.')[0] + '.ipynb'
    with open(outfile, 'wt', encoding='utf8') as f:
        json.dump(o, f, ensure_ascii=False, indent=2)


def main():
    import os
    dirpath = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.normpath(os.path.join(dirpath, '../nn_simple.md'))
    print(filename)
    md_to_ipynb(filename)


if __name__ == '__main__':
    main()
