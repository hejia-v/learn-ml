# -*- coding: utf-8 -*-
import os
import subprocess


class CmdError(Exception):
    pass


def run_cmd(command):
    ret = subprocess.call(command, shell=True)
    if ret != 0:
        cmd_str = command
        if isinstance(command, list):
            cmd_str = ' '.join(command)
        message = "Error running command: %s", cmd_str
        raise CmdError(message)


def update_blog(dst_dir, dst_name, src_filename):
    dst_dir = os.path.abspath(dst_dir)
    dst_image_dir = os.path.abspath(os.path.join(dst_dir, 'source'))
    src_image_dir = os.path.abspath(os.path.join(os.path.dirname(src_filename),  'images'))
    dst_filename = os.path.abspath(os.path.join(dst_dir, 'source', '_posts', dst_name))

    cp = 'D:/Program Files/Git/usr/bin/cp.exe'
    run_cmd([cp, '-rfv', src_image_dir, dst_image_dir])
    with open(src_filename, 'rt', encoding='utf8') as fd:
        src_blog = fd.readlines()
    title = src_blog[0][1:].strip()
    src_blog = src_blog[1:]
    with open(dst_filename, 'rt', encoding='utf8') as fd:
        dst_blog = fd.readlines()

    new_post = []
    headline_count = 0
    for line in dst_blog:
        line = line.strip()
        if line.startswith('---'):
            headline_count += 1
        if line.startswith('title'):
            line = f'title: {title}'
        new_post.append(line + '\n')
        if headline_count == 2:
            break
    new_post.extend(src_blog)
    new_post_text = ''.join(new_post)

    with open(dst_filename, 'wt', encoding='utf8') as fd:
        fd.write(new_post_text)

    print(dst_image_dir)
    print(src_image_dir)
    print(src_filename)
    print(dst_filename)


def main():
    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    md_filename = os.path.join(basedir, 'nn_simple.md')
    print(basedir)
    update_blog(f'{basedir}/../hejia-v.github.io', 'bp-neural-network.md', md_filename)


if __name__ == '__main__':
    main()
