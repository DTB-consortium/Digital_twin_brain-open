# -*- coding: utf-8 -*- 
# @Time : 2022/8/7 11:13 
# @Author : lepold
# @File : pprint.py

import prettytable as pt


def pretty_print(content: str):
    """
    pretty print something inside a box.

    Parameters
    ----------
    content: str
        the information to print.

    Examples
    --------

    >>> pretty_print("Initialization")

    """
    screen_width = 80
    text_width = len(content)
    box_width = text_width + 6
    left_margin = (screen_width - box_width) // 2
    print()
    print(' ' * left_margin + '+' + '-' * (text_width + 2) + '+')
    print(' ' * left_margin + '|' + ' ' * (text_width + 2) + '|')
    print(' ' * left_margin + '|' + content + ' ' * (box_width - text_width - 4) + '|')
    print(' ' * left_margin + '|' + ' ' * (text_width + 2) + '|')
    print(' ' * left_margin + '+' + '-' * (text_width + 2) + '+')
    print()


def table_print(content: dict, n_rows=None):
    """
    display something inside a table.

    Parameters
    ----------
    content: dict
        key-value represent the variable name and variable value.

    n_rows: int

    n_columns: int

    """
    assert isinstance(content, dict)
    if n_rows is None or n_rows < len(content):
        n_rows = len(content)
    tb = pt.PrettyTable()
    content_list = [list(x) for x in content.items()]
    content_list = sum(content_list, [])
    tb.field_names = ["name", "value"]
    for i in range(n_rows):
        tb.add_row(content_list[2 * i: 2 * (i+1)])
    print(tb)