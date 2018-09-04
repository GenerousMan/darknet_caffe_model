#-*-coding:utf-8-*-
"""
Created on Tue Jun 6 11:40:23 2018

@author: yangs
"""
from collections import defaultdict

import sys
import os
import io

def check_if_exist(path):
    return os.path.exists(path)

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

# the function was copied from yad2k.py
def unique_config_sections(config_file):
    """Convert all config sections to have unique names.

    Adds unique suffixes to config sections for compability with configparser.
    """
    def one():
        return 1

    section_counters = defaultdict(one)
    output_stream = io.BytesIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream

# "UnpackVariable" was copied
def UnpackVariable(var, num):
    if type(var) is list and len(var) == num:
        return var
    else:
        ret = []
        if type(var) is list:
            assert len(var) == 1
            for i in range(num):
                ret.append(var[0])
        else:
            for i in range(num):
                ret.append(var)
        return ret    