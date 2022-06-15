#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# =============================================================================
#  Version: 3.0 (July 22, 2020)
#  Author: Giuseppe Attardi (attardi@di.unipi.it), University of Pisa

# =============================================================================
#  Copyright (c) 2009. Giuseppe Attardi (attardi@di.unipi.it).
# =============================================================================
#  This file is part of Tanl.
#
#  Tanl is free software; you can redistribute it and/or modify it
#  under the terms of the GNU Affero General Public License, version 3,
#  as published by the Free Software Foundation.
#
#  Tanl is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
# =============================================================================

"""Wikipedia Page Extractor:
Extracts a single page from a Wikipedia dump file.
"""

import logging
from typing import Union
import sys
import os.path
import re
import argparse
import bz2
from io import StringIO
from .extract import Extractor

# Program version
__version__ = '3.0.5'

# ----------------------------------------------------------------------
# READER

tagRE = re.compile(r'(.*?)<(/?\w+)[^>]*>(?:([^<]*)(<.*?>)?)?')
#                    1     2               3      4

def extract_page(input_file: str, id: Union[int, str] = 1, template=False, quiet=True) -> str:
    """
    :param input_file: name of the wikipedia dump file
    :param id: article id
    :param template: whether the article is a template or not
    :param quiet: whether to log info
    """
    set_logging(quiet)
    if input_file.lower().endswith(".bz2"):
        input = bz2.open(input_file, mode='rt', encoding='utf-8')
    else:
        input = open(input_file, encoding='utf-8')

    id = str(id)
    res = ""
    page = []
    for line in input:
        # line = line
        if '<' not in line:         # faster than doing re.search()
            if page:
                page.append(line)
            continue
        m = tagRE.search(line)
        if not m:
            continue
        tag = m.group(2)
        if tag == 'page':
            page = []
            page.append(line)
            in_article = False
        elif tag == 'id':
            curid = m.group(3)
            if id == curid:
                page.append(line)
                in_article = True
            elif not in_article and not template:
                page = []
        elif tag == 'title':
            if template:
                if m.group(3).startswith('Template:'):
                    page.append(line)
                else:
                    page = []
            else:
                page.append(line)
        elif tag == '/page':
            if page:
                page.append(line)
                res = ''.join(page)
                if not template:
                    break
            page = []
        elif page:
            page.append(line)

    input.close()
    return res

def extract_clean(input_file: str, id: Union[int, str] = 1, template=False, quiet=True) -> str:
    """
    :param input_file: name of the wikipedia dump file
    :param id: article id
    :param template: whether the article is a template or not
    :param quiet: whether to log info
    """
    set_logging(quiet)
    if input_file.lower().endswith(".bz2"):
        input = bz2.open(input_file, mode='rt', encoding='utf-8')
    else:
        input = open(input_file, encoding='utf-8')

    Extractor.tsMode = True

    id = str(id)
    res = StringIO()
    page = []

    found = False
    in_text = False
    for line in input:
        if '<' not in line:  # faster than doing re.search()
            if in_text and found:
                page.append(line)
            continue
        m = tagRE.search(line)
        if not m:
            continue
        tag = m.group(2)
        if tag == 'page':
            page = []
        elif tag == 'id':
            curid = m.group(3)
            if id == str(curid):
                found = True
        elif tag == 'title':
            title = m.group(3)
        elif tag == 'text':
            in_text = True
            line = line[m.start(3):m.end(3)]
            page.append(line)
            if m.lastindex == 4:  # open-close
                in_text = False
        elif tag == '/text':
            if m.group(1):
                page.append(m.group(1))
            in_text = False
        elif tag == 'base':
            # discover urlbase from the xml dump file
            # /mediawiki/siteinfo/base
            base = m.group(3)
            urlbase = base[:base.rfind("/")]
        elif in_text and found:
            page.append(line)
        elif tag == '/page' and found:
            Extractor(id=id, revid='', urlbase=urlbase, title=title, page=page).extract(out=res)
            break

    input.close()
    result = res.getvalue()
    return result if result else print("ERROR: left with empty doc")


def set_logging(quiet=True):
    '''
    Initialize logger for info
    '''
    if not quiet:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
    else:
        try:
            del logger
        except NameError:
            pass

def main():
    '''
    Main for parsing cmd arguments
    '''
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)
    parser.add_argument("input",
                        help="XML wiki dump file")
    parser.add_argument("--id", default="1",
                        help="article number")
    parser.add_argument("--template", action="store_true",
                        help="template number")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="suppress reporting progress info")
    parser.add_argument("-c", "--clean", action="store_false",
                        help="whether to clean up the document")
    parser.add_argument("-v", "--version", action="version",
                        version='%(prog)s ' + __version__,
                        help="print program version")

    args = parser.parse_args()
    if args.clean:
        print(extract_clean(args.input, args.id, args.template, args.quiet))
    else:
        print(extract_page(args.input, args.id, args.template, args.quiet))

if __name__ == '__main__':
    main()
