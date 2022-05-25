#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =============================================================================
#  Version: 3.0 (July 22, 2020)
#  Author: Giuseppe Attardi (attardi@di.unipi.it), University of Pisa
#
#  Contributors:
#   Antonio Fuschetto (fuschett@aol.com)
#   Leonardo Souza (lsouza@amtera.com.br)
#   Juan Manuel Caicedo (juan@cavorite.com)
#   Humberto Pereira (begini@gmail.com)
#   Siegfried-A. Gevatter (siegfried@gevatter.com)
#   Pedro Assis (pedroh2306@gmail.com)
#   Wim Muskee (wimmuskee@gmail.com)
#   Radics Geza (radicsge@gmail.com)
#   Nick Ulven (nulven@github)
#
# =============================================================================
#  Copyright (c) 2009-2020. Giuseppe Attardi (attardi@di.unipi.it).
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

# pylint: disable=unspecified-encoding
"""Wikipedia Extractor:
Extracts and cleans text from a Wikipedia database dump and stores output in a
number of files of similar size in a given directory.
Each file will contain several documents in the format:

    <doc id="" url="" title="">
        ...
        </doc>

If the program is invoked with the --json flag, then each file will                                            
contain several documents formatted as json ojects, one per line, with                                         
the following structure

    {"id": "", "revid": "", "url": "", "title": "", "text": "..."}

The program performs template expansion by preprocesssng the whole dump and
collecting template definitions.
"""

import argparse
import bz2
import logging
import os.path
from pathlib import Path
import re
import sys
from io import StringIO
from multiprocessing import Queue, Value, get_context, cpu_count
from timeit import default_timer
from typing import Union

# ===========================================================================

# Program version
__version__ = '3.0.6'

##
# Defined in <siteinfo>
# We include as default Template, when loading external template file.
known_namespaces = set(['Template'])

##
# The namespace used for template definitions
# It is the name associated with namespace key=10 in the siteinfo header.
template_namespace = ''
template_prefix = ''

##
# The namespace used for module definitions
# It is the name associated with namespace key=828 in the siteinfo header.
module_namespace = ''
module_prefix = ''

# ----------------------------------------------------------------------
# Expand using WikiMedia API
# import json

# def expandTemplates(text):
#     """Expand templates invoking MediaWiki API"""
#     text = urlib.urlencodew(text)
#     base = urlbase[:urlbase.rfind('/')]
#     url = base + "/w/api.php?action=expandtemplates&format=json&text=" + text
#     exp = json.loads(urllib.urlopen(url))
#     return exp['expandtemplates']['*']

# ------------------------------------------------------------------------------
# Output


class NextFile():
    """
    Synchronous generation of next available file name.
    """

    # FILES_PER_DIR = 100

    def __init__(self, path_name):
        self.path_name = path_name
        # self.dir_index = -1
        self.file_index = -1

    def next(self):
        # self.file_index = (self.file_index + 1) % NextFile.FILES_PER_DIR
        self.file_index += 1
        # if self.file_index == 0:
        #     self.dir_index += 1
        # dirname = self._dirname()
        # if not os.path.isdir(dirname):
        #     os.makedirs(dirname)
        return os.path.join(self.path_name, f'wiki_{self.file_index}')

    def _dirname(self):
        char1 = self.dir_index % 26
        char2 = int(self.dir_index / 26) % 26
        return os.path.join(self.path_name, '%c%c' % (ord('A') + char2, ord('A') + char1))

    def _filepath(self):
        # return '%s/wiki_%02d' % (self._dirname(), self.file_index)
        return f'wiki{self.file_index}'


class OutputSplitter():
    """
    File-like object, that splits output to multiple files of a given max size.
    """

    def __init__(self, next_file, max_file_size=0, compress=True):
        """
        :param nextFile: a NextFile object from which to obtain filenames
            to use.
        :param max_file_size: the maximum size of each file.
        :para compress: whether to write data with bzip compression.
        """
        self.next_file = next_file
        self.compress = compress
        self.max_file_size = max_file_size
        self.file = self.open(self.next_file.next())

    def reserve(self, size):
        if self.file.tell() + size > self.max_file_size:
            self.close()
            self.file = self.open(self.next_file.next())

    def write(self, data):
        self.reserve(len(data))
        if self.compress:
            self.file.write(data)
        else:
            self.file.write(data)

    def close(self):
        self.file.close()

    def open(self, filename):
        if self.compress:
            return bz2.BZ2File(filename + '.bz2', 'w')
        else:
            return open(filename, 'w')


# ----------------------------------------------------------------------
# READER

tagRE = re.compile(r'(.*?)<(/?\w+)[^>]*>(?:([^<]*)(<.*?>)?)?')
#                    1     2               3      4


def load_templates(file, n_articles=0, output_file=None):
    """
    Load templates from :param file:.
    :param output_file: file where to save templates and modules.
    """
    global template_prefix, template_namespace
    template_prefix = template_namespace + ':'
    global module_namespace, module_prefix
    module_prefix = module_namespace + ':'
    articles = 0
    templates = 0
    page = []
    in_text = False
    if output_file:
        output = open(output_file, 'w')
    for line in file:
        #line = line.decode('utf-8')
        if '<' not in line:  # faster than doing re.search()
            if in_text:
                page.append(line)
            continue
        m = tagRE.search(line)
        if not m:
            continue
        tag = m.group(2)
        if tag == 'page':
            page = []
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
        elif in_text:
            page.append(line)
        elif tag == '/page':
            if not output_file and not template_namespace:  # do not know it yet
                # we reconstruct it from the first title
                colon = title.find(':')
                if colon > 1:
                    template_namespace = title[:colon]
                    template_prefix = title[:colon + 1]
            # FIXME: should reconstruct also moduleNamespace
            if title.startswith(template_prefix):
                define_template(title, page)
                templates += 1
            # save templates and modules to file
            if output_file and (title.startswith(template_prefix) or
                                title.startswith(module_prefix)):
                output.write('<page>\n')
                output.write(f'   <title>{title}</title>\n')
                output.write('   <ns>10</ns>\n')
                output.write('   <text>')
                for p_line in page:
                    output.write(p_line)
                output.write('   </text>\n')
                output.write('</page>\n')
            page = []
            articles += 1
            if articles % 100000 == 0:
                logging.info("Preprocessed %d pages", articles)
            if n_articles != 0 and articles == n_articles:
                break
    if output_file:
        output.close()
        logging.info("Saved %d templates to '%s'", templates, output_file)
    return templates


def decode_open(filename, mode='rt', encoding='utf-8'):
    """
    Open a file, decode and decompress, depending on extension `gz`, or 'bz2`.
    :param filename: the file to open.
    """
    ext = os.path.splitext(filename)[1]
    if ext == '.gz':
        import gzip
        return gzip.open(filename, mode, encoding=encoding)
    if ext == '.bz2':
        return bz2.open(filename, mode=mode, encoding=encoding)
    return open(filename, mode, encoding=encoding)


def process_dump(input_file, template_file, out_file, file_size, file_compress,
                 process_count, html_safe, expand_templates, max_n=0):
    """
    :param input_file: name of the wikipedia dump file; '-' to read from stdin
    :param template_file: optional file with template definitions.
    :param out_file: directory where to store extracted data, or '-' for stdout
    :param file_size: max size of each extracted file, or None for no max (one file)
    :param file_compress: whether to compress files with bzip.
    :param process_count: number of extraction processes to spawn.
    :param html_safe: whether to produce html-safe output.
    :param expand_templates: whether to exapnd templates.
    :param max_n: (up to) how many articles to process; process all if 0.
    """
    global known_namespaces
    global template_namespace, template_prefix
    global module_namespace, module_prefix

    urlbase = ''                # This is obtained from <siteinfo>

    input = decode_open(input_file)

    # collect siteinfo
    for line in input:
        # line = line.decode('utf-8')
        m = tagRE.search(line)
        if not m:
            continue
        tag = m.group(2)
        if tag == 'base':
            # discover urlbase from the xml dump file
            # /mediawiki/siteinfo/base
            base = m.group(3)
            urlbase = base[:base.rfind("/")]
        elif tag == 'namespace':
            known_namespaces.add(m.group(3))
            if re.search('key="10"', line):
                template_namespace = m.group(3)
                template_prefix = template_namespace + ':'
            elif re.search('key="828"', line):
                module_namespace = m.group(3)
                module_prefix = module_namespace + ':'
        elif tag == '/siteinfo':
            break

    if expand_templates:
        # preprocess
        template_load_start = default_timer()
        if template_file and os.path.exists(template_file):
            logging.info("Preprocessing '%s' to collect template definitions: this may take some time.", template_file)
            file = decode_open(template_file)
            templates = load_templates(file, max_n)
            file.close()
        else:
            if input_file == '-':
                # can't scan then reset stdin; must error w/ suggestion to specify template_file
                raise ValueError("to use templates with stdin dump, must supply explicit template-file")
            logging.info("Preprocessing '%s' to collect template definitions: this may take some time.", input_file)
            templates = load_templates(input, max_n, template_file)
            input.close()
            input = decode_open(input_file)
        template_load_elapsed = default_timer() - template_load_start
        logging.info("Loaded %d templates in %.1fs", templates, template_load_elapsed)

    if out_file == '-':
        output = sys.stdout
        if file_compress:
            logging.warning("writing to stdout, so no output compression (use an external tool)")
    else:
        next_file = NextFile(out_file)
        output = OutputSplitter(next_file, file_size, file_compress)

    # process pages
    logging.info("Starting page extraction from %s.", input_file)
    extract_start = default_timer()

    # Parallel Map/Reduce:
    # - pages to be processed are dispatched to workers
    # - a reduce process collects the results, sort them and print them.

    # fixes MacOS error: TypeError: cannot pickle '_io.TextIOWrapper' object
    Process = get_context("fork").Process

    maxsize = 10 * process_count
    # output queue
    output_queue = Queue(maxsize=maxsize)

    ordinal = Value('i', 0)

    # Reduce job that sorts and prints output
    reduce = Process(name='reduce_process', target=reduce_process, args=(output_queue, output))
    reduce.start()

    # initialize jobs queue
    jobs_queue = Queue(maxsize=maxsize)

    # start worker processes
    logging.info("Using %d extract processes.", process_count)
    workers = []
    for i in range(max(1, process_count)):
        extractor = Process(name=f'extract_process{i}', target=extract_process,
                            args=(jobs_queue, output_queue, ordinal, max_n, html_safe))
        extractor.daemon = True  # only live while parent process lives
        extractor.start()
        workers.append(extractor)

    # Mapper process

    # we collect individual lines, since str.join() is significantly faster
    # than concatenation
    page = []
    id = ''
    revid = ''
    last_id = ''
    # ordinal = 0  # page count
    in_text = False
    redirect = False
    for line in input:
        if '<' not in line:  # faster than doing re.search()
            if in_text:
                page.append(line)
            continue
        m = tagRE.search(line)
        if not m:
            continue
        tag = m.group(2)
        if tag == 'page':
            page = []
            redirect = False
        elif tag == 'id' and not id:
            id = m.group(3)
        elif tag == 'id' and id: # <revision> <id></id> </revision>
            revid = m.group(3)
        elif tag == 'title':
            title = m.group(3)
        elif tag == 'redirect':
            redirect = True
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
        elif in_text:
            page.append(line)
        elif tag == '/page':
            colon = title.find(':')
            if (colon < 0 or (title[:colon] in accepted_namespaces)) and (id != last_id and
                    not redirect and not title.startswith(template_namespace)):
                # job = (id, revid, urlbase, title, page, ordinal)
                job = (id, revid, urlbase, title, page)
                jobs_queue.put(job)  # goes to any available extract_process
                last_id = id
                # ordinal += 1

            id = ''
            revid = ''
            page = []
            with ordinal.get_lock():
                if max_n != 0 and ordinal.value >= max_n:
                    break

    input.close()

    # signal termination
    for _ in workers:
        jobs_queue.put(None)
    # wait for workers to terminate
    for w in workers:
        w.join()

    # signal end of work to reduce process
    output_queue.put(None)
    # wait for it to finish
    reduce.join()

    if output != sys.stdout:
        output.close()
    extract_duration = default_timer() - extract_start
    extract_rate = (ordinal.value-1) / extract_duration
    logging.info("Finished %d-process extraction of %d articles in %.1fs (%.1f art/s)",
                 process_count, ordinal.value-1, extract_duration, extract_rate) # Count goes one over


# ----------------------------------------------------------------------
# Multiprocess support


def extract_process1(jobs_queue, output_queue, html_safe):
    """Pull tuples of raw page content, do CPU/regex-heavy fixup, push finished text
    :param jobs_queue: where to get jobs.
    :param output_queue: where to queue extracted text for output.
    :html_safe: whether to convert entities in text to HTML.
    """
    while True:
        job = jobs_queue.get()  # job is (id, revid, urlbase, title, page, ordinal)
        if job:
            out = StringIO()  # memory buffer
            Extractor(*job[:-1]).extract(out, html_safe)  # (id, revid, urlbase, title, page)
            text = out.getvalue()
            output_queue.put((job[-1], text))  # (ordinal, extracted_text)
            out.close()
        else:
            break

def extract_process(jobs_queue, output_queue, ordinal, max_count, html_safe):
    """Pull tuples of raw page content, do CPU/regex-heavy fixup, push finished text
    :param jobs_queue: where to get jobs.
    :param output_queue: where to queue extracted text for output.
    :html_safe: whether to convert entities in text to HTML.
    """
    while True:
        job = jobs_queue.get()  # job is (id, revid, urlbase, title, page, ordinal)
        if job:
            out = StringIO()  # memory buffer
            Extractor(*job).extract(out, html_safe)  # (id, revid, urlbase, title, page)
            text = out.getvalue()
            if text:
                with ordinal.get_lock():
                    if ordinal.value > max_count: # Last article doesnt get written, this fixes it (kinda)
                        out.close()
                        break
                    output_queue.put((ordinal.value, text))  # (ordinal, extracted_text)
                    ordinal.value += 1
            out.close()
        else:
            break


def reduce_process(output_queue, output):
    """Pull finished article text, write series of files (or stdout)
    :param output_queue: text to be output.
    :param output: file object where to print.
    """

    interval_start = default_timer()
    period = 100000
    # FIXME: use a heap
    ordering_buffer = {}  # collected pages
    next_ordinal = 0  # sequence number of pages
    while True:
        if next_ordinal in ordering_buffer:
            output.write(ordering_buffer.pop(next_ordinal))
            next_ordinal += 1
            # progress report
            if next_ordinal % period == 0:
                interval_rate = period / (default_timer() - interval_start)
                logging.info("Extracted %d articles (%.1f art/s)",
                             next_ordinal, interval_rate)
                interval_start = default_timer()
        else:
            # mapper puts None to signal finish
            pair = output_queue.get()
            if not pair:
                break
            ordinal, text = pair
            ordering_buffer[ordinal] = text


# ----------------------------------------------------------------------

# Minimum size of output files
minFileSize = 200 * 1024

def wiki_extract(input: Union[Path, str], output: Union[Path, str]="-", # Required arguments
    bytes="0", compress=False, json=False, # Output arguments (optional)
    max_n=0, keep_headers=True, headers_mark: str="===", html=False, keep_links=False, namespaces: str=None, templates: Union[Path, str]=None, expand_templates=True, html_safe=True, process_count=cpu_count()-1, # Processing arguments (optional)
    quiet=True, debug=False, article=False, ts_mode=False # Special arguments (optional)
):
    """
    main() from original WikiExtractor.py transformed into callable function.

    Parameters
    ----------
    input : Path or str
        Location of XML wiki dump file
    output : Path or str
        Location of directory for extracted files (or '-' for dumping to stdout). Directory is created if it doesn't exist.
    
    Optional output arguments
    bytes : str, default = "0"
        Maximum bytes per output file (n[KMG], e.g.: "1M"); "0" means to put a single article per file.
    compress : bool, default = False
        If True, compress the output files using bzip.
    json : bool, default = False
        If True, write output in json format instead of the default <doc> format.
    
    Optional processing arguments
    max_n: int, default = 0,
        Number of articles to process. If 0, process all.
    keep_headers: bool, default = True
        If True, preserve headers.
    headers_mark: str, default = '==='
        Set a string to use as marking for headers.
    html: bool, default = False
        If True, produce HTML output, subsumes "links" parameter.
    keep_links: bool, default = False
        If True, perserve links.
    namespaces: ["ns1", "ns2"], default = None
        Set accepted namespaces.
    templates: Path or str, default = None
        Set to use/create a file containing the expanded templates (to speed up subsequent extractions).
    expand_templates: bool, default = True
        If False, does not expand templates, making procesing significantly quicker.
    html_safe: bool, default = True
        If True, produce HTML safe ouput.
    process_count: int, default = cpu_count()-1
        Number of processes to use.

    Optional special arguments
    quiet: bool, default = True
        If True, does not report progress info.
    debug: bool, default = False
        If True, print debug info.
    article: bool, default = False
        If True analyze a file containing a single article (debug option).
    ts_mode: bool, default = False
        Specific for Text Segmentation datasets, if True, filters out short segments and lines.
    """
    global accepted_namespaces

    Extractor.keepLinks = keep_links
    Extractor.HtmlFormatting = html
    if html:
        Extractor.keepLinks = True
    Extractor.toJson = json
    Extractor.keepHeaders = keep_headers
    Extractor.headersMark = headers_mark
    Extractor.tsMode = ts_mode

    try:
        power = 'kmg'.find(bytes[-1].lower()) + 1
        # 0 bytes means put a single article per file.
        file_size = 0 if bytes == '0' else int(bytes[:-1]) * 1024 ** power
        if file_size and file_size < minFileSize:
            raise ValueError()
    except ValueError:
        logging.error('Insufficient or invalid size: %s', bytes)
        return

    if namespaces:
        accepted_namespaces = set(namespaces.split(','))

    FORMAT = '%(levelname)s: %(message)s'
    logging.basicConfig(format=FORMAT)

    logger = logging.getLogger()
    if not quiet:
        logger.setLevel(logging.INFO)
    if debug:
        logger.setLevel(logging.DEBUG)

    input_file = input

    if not Extractor.keepLinks:
        ignoreTag('a')

    if article:
        if templates:
            if os.path.exists(templates):
                with open(templates) as file:
                    load_templates(file)

        with open(input_file) as file:
            page = file.read()
            ids = re.findall(r'<id>(\d*?)</id>', page)
            id = ids[0] if ids else ''
            revid = ids[1] if len(ids) > 1 else ''
            m = re.search(r'<title>(.*?)</title>', page)
            if m:
                title = m.group(1)
            else:
                logging.error('Missing title element')
                return
            m = re.search(r'<base>(.*?)</base>', page)
            if m:
                base = m.group(1)
                urlbase = base[:base.rfind("/")]
            else:
                urlbase = ''
            Extractor(id, revid, urlbase, title, [page]).extract(sys.stdout)
        return

    output_path = output
    if output_path != '-' and not os.path.isdir(output_path):
        os.makedirs(output_path)

    process_dump(input_file, templates, output_path, file_size,
                 compress, process_count, html_safe, expand_templates, max_n)

def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)
    parser.add_argument("input",
                        help="XML wiki dump file")
    groupO = parser.add_argument_group('Output')
    groupO.add_argument("-o", "--output", default="-",
                        help="directory for extracted files (or '-' for dumping to stdout)")
    groupO.add_argument("-b", "--bytes", default="0",
                        help="maximum bytes per output file (default %(default)s); 0 means to put a single article per file",
                        metavar="n[KMG]")
    groupO.add_argument("-c", "--compress", action="store_true",
                        help="compress output files using bzip")
    groupO.add_argument("--json", action="store_true",
                        help="write output in json format instead of the default <doc> format")

    groupP = parser.add_argument_group('Processing')
    groupP.add_argument("-n", "--max-n", default=0,
                        help="max amount of articles to process; process all for 0")
    groupP.add_argument("-rh", "--remove-headers", action="store_false",
                        help="whether to remove the headers")
    groupP.add_argument("-hm", "--headers-mark", default="===",
                        help="set a string to use as marking for headers (default='===')")
    groupP.add_argument("--html", action="store_true",
                        help="produce HTML output, subsumes --links")
    groupP.add_argument("-l", "--links", action="store_true",
                        help="preserve links")
    groupP.add_argument("-ns", "--namespaces", default="", metavar="ns1,ns2",
                        help="accepted namespaces")
    groupP.add_argument("--templates",
                        help="use or create file containing templates")
    groupP.add_argument("--no-templates", action="store_false",
                        help="Do not expand templates")
    groupP.add_argument("--html-safe", action="store_false",
                        help="use to produce HTML safe output within <doc>...</doc>")
    parser.add_argument("--processes", type=int, default=cpu_count()-1,
                        help="Number of processes to use (default %(default)s)")

    groupS = parser.add_argument_group('Special')
    groupS.add_argument("-q", "--quiet", action="store_true",
                        help="suppress reporting progress info")
    groupS.add_argument("--debug", action="store_true",
                        help="print debug info")
    groupS.add_argument("-a", "--article", action="store_true",
                        help="analyze a file containing a single article (debug option)")
    groupS.add_argument("-ts", "--ts-mode", action="store_true",
                        help="activate for text segmentation specific filters")
    groupS.add_argument("-v", "--version", action="version",
                        version='%(prog)s ' + __version__,
                        help="print program version")

    args = parser.parse_args()

    wiki_extract(input=args.input, output=args.output, 
        bytes=args.bytes, compress=args.compress, json=args.json, 
        max_n=args.n_articles, keep_headers=not args.remove_headers, headers_mark=args.headers_mark, html=args.html, keep_links=args.links, namespaces=args.namespaces, templates=args.templates, expand_templates=not args.no_templates, html_safe=args.html_safe, process_count=args.process_count,
        quiet=args.quiet, debug=args.debug, article=args.article, ts_mode=args.ts_mode)


if __name__ == '__main__':
    from extract import Extractor, ignoreTag, define_template, accepted_namespaces # pylint: disable=import-error
    main()
else:
    from .extract import Extractor, ignoreTag, define_template, accepted_namespaces
