"""Data scraper for C19-site

This script downloads the full html source code for the page
https://c19.se/

Then it locates the correct <script> tag (out of several possible)
and uses a regex to read a JSON-like struct to a string.

The string is parsed to the data format YAML,
it is a superset of JSON and able to handle a larger set of strings
(specifically here we need to be able to parse keys not in quotation marks).

The YAML data constructs a local data class `data.c19.C19Data`
and writes it to a JSON file. This last step can of course be modified to
instead write to another format.
"""

from typing import Union
import argparse
import re
from pathlib import Path
import yaml
import requests
from bs4 import BeautifulSoup as Bs
from data import c19

URL = "https://c19.se/"
TEST_URL = (Path.cwd() / "raw.html")
"""Regex patterns to match data entry in javascript"""
SERIES_START = re.compile(r"^\s*series: \[{\s*$")
SERIES_END = re.compile(r"^\s*}],\s*$")


def main():
    """Main entry point"""
    args = parse_args()
    if args.use_local:
        source = _get_local_source(TEST_URL)
    else:
        page = requests.get(URL)
        source = Bs(page.content, "html.parser")
    data_script = extract_data_script(source)
    time_series = extract_time_series(data_script)
    yaml_data = parse_to_dict(time_series)
    parsed_data = c19.C19Data.from_yaml_dict(yaml_data)
    parsed_data.save_to_json(args.output_file)


def extract_data_script(source):
    """Heuristic search for the correct script tag
    TODO: More specific search than just the size of the script.
    """
    data_script = None
    for script in source.find_all("script"):
        if script.contents:
            if len(script.contents[0]) > 5:
                data_script = script.contents[0]
    return data_script


def extract_time_series(data_script):
    """Extract times series from full javascript
    This looks for a starting json-like element
    'series [{ * }],'
    (spanning multiple rows)

    This would be much better if the JS could be properly parsed.
    That did not work out yet, though
    """
    script_lines = data_script.splitlines()
    in_series = False
    series_lines = []
    for line in script_lines:
        in_series |= SERIES_START.match(line) is not None

        if in_series:
            series_lines.append(line)

        if SERIES_END.match(line) is not None:
            break

    return "\n".join(series_lines)


def parse_to_dict(series_string):
    """Parse raw string to yaml"""
    # Remove trailing comma
    series_string = series_string[:-1]
    series_dict = yaml.load(series_string, Loader=yaml.SafeLoader)
    return series_dict


def _get_local_source(pseudo_url: Union[str, Path]):
    """Get locally stored html file.

    Avoids a download of the full site.
    Used only for testing during development since no new data is parsed.
    Raises exception if file does not exist or does not have a 'html' extension
    """
    pseudo_url = Path(pseudo_url)
    if pseudo_url.exists():
        if pseudo_url.suffix == ".html":
            return Bs(open(TEST_URL), "html.parser")

        raise ValueError("Expected file type 'html', found '{}'".format(
            pseudo_url.suffix))
    raise FileNotFoundError(
        "Local url '{}' does not exist. Re-run without 'use_local' arg.".
        format(pseudo_url.name))


def parse_args():
    """Args"""
    parser = argparse.ArgumentParser(description="Scrape data from 'C19.SE'")
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Location to save data to. Will overwrite existing file.")
    parser.add_argument(
        "--use_local",
        action="store_true",
        help="Use local html file: 'raw.html'. For debugging purposes")

    return parser.parse_args()


if __name__ == "__main__":
    main()
