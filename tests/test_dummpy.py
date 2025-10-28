import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import kft_parser


def test_addition():
    assert 1 + 1 == 2

def test_import_kft_parser():
    from app import kft_parser
    assert hasattr(kft_parser, "parse_kft_report")
