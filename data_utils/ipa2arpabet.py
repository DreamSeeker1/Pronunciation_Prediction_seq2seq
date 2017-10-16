# coding=utf-8

"""Convert uft8 represented IPA to arpabet using ipapy(stress marks not supported yet)."""

import sys

sys.path.append('./ipapy/')

from ipapy.ipastring import IPAString
from ipapy.arpabetmapper import ARPABETMapper


def i2a(ipastring):
    """Convert an IPA string to an arpabet string.
    Args:
        ipastring: A string representing ipa using utf-8 encoding.
    Returns:
        arpabetstring: A list representing the corresponding arpabet.

    """
    amapper = ARPABETMapper()
    arpastring = amapper.map_unicode_string(ipastring, ignore=True, return_as_list=True)
    return arpastring


def test():
    """test i2a function, print the original IPA to arpabet.
    """
    a = u"ˈɑkən"
    print IPAString(unicode_string=a)
    print i2a(a)

# test()
