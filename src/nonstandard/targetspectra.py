import xraylib, collections

elements = ["Cr","Fe","Co","Cu","Dy","Mn"]
atomic_numbers = [xraylib.SymbolToAtomicNumber(s) for s in elements]
lines = collections.OrderedDict([("Ka",xraylib.KA_LINE), ("Kb",xraylib.KB_LINE), ("Lb",xraylib.LB_LINE), ("La",xraylib.LA_LINE), ("Lg1",xraylib.LG1_LINE), ("Ll",xraylib.LL_LINE)])

def get_cs(element_symbol, exciation_keV, lines):
    atomic_number = xraylib.SymbolToAtomicNumber(element_symbol)
    crossections = [xraylib.CS_FluorLine_Kissel_Cascade(atomic_number, line, exciation_keV) for line in lines.values()]
    return crossections

get_cs("Mn", 12, lines)
