# example of how to use fisx to estimate the spectrum from a target

from fisx import Elements
from fisx import Material
from fisx import Detector
from fisx import XRF
import pylab as plt
from collections import OrderedDict
import numpy as np
import random
import string


def randomstring(n):
    return "".join([random.choice("abcdefghijklmnopqrstuvwzyx") for i in range(n)])


def makexy_by_linetype(fluo):
    out = OrderedDict()
    for key in fluo:
        x = []
        y = []
        for layer in fluo[key]:
            peakList = list(fluo[key][layer].keys())
            peakList.sort()
            for peak in peakList:
                if "esc" in peak:
                    continue
                # energy of the peak
                energy = fluo[key][layer][peak]["energy"]
                # expected measured rate
                rate = fluo[key][layer][peak]["rate"]
                x.append(energy)
                y.append(rate)
        out[key] = (x, y)
    return out


def makexy_all(fluo):
    x = []
    y = []
    peaks = {}
    for key in fluo:
        for layer in fluo[key]:
            peakList = list(fluo[key][layer].keys())
            peakList.sort()
            for peak in peakList:
                if "esc" in peak:
                    continue
                # energy of the peak
                energy = fluo[key][layer][peak]["energy"]
                # expected measured rate
                rate = fluo[key][layer][peak]["rate"]
                x.append(energy)
                y.append(rate)
                peaks[energy] = key+" "+peak
    return np.array(x), np.array(y), peaks


def maketarget(target_elements):
    v = 1.0/len(target_elements)
    return {k: v for k in target_elements}


def getallKandL(wanted_K, wanted_L):
    return sorted([k+" K" for k in wanted_K]+[k+" L" for k in wanted_L])


elementsInstance = Elements()
elementsInstance.initializeAsPyMca()
Air = Material("Air", 0.0012048, 1.0)
Air.setCompositionFromLists(["C1", "N1", "O1", "Ar1", "Kr1"],
                            [0.0012048, 0.75527, 0.23178, 0.012827, 3.2e-06])
elementsInstance.addMaterial(Air)


def getfluo(target_elements_K, target_elements_L):

    # After the slow initialization (to be made once), the rest is fairly fast.
    xrf = XRF()
    xrf.setBeam(16.0)  # set incident beam as a single photon energy of 16 keV
    xrf.setBeamFilters([["Al1", 2.72, 0.11, 1.0]])  # Incident beam filters
    target_composition = maketarget(target_elements_K+target_elements_L)
    targetname = "".join(target_elements_K+target_elements_L)+randomstring(5)
    target = Material(targetname, 1.0, 1.0)
    target.setComposition(target_composition)
    elementsInstance.addMaterial(target)
    xrf.setSample([[targetname, 10.0, 0.1]])  # Sample, density and thickness (g/cm^3, cm)
    xrf.setGeometry(45., 45.)               # Incident and fluorescent beam angles
    detector = Detector("Au1", 19.3, 0.0002)  # Detector Material, density, thickness
    detector.setActiveArea(0.35*0.35)            # Area and distance in consistent units
    detector.setDistance(3.0)               # expected cm2 and cm.
    xrf.setDetector(detector)
    xrf.setAttenuators([["Air", 0.0012048, 5.0, 1.0],
                        ["Al1", 2.1, 0.002, 1.0]])  # Attenuators (density, thickness cm, funny factor)

    lines_wanted = getallKandL(target_elements_K, target_elements_L)
    fluo = xrf.getMultilayerFluorescence(lines_wanted,
                                         elementsInstance,
                                         secondary=2)
    return fluo


# print("Element   Peak          Energy       Rate      Secondary  Tertiary")
# for key in fluo:
#     for layer in fluo[key]:
#         peakList = list(fluo[key][layer].keys())
#         peakList.sort()
#         for peak in peakList:
#             # energy of the peak
#             energy = fluo[key][layer][peak]["energy"]
#             # expected measured rate
#             rate = fluo[key][layer][peak]["rate"]
#             # primary photons (no attenuation and no detector considered)
#             primary = fluo[key][layer][peak]["primary"]
#             # secondary photons (no attenuation and no detector considered)
#             secondary = fluo[key][layer][peak]["secondary"]
#             # tertiary photons (no attenuation and no detector considered)
#             tertiary = fluo[key][layer][peak].get("tertiary", 0.0)
#             # correction due to secondary excitation
#             enhancement2 = (primary + secondary) / primary
#             enhancement3 = (primary + secondary + tertiary) / primary
#             print("%s   %s    %.4f     %.3g     %.5g    %.5g" % \
#                                (key, peak + (13 - len(peak)) * " ", energy,
#                                rate, enhancement2, enhancement3))

def findinterferences(fluo, considerfactor=1e-2, interfere_dist_ev=50):
    xall, yall, peaks = makexy_all(fluo)
    sortargs = np.argsort(xall)
    xall = xall[sortargs]
    yall = yall[sortargs]
    miny = max(yall)*considerfactor
    considerinds = yall > miny
    xc = xall[considerinds]
    yc = yall[considerinds]
    diffs = np.diff(xc)
    interfereinds = np.where(np.diff(xc) < interfere_dist_ev*1e-3)[0]
    out = []
    for i in interfereinds:
        name1 = peaks[xc[i]]
        name2 = peaks[xc[i+1]]
        if name1.startswith(name2[:2]):
            continue
        out.append([i, xc[i], xc[i+1], yc[i], yc[i+1], name1, name2])
    return out


def printinterferences(fluo, considerfactor=1e-2, interfere_dist_ev=50):
    a = findinterferences(fluo, considerfactor, interfere_dist_ev=50)
    for (i, x1, x2, y1, y2, name1, name2) in a:
        print("%s %0.1f and %s %0.1f interere (diff %0.1f) with rates %0.1g and %0.1g" %
              (name1, x1*1e3, name2, x2*1e3, (x2-x1)*1e3, y1, y2))


def plotfluo(fluo):
    d = makexy_by_linetype(fluo)
    xall, yall, peaks = makexy_all(fluo)
    average_photon_energy = np.sum(xall*yall*1000.)/np.sum(yall)
    maxy = max(yall)
    plt.figure()
    for (k, v) in d.items():
        x, y = v
        plt.semilogy(np.array(x)*1000.0, y, "o", label=k)
    plt.legend()
    plt.xlabel("energy (eV)")
    plt.ylabel("rate")
    plt.ylim(maxy*1e-2, maxy*np.sqrt(10))
    plt.xlim(4000, 10000)
    plt.title("average photon energy {:.2f} eV".format(average_photon_energy))


fluo2 = getfluo([], ["Sm"])
plotfluo(fluo2)
print("Sm")
plt.show()
fluo2 = getfluo([], ["Gd"])
plotfluo(fluo2)
print("Nd")
plt.show()
fluo2 = getfluo(["V", "Mn", "Co"], [])
plotfluo(fluo2)
plt.show()
fluo2 = getfluo(["V", "Fe", "Co"], [])
plotfluo(fluo2)
plt.show()
fluo1 = getfluo(["Cr", "Fe", "Ni"], [])
plotfluo(fluo1)
plt.show()
fluo1 = getfluo(["V", "Mn", "Co", "Cr", "Fe", "Ni"], ["Sm", "Gd"])
plotfluo(fluo1)
plt.show()
fluo1 = getfluo(["Co"], ["Gd"])
printinterferences(fluo1, considerfactor=1e-3, interfere_dist_ev=150)
plotfluo(fluo1)
plt.show()
fluo1 = getfluo(["Fe"], ["Sm"])
printinterferences(fluo1, considerfactor=1e-3, interfere_dist_ev=150)
plotfluo(fluo1)
plt.show()
