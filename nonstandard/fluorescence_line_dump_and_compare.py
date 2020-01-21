import pickle
import mass
import numpy as np

attrs1 = ["integral_intensity","energies","fwhm"]
attrs2 = ["normalized_lorentzian_integral_intensity","energies","lorentzian_fwhm"]
# if mass._version.__version__=="0.6.2":
#     dumpattrs = attrs1
# elif mass._version.__version__=="0.6.3":
#     dumpattrs = attrs2
# with open("dumped_spectra{}.pkl".format(mass._version.__version__),"w") as f:
#     d={k:v for k,v in mass.fluorescence_lines.__dict__.items()
#         if k.endswith("KBeta") or k.endswith("KAlpha")}
#     d2 = {}
#     for k,v in d.items():
#         for attr in dumpattrs:
#             d2[k+attr]=getattr(v,attr)
#     d2["massversion"]=mass._version.__version__
#     pickle.dump(d2,f)
# print("dumped version {}".format(mass._version.__version__))

with open("dumped_spectra0.6.2.pkl","r") as f:
    d1 = pickle.load(f)
with open("dumped_spectra0.6.3.pkl","r") as f:
    d2 = pickle.load(f)

wrongs = {}
for (k1,v1) in d1.items():
    if k1 == "massversion":
        continue
    s="KAlpha"
    i =  k1.find("KAlpha")
    if i<=0:
        s = "KBeta"
        i = k1.find("KBeta")
    k = k1[:i+len(s)]
    k1b = k1[i+len(s):]
    k2b = attrs2[attrs1.index(k1b)]
    k2 = k+k2b
    v2 = d2[k2]
    result = True
    try:
        result = all(np.abs(np.array(v1)-np.array(v2))<1e-6)
    except:
        result = False
    if not result:
        # print("old: {}\n{}\nnew: {}\n{}".format(k1,v1,k2,v2))
        wrongs[k] = wrongs.get(k,"")+"old: {}\n{}\nnew: {}\n{}]\n".format(k1,v1,k2,v2)

for kw,vw in wrongs.items():
    print(kw)
    print vw
