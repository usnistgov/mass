"""
nist_xray_database

Download the NIST x-ray line database from the website, and parse the
downloaded data into useable form.

For loading a file (locally, from disk) and plotting some information:
* NISTXrayDBFile
* plot_line_uncertainties

For updating the data files:
* NISTXrayDBRetrieve
* GetAllLines

Basic usage (assuming you put the x-ray files in
${MASS_HOME}/mass/calibration/nist_xray_data.dat):


J. Fowler, NIST
February 2014
"""

ELEMENTS = ('', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
            'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
            'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
            'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
            'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
            'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
            'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
            'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr')
ATOMIC_NUMBERS = dict((ELEMENTS[i], i) for i in range(len(ELEMENTS)))


class NISTXrayDBFile:
    DEFAULT_FILENAMES = "nist_xray_data.dat", "low_z_xray_data.dat"

    def __init__(self, *filenames):
        """Initialize the database from 1 or more <filenames>, which point to
        files downloaded using NISTXrayDBRetrieve. If the list is empty (the
        default), then the file named by self.DEFAULT_FILENAME will be used."""

        self.lines = {}
        self.alllines = set()

        import os
        if not filenames:
            path = os.path.split(__file__)[0]
            filenames = [os.path.join(path, df) for df in self.DEFAULT_FILENAMES]

        self.loaded_filenames = []
        for filename in filenames:
            try:
                fp = open(filename, "r")
            except OSError:
                print("'%s' is not a readable file with X-ray database info! Continuing..." % filename)
                continue

            while True:
                line = fp.readline()
                if "Theory" in line and "Blend" in line and "Ref." in line:
                    break

            for textline in fp.readlines():
                try:
                    xrayline = NISTXrayLine(textline)
                    self.lines[xrayline.name] = xrayline
                    self.alllines.add(xrayline)
                except Exception:
                    continue

            self.loaded_filenames.append(filename)
            fp.close()

    LINE_NICKNAMES = {
        'KA1': 'KL3',
        'KA2': 'KL2',
        'KB1': 'KM3',
        'KB3': 'KM2',
        'KB5': 'KM5',
        'LA1': 'L3M5',
        'LA2': 'L3M4',
        'Ll': 'L3M1',
        'LB3': 'L1M3',
        'LB1': 'L2M4',
        'LB2': 'L3N5',
        'LG1': 'L2N4',
    }

    def get_lines_by_type(self, linetype):
        """Return a tuple containing all lines of a certain type, e.g., "KL3".
        See self.LINE_NICKNAMES for some known line "nicknames"."""
        linetype = linetype.upper()
        if "ALPHA" in linetype:
            linetype = linetype.replace("ALPHA", "A")
        elif "BETA" in linetype:
            linetype = linetype.replace("BETA", "B")
        elif "GAMMA" in linetype:
            linetype = linetype.replace("GAMMA", "G")
        linetype = self.LINE_NICKNAMES.get(linetype, linetype)
        lines = []
        for element in ELEMENTS:
            linename = f'{element} {linetype}'
            if linename in self.lines:
                lines.append(self.lines[linename])
        return tuple(lines)

    def __getitem__(self, key):
        element, line = key.split()[:2]
        element = element.capitalize()
        line = line.upper()
        key = f'{element} {line}'
        if key in self.lines:
            return self.lines[key]
        lcline = line.lower()
        lcline = lcline.replace('alpha', 'a')
        lcline = lcline.replace('beta', 'b')
        lcline = lcline.replace('gamma', 'g')
        if lcline in self.LINE_NICKNAMES:
            key = f"{element} {self.LINE_NICKNAMES[lcline]}"
            return self.lines[key]
        raise KeyError("%s is not a known line or line nickname" % key)


class NISTXrayLine:
    DEFAULT_COLUMN_DEFS = {'element': (1, 4),
                           'transition': (10, 16),
                           'peak': (45, 59),
                           'peak_unc': (61, 72),
                           'blend': (74, 79),
                           'ref': (81, 91)}

    def __init__(self, textline, column_defs=None):
        if column_defs is None:
            column_defs = self.DEFAULT_COLUMN_DEFS
        for name, colrange in column_defs.items():
            a = colrange[0]-1
            b = colrange[1]
            self.__dict__[name] = textline[a:b].rstrip()
        self.peak = float(self.peak)
        self.peak_unc = float(self.peak_unc)
        self.name = f'{self.element} {self.transition}'
        self.raw = textline.rstrip()

    def __str__(self):
        return '{} {} line: {:.3f} +- {:.3f} eV'.format(self.element, self.transition,
                                                self.peak, self.peak_unc)

    def __repr__(self):
        return self.raw


def plot_line_uncertainties():
    import pylab as plt
    db = NISTXrayDBFile()
    transitions = ('KL3', 'KL2', 'KM3', 'KM5', 'L3M5', 'L3M4', 'L2M4', 'L3N5', 'L2N4', 'L1M3', 'L3N7', 'L3M1')
    titles = {
        'KL3': 'K$\\alpha_1$: Intense',
        'KL2': 'K$\\alpha_2$: Intense, but not easily resolved',
        'KM3': 'K$\\beta_1$: Intense',
        'KM2': 'K$\\beta_3$: Intense, usually unresolvable',
        'KM5': 'K$\\beta_5$: Weak line on high-E tail of K$\\beta_1$',
        'L3M5': 'L$\\alpha_1$: Prominent',
        'L3M4': 'L$\\alpha_2$: Small satellite',
        'L2M4': 'L$\\beta_1$: Prominent',
        'L3N5': 'L$\\beta_2$: Prominent',
        'L2N4': 'K$\\gamma_1$: Weaker',
        'L1M3': 'L$\\beta_3$: Weaker',
        'L3N7': 'Lu: barely visible',
        'L3M1': 'L$\\ell$: very weak',
    }

    axes = {}
    NX, NY = 3, 4
    plt.clf()
    for i, tr in enumerate(transitions):
        axes[i] = plt.subplot(NY, NX, i+1)
        plt.loglog()
        plt.grid(True)
        plt.title(titles[tr])
        if i >= NX*(NY-1):
            plt.xlabel("Line energy (eV)")
        if i % NX == 0:
            plt.ylabel("Line uncertainty (eV)")
        plt.ylim([1e-3, 10])
        plt.xlim([100, 3e4])

    for line in db.lines.values():
        if line.transition not in transitions:
            continue
        i = transitions.index(line.transition)
        plt.sca(axes[i])
        plt.plot(line.peak, line.peak_unc, 'or')
        plt.text(line.peak, line.peak_unc, line.name)


def plot_line_energies():
    db = NISTXrayDBFile()
    import pylab as plt
    plt.clf()
    cm = plt.cm.nipy_spectral
    transitions = ('KL2', 'KL3', 'KM5', 'KM3', 'KM2', 'L3M5', 'L3M4', 'L3M1', 'L2M4', 'L2N4', 'L3N5',
                   'L1M3', 'L3N7', 'M5N7', 'M5N6', 'M4N6', 'M3N5', 'M3N4')
    for i, linetype in enumerate(transitions):
        lines = db.get_lines_by_type(linetype)
        z = [ATOMIC_NUMBERS[line.element] for line in lines]
        e = [line.peak for line in lines]
        plt.loglog(z, e, 'o-', color=cm(float(i)/len(transitions)), label=linetype)
    plt.legend(loc='upper left')
    plt.xlim([6, 100])
    plt.grid()
    r = list(range(6, 22)) + list(range(22, 43, 2)) + list(range(45, 75, 3)) + list(range(75, 100, 5))
    plt.xticks(r, ['\n'.join([ELEMENTS[i], str(i)]) for i in r])


###############################################################
# Below here are functions to recreate the nist_xray_data.dat
# file, which I don't think anyone will ever need again.
# J Fowler, Feb 28, 2014.
###############################################################


def _NISTXrayDBRetrieve(line_names, savefile, min_E=150, max_E=25000):
    """Use this for updating the database file. You should not
    ever (?) need to do this, but who knows?"""
    form = "http://physics.nist.gov/cgi-bin/XrayTrans/search.pl?"
    args = {'download': 'column',
            'element': 'All',
            'units': 'eV',
            'lower': str(min_E),
            'upper': str(max_E)}
    joined_args = '&'.join([f'{k}={v}' for (k, v) in args.items()])
    joined_lines = '&'.join(['trans=%s' % name for name in line_names])
    get = f'{form}{joined_args}&{joined_lines}'
    print('Grabbing %s' % get)

    import urllib
    page = urllib.urlopen(get)
    fp = open(savefile, "w")
    fp.writelines(page)
    fp.close()


def _RetrieveAllLines(savefile, max_E=30000):
    """Use this for updating the database file. You should not
    ever (?) need to do this, but who knows?"""
    lines = ('KL2', 'KL3', 'KM5', 'KM3', 'KM2', 'L3M5', 'L3M4', 'L3M1', 'L2M4', 'L2N4', 'L3N5',
             'L1M3', 'L3N7')
    _NISTXrayDBRetrieve(lines, savefile, max_E=max_E)
