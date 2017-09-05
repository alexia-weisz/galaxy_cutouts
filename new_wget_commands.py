import numpy as np

infile = '/Users/alexialewis/research/galbase/code/adam/MyAllSkyTable_akleroy.csv'
newcommandfile = '/Users/alexialewis/research/galbase/new_wget_commands.txt'


testfiles = ['AIS_183_sg55-fd', 'AIS_183_sg65-fd', 'AIS_183_sg66-fd', 'AIS_183_sg74-fd', 'AIS_183_sg75-fd', 'AIS_183_sg84-fd', 'GI1_047008_UGC01176-fd', 'GI3_050001_NGC628-fd', 'MISDR2_17173_0426_css7661-fd', 'MISDR2_17173_0426-fd', 'NGA_NGC0628-fd']


f = open(infile, 'r')
lines = f.readlines()
f.close()

goodlines = lines[1:]
allfiles = [goodlines[i].split(' ')[-1].strip('"').strip('\n').split('/')[-1].strip('"').split('.')[0].rstrip('-flags').rstrip('-int') for i in range(len(goodlines))]

blah = np.where(np.in1d(allfiles, testfiles))
names = np.asarray(allfiles)[blah[0]]


from collections import defaultdict
D = defaultdict(list)
for i,item in enumerate(names):
    D[item].append(i)
D = {k:v for k,v in D.items() if len(v)>1}

inds = []
for i in range(len(names)):
    inds.append(D[names[i]][0])

unique_inds = np.unique(inds)
goodinds = blah[0][unique_inds]

filestarts = np.asarray(goodlines)[goodinds]
filestarts = [f.replace('-int', '-flags')for f in filestarts]

newlines1 = [f.replace('-flags', '-cnt')for f in filestarts]
newlines2 = [f.replace('-flags', '-exp')for f in filestarts]
newlines3 = [f.replace('-flags', '-intbgsub')for f in filestarts]
newlines4 = [f.replace('-flags', '-wt')for f in filestarts]

g = open(newcommandfile, 'w')
for nl in [newlines1, newlines2, newlines3, newlines4]:
    g.writelines(nl)

g.close()
