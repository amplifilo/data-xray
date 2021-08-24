#################################
#############DAT FILES
############################

from ..modules import *

def PlotSpectrum(spec, f_r='f', fig=[], ax=[], chans='fil', phdf='', ppl=1):
    # the plotter! plots all channels contained in the sxm-dict structure
    import matplotlib.gridspec as gridspec
    from matplotlib import pyplot as plt
    from matplotlib.figure import Figure

    def primeChans(datchans, mods):
        primechans = datchans
        for m in mods:
            modc = [j for j in datchans if len(re.findall(m, j))]
            primechans = np.setxor1d(primechans, modc)
        return primechans

    dat = spec['data']
    fn = spec['file']
    chans = spec['chans']

    datchans = [i for i, j in chans.items() if j != 0]
    xvec = dat[:, chans[np.setxor1d(datchans, list(chans.keys()))[0]]]

    plotdict = dict()
    for dk in np.sort(primeChans(datchans, ['AVG', 'bwd', 'fil'])):

        yvec = dat[:, chans[dk]]
        ynorm = yvec / np.mean(yvec[~np.isnan(yvec)])
        yclean = yvec[~np.isnan(yvec)]

        stat = np.mean([np.std(yclean / np.max(yclean)), np.std(yclean / np.min(yclean))])
        # stat = [np.mean(yvec[~np.isnan(yvec)]), np.std(yvec[~np.isnan(yvec)])]
        # print(dk + 'mean:' + str(stat[0]) + ' ' + 'std:' + str(stat[1]))

        if stat > 0.2:
            if dk + '_bwd' in datchans:
                plotdict[dk] = [yvec, dat[:, chans[dk + '_bwd']]]
            else:
                plotdict[dk] = [yvec]

    if len(plotdict) > 0:
        if ppl:
            fig = plt.figure()
        else:
            fig = Figure()
        plotrows = np.ceil(len(plotdict) / 3.0)
        if plotrows > 3:
            gs = gridspec.GridSpec(3, np.int(plotrows))
        else:
            gs = gridspec.GridSpec(1, len(plotdict))

        cind = 0
        for j in np.sort(list(plotdict.keys())):
            if cind < len(plotdict):
                print(cind)
                ax = fig.add_subplot(gs[cind])  ######this needs to be rephrased
                for pl in plotdict[j]:
                    ax.plot(xvec, pl)
                ax.set_title(j)
                # ax.set_title(phdf+j)
                cind += 1

    if not (ppl):
        return [fig, fn]

