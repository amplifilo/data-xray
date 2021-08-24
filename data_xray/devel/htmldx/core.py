from data_xray.modules import *
################################################################
############## create HTML tree ################################
################################################################

# def Sxm2hdf(topdir=None, saveh5='images'):
#     #_fld = '/media/peter/1246A17C46A16169/USBBackup/MighuData/phenylacetylene under LN2/2011-02-11_surfaceprep'
#     from nanonisfiles_io import Scan
#     import deepdish as dd
#     if topdir is not None:
#         sxms = crawldir(ext='sxm', topdir= topdir)
#
#         lst = []
#         for f in sxms:
#             b = Scan(f)
#             b._to_dataset()
#             lst.append(b.ds)
#
#         dd.io.save(topdir+'/'+saveh5 + '.h5', {'ds':lst})
#


def sxm2html(topdir=[], saveh5='images'):
    from nanonis_io import NanonisFile
    from nanonis_plotutils import plot_sxm_chan
    from matplotlib_scalebar.scalebar import ScaleBar
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar)
#create sxm-dict structure and stick it into hdf file. returns sxmdict

    chan = 0
    sxmdict = dict()
    if topdir == []:
         fn = crawldir('sxm')
    else:
         fn = crawldir(topdir=topdir)

    for dirr in tqdm(fn.keys()):
        #filelist = list()
        #print(dirr)

        doc, tag, text = Doc().tagtext()
        doc.asis('<!DOCTYPE html>')
        with tag('html'):
            with tag('title'):
                doc.text(dirr)
            with tag('body'):

                for idnum, fl in enumerate(fn[dirr]):
                    try:
                        #c1 = sxm2dict(fl)
                        print(fl)
                        c1 = NanonisFile(fl)
                        #c1 = NanonisFile(fl)
                        c1 = c1.sxmdict
                        if len(c1):
                            try:
                                print('...processing:' + fl)
                                with tag('div', id='layer' + str(idnum)+'comment'):
                                    doc.text(re.sub(',\n', '::', c1['params']['comment']))
                                    doc.asis('<br><br>')
                                    relname = '.' + re.split(os.path.commonprefix([topdir,fl]), fl)[1]
                                    doc.text("plot_sxm_chan(sxm2dict('" + relname +"'), ppl=" + str(1) + "); plt.show()") #instead of relname use fl for absolute path

                                    with tag('div', id='layer' + str(idnum)):
                                        fig, _filename = plot_sxm_chan(c1, ppl=0)
                                        canv = FigureCanvas(fig)
                                        tmp_path = dirr + '/' + 't1.png'
                                        canv.print_figure(tmp_path, transparent=1, format='png', dpi=150)
                                        data_uri = base64.b64encode(open(tmp_path, 'rb').read()).decode('utf-8').replace('\n', '')
                                        img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
                                        doc.asis(img_tag)
                            except KeyError:
                                print('problem with ' + c1['file'])
                    except ValueError:
                        print('something is wrong with file ' + fl)
        Html_path = dirr + '/' + os.path.basename(dirr) + '.html'
        Html_file = open(Html_path, "w")
        Html_file.write(doc.getvalue())
        Html_file.close()



#####################summarize HTML####################
####################################################

def summarize_HTML(topdir=[], ext='sxm'):
    import shutil
    #directory crawler: return all files of a given extension
#in the current and all the nested directories
    summary = os.path.basename(topdir) + ".html"
    htmlfolder = topdir + '/' + os.path.basename(topdir) + '_tree/'
    htmlfiles = []
    if os.path.exists(htmlfolder):
        shutil.rmtree(htmlfolder)

    for root, dirs, files in os.walk(topdir):
        for name in files:
            if name.endswith('.html') and name != summary:
                relname = '.\\' + re.split(os.path.commonprefix([topdir, root]), root)[1]
                relname = os.path.join(relname, name)
                addname = os.path.join(root, name)
                htmlfiles.append(addname)

    ulopen = 0
    #

    f = open(topdir + "/" + summary, "w")
    f.write('<html><head></head><body><ul><p></p>')


    if not(os.path.exists(htmlfolder)):
         os.mkdir(htmlfolder)

    for fi in tqdm(htmlfiles):
        shutil.copy(os.path.abspath(fi), htmlfolder)
        lnk = htmlfolder + os.path.basename(fi)
        f.write('<li><a href="%s">%s</a></li>' % (lnk, fi))
                           # print(addname)
                        # else:
                        #     if ulopen:
                        #         f.write('</ul><ul title="%s">' % (root))
                        #         ulopen = 1
                        #     else:
                        #         f.write('<ul title="%s">' % (root))
                        #         ulopen = 1
                        #
                        #     f.write('<p style="color:green;"><strong>%s</strong></p>' % (root))
                        #     f.write('<li><a href="%s">%s</a></li>' % (relname, name)) #(addname, name))
    f.write('</ul></body></html>')


######### homecooked wrapper for oft-used matplotlib functions ################

class htmlPlot(object):
    #import mpld3
    import os
    def __init__(self, fromf=1, folderpath=[]):
        from collections import OrderedDict

        self.topdir = folderpath
        if not (len(folderpath)):
            self.topdir = os.getcwd()

        #f.write('<html><head></head><body><ul><p></p>')
        self.figpool = OrderedDict()

        #print(os.path.basename(self.topdir))

    def _addfig(self,fig):
        self.figpool[fig.canvas.get_window_title()] = fig

    def _publish_fig(self):
        self.f = open(self.topdir + "/" + os.path.basename(self.topdir) + ".html", "w")

        self.f.write('<html><body><ul>')

        for j in self.figpool.keys():
            self.f.write('<p><center><strong>')
            self.f.write(j + '</strong>')
            self.f.write(mpld3.fig_to_html(self.figpool[j], no_extras=True))
            self.f.write('</center></p>')

        self.f.write('</ul></body></html>')
        self.f.close()


def summarize_HTML2(topdir=[], ext='sxm'):
    import shutil
    #directory crawler: return all files of a given extension
#in the current and all the nested directories
    summary = os.path.basename(topdir) + ".html"
    htmlfolder = topdir + '/' + os.path.basename(topdir) + '_tree/'
    htmlfiles = []
    if os.path.exists(htmlfolder):
        shutil.rmtree(htmlfolder)

    for root, dirs, files in os.walk(topdir):
        for name in files:
            if name.endswith('.html') and name != summary:
                relname = '.\\' + re.split(os.path.commonprefix([topdir, root]), root)[1]
                relname = os.path.join(relname, name)
                addname = os.path.join(root, name)
                htmlfiles.append(addname)

    ulopen = 0
    #

    f = open(topdir + "/" + summary, "w")
    f.write('<html><head></head><body><ul><p></p>')


    if not(os.path.exists(htmlfolder)):
         os.mkdir(htmlfolder)

    for fi in tqdm(htmlfiles):
        shutil.copy(os.path.abspath(fi), htmlfolder)
        lnk = htmlfolder + os.path.basename(fi)
        f.write('<li><a href="%s">%s</a></li>' % (lnk, fi))
                           # print(addname)
                        # else:
                        #     if ulopen:
                        #         f.write('</ul><ul title="%s">' % (root))
                        #         ulopen = 1
                        #     else:
                        #         f.write('<ul title="%s">' % (root))
                        #         ulopen = 1
                        #
                        #     f.write('<p style="color:green;"><strong>%s</strong></p>' % (root))
                        #     f.write('<li><a href="%s">%s</a></li>' % (relname, name)) #(addname, name))

    f.write('</ul></body></html>')
