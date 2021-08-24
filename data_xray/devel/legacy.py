from ..modules import *
from pyearth import Earth

def get_slope(x,y, thresh=None):
#find a log-log slope using pyearth

    m = Earth()
    npl = lambda x: np.real(np.log10(x))
    xclean = np.arange(np.size(x))[~np.isinf(npl(x))]
    yclean = np.arange(np.size(y))

    if type(thresh) is float:
        yclean = np.where(npl(y)>thresh)

    fitind = np.intersect1d(xclean,yclean)
    nx = npl(x)[fitind] #take care of that np.log(0)
    ny = npl(y)[fitind]
    m.fit(nx, ny)
    yh = m.predict(nx)
    dyh = np.abs(np.gradient(yh)/np.gradient(nx))
    figs, axfit =plt.subplots()
    axfit.plot(npl(x),npl(y),marker="o", fillstyle='none', color='navy', linestyle='none', markersize=4)
    axfit.plot(nx,yh,'r-',linewidth=2)
    axfit.set_xlabel(r'$log(bias)$')
    axfit.set_ylabel(r'$log(current)$')
    for tl in axfit.get_yticklabels():
        tl.set_color('navy')
    axslop = axfit.twinx()
    axslop.plot(nx,np.gradient(yh)/np.gradient(nx),'crimson',linewidth=0.6)
    axslop.set_ylabel(r'$log-log \ slope$')
    for tl in axslop.get_yticklabels():
        tl.set_color('crimson')
    plt.show()
    return figs




################
def earth_slope(xvec,yvec, p, incrj=0):
#get slopes of a Z-V curve
    m = Earth()
    ### comment: noise subtraction carried out for all IVs
    # m.fit(xvec,yvec)
    # y_hat = m.predict(xvec)
    # datind = np.where(abs(np.gradient(y_hat))>1e-4)
    # nind = np.where(abs(np.gradient(y_hat))<=1e-4)
    # yvec = abs(yvec - np.mean(yvec[nind]))
    # sigind = np.where(np.log10(np.abs(yvec))>self.allnoise)
    # vpos = np.where(xvec[sigind]>0)
    # vneg = np.where(xvec[sigind]<0)
    sigind = np.where(np.log10(np.abs(yvec))>p['thresh']*self.allnoise)
    vpos = np.intersect1d(sigind,np.where(xvec>0))
    vneg = np.intersect1d(sigind, np.where(xvec<0))

    posx, posc, negx, negc = np.abs(xvec[vpos]),np.abs(yvec[vpos]), np.abs(xvec[vneg]), np.abs(yvec[vneg])
    tl = np.zeros(xvec.shape)
    l1,l2 = [],[]
    rsq = []

    if len(posc)>4: #a meaningful length of iv curve
        m.fit(np.log(posx),np.log(posc))
        l1 = m.predict(np.log(posx))
        tl[vpos] = np.gradient(l1)/np.gradient(np.log(posx))
        rsq.append(m.rsq_)
        if p['plotfit']:
            p['plotax'].plot(np.log(posx),incrj*p['incr']+l1, linewidth=0.4, color='red')
            p['plotax'].scatter(np.log(posx), incrj*p['incr']+np.log(posc),  s=10, facecolors='none', edgecolors='blue')

    if len(negc)>4:
        m.fit(np.log(negx),np.log(negc))
        l2 = m.predict(np.log(negx))
        tl[vneg] = np.gradient(l2)/np.gradient(np.log(negx))
        rsq.append(m.rsq_)
        if p['plotfit']:
            p['plotax'].plot(np.log(negx),incrj*p['incr'] + l2,linewidth=0.4, color='red')
            p['plotax'].scatter(np.log(negx),incrj*p['incr'] + np.log(negc), s=10, facecolors='none', edgecolors='blue')

    return tl, np.mean(rsq)


def sub_iv_noise(y,x=[], rng=[]):
#subtract noise from iv data
    if not(len(x)):
        x = np.arange(len(y))
    if not(len(rng)):
        m = Earth()
        m.fit(x,y)
        y_hat = m.predict(x)
        datind = np.where(abs(np.gradient(y_hat))>1e-1)
        nind = np.where(abs(np.gradient(y_hat))<=1e-1)
    else:
        nind1 = np.where(x>np.min(rng))[0]
        nind2 = np.where(x<np.max(rng))[0]
        nind = np.intersect1d(nind1,nind2)
    y = y - np.mean((y[nind]))
    return y, np.mean(np.abs(y[nind]))

