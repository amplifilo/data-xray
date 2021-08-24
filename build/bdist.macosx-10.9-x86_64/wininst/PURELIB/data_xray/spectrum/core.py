from ..modules import *

def EarthSlope(x=None,y=None, thresh=1e-3, max_terms=None):

    if max_terms is None:
        mfull = Earth(thresh=thresh)
    else:
        mfull = Earth(thresh=thresh, max_terms=max_terms)

    #y = dat.dropna()

    if x is None:
        x = np.arange(len(y))
    mfull.fit(x, y)
    return mfull.predict(x)


def WaveletFilter(dat, l=10, wlet='db16'):
    from statsmodels.robust import mad
    ###truncate dataset to 2**n
    # maxl = int(np.floor(np.log2(len(pd_src))))

    md = 'sym'

    maxl = pywt.dwt_max_level(len(dat), pywt.Wavelet(wlet))
    # print(pywt.swt_max_level(len(dat)))
    noisy_coefs = pywt.wavedec(dat, wlet, level=np.min([l, maxl]), mode=md)
    sigma = mad(noisy_coefs[-1])
    # pdb.set_trace()

    uthresh = sigma * np.sqrt(2 * np.log(len(dat)))
    denoised = noisy_coefs[:]
    denoised[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in denoised[1:])
    signal = pywt.waverec(denoised, wlet, mode=md)

    return signal
