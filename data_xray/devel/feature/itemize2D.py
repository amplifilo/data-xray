from data_xray.modules import *
from .core2D import VoronoiTesselation, PolygonMaskExtract
#pts refers to the lattice of object centroids (derived from the image, overlaid or otherwise)


def MakeLibrary(im4, pts, width=30, use_trimmed=1):
    """
    a couple of choices here. We can use the complete raw data image behind the polygon, or
    use only the trimmed version, that follows the Voronoi polygon. Just use whichever is better with a flage
    width controls the size of the final image
    zoom functionality removed to the scaling of the original image


    :param im4:
    :param pts:
    :param width:
    :param use_trimmed:
    :return:
    """

    image_library = []
    region_id = []
    fill_constant_value=0
    regions, vertices = VoronoiTesselation(pts)
    for ir, r in enumerate(regions):

        masked, trimmed = PolygonMaskExtract(im4, polygon=vertices[r], plotit=0)
        try:
            if 0.85 < trimmed.shape[0]/trimmed.shape[1] < 1.15: #this ensures that we are dealing with square images
                trimmed.set_fill_value(fill_constant_value)

                if use_trimmed:
                    _face = trimmed.filled()
                else:
                    _face = trimmed.data
                _newface = np.pad(_face, pad_width=int(np.ceil((width - _face.shape[0]) / 2)), mode='constant',
                              constant_values=fill_constant_value)


                image_library.append(_newface)
                region_id.append(ir)

        except:
            print(ir, ' seems problematic')

    return image_library, region_id