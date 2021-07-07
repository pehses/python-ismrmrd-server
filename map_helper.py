"""
functions for mapping algorithmen
"""
import numpy as np

def B1Map_DREAM(Iste, Ifid):
    """ Calculates the B1 Map using the DREAM approach

    :param Iste: STE image (magnitude)
    :param Ifid: FID image (magnitude)
    :returns: Map of the flip angle alpha (STEAM prep)
    """
    fa_map = np.rad2deg(np.arctan(np.sqrt(2*abs(Iste/Ifid))))
    
    return fa_map
