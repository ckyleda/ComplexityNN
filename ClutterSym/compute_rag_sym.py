from skimage import segmentation, graph
import cv2
import numpy as np
from ClutterSym.Symmetry import plot_patch_symmetry_scores
import json
import pathlib

# Load configuration
with open(pathlib.Path(__file__).parent / "configuration.json") as configfile:
    CONFIG = json.load(configfile)

# Fix random seed
np.random.seed(42)


def compute_rag_output_image(image):
    from skimage import color
    labels1 = segmentation.slic(image, compactness=CONFIG['compactness'], n_segments=CONFIG['n_segments'], start_label=1)
    g = graph.rag_mean_color(image, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, g)
    out2 = color.label2rgb(labels2, image, kind='avg', bg_label=0)

    histogram = np.histogram(labels2, bins=128)[0]
    rag_measure = len(histogram[histogram > 0])

    return out2, rag_measure


def output_maps(file):
    file = str(file)
    image = cv2.imread(file)

    rag_image, rag_score = compute_rag_output_image(image)

    l_score = plot_patch_symmetry_scores(file, window_size=CONFIG['large_window_size'],
                                         step_size=CONFIG['large_step_size'], pad_image=True, store_single=True)
    m_score = plot_patch_symmetry_scores(file, window_size=CONFIG['med_window_size'],
                                         step_size=CONFIG['med_step_size'], pad_image=True, store_single=True)
    s_score = plot_patch_symmetry_scores(file, window_size=CONFIG['small_window_size'],
                                         step_size=CONFIG['small_step_size'], pad_image=True, store_single=True)

    return rag_image, rag_score, np.mean([l_score, m_score, s_score], axis=None)


def clutter_sym_worker(file):
    try:
        _, rag_score, sym_score = output_maps(file)
        return {'status': 1, 'filename': file.name, 'clutter': rag_score, 'sym': sym_score}
    except Exception as e:
        print("MP Worker failure!")
        print(file)
        print(e)
        return {'status': 0, 'filename': file.name, 'clutter': None, 'sym': None}



