import numpy as np
from scipy.spatial.distance import cdist
from skimage import measure
from skimage.filters import threshold_otsu

# compute centroids
def get_centroids(all_roi):
    centroids = []
    for i in range(all_roi.shape[2]):
        roi = all_roi[:,:,i]
        labels = measure.label(roi)
        features = measure.regionprops(labels)
        try:
            centroids.append(features[0].centroid)
        except IndexError:
            centroids.append([0, 0])

    return np.array(centroids)

def get_cent_dist_mat(all_roi_ref, all_roi_reg):
    # compute distances
    centroids_ref = get_centroids(all_roi_ref)
    centroids_reg = get_centroids(all_roi_reg)
    distances = cdist(centroids_ref, centroids_reg)
    return distances

def filt_non_overlap(all_roi1, all_roi2, cent_dist_mat):
    all_inds = np.arange(all_roi1.shape[2])
    filt_inds_ref = []
    for i in range(all_roi1.shape[2]):
        # get the index of closest roi from the distances matrix
        roi = all_roi1[:,:,i]
        closest_roi_idx = np.argmin(cent_dist_mat[i,:])
        closest_roi = all_roi2[:,:,closest_roi_idx]
        intersection = roi*closest_roi
        if np.sum(intersection)==0: # TODO:maybe it overlaps a bit with non-first closest roi?
            filt_inds_ref.append(i)

    # get all indices that are not in filt_inds_ref
    all_inds_filt = all_inds[~np.isin(all_inds, filt_inds_ref)]  
    return all_inds_filt

def get_cent_dist_mat_non_overlap(all_roi_ref, all_roi_reg):

    cent_dist_mat = get_cent_dist_mat(all_roi_ref, all_roi_reg)
    
    # get indices of neuorns that overlap with their closest neighbor (at least will have some chance to match)
    all_inds_ref_filt = filt_non_overlap(all_roi_ref, all_roi_reg, cent_dist_mat)
    all_inds_reg_filt = filt_non_overlap(all_roi_reg, all_roi_ref, cent_dist_mat.T)

    cost_mat = cent_dist_mat[all_inds_ref_filt, :]
    cost_mat = cost_mat[:, all_inds_reg_filt]

    return cost_mat, all_inds_ref_filt, all_inds_reg_filt

def get_cost_mat(all_roi_ref, all_roi_reg, track_ops):
    # compute distances
    if track_ops.matching_method=='cent': # simple assignment based on centroids (probably works with sparse data but not in development)
        cost_mat = get_cent_dist_mat(all_roi_ref, all_roi_reg)
        all_inds_ref_filt = np.arange(all_roi_ref.shape[2]) # here we don't filter the indices
        all_inds_reg_filt = np.arange(all_roi_reg.shape[2]) # here we don't filter the indices
    elif track_ops.matching_method=='cent_int-filt': # here to simplify the matching we first filter out the ROIs that have no intersection with their closest neighbor
        # this also outputs the indices of neurons after filtering
        # costa mat here is smaller than above since the matches are filtered
        # additionally when outputting we need to be careful to index with the filtered indices as well
        cost_mat, all_inds_ref_filt, all_inds_reg_filt = get_cent_dist_mat_non_overlap(all_roi_ref, all_roi_reg) 
    elif track_ops.matching_method=='iou':
        cost_mat = 1-get_cross_iou_mat(all_roi_ref, all_roi_reg, dist_thr=track_ops.iou_dist_thr)
        all_inds_ref_filt = np.arange(all_roi_ref.shape[2])
        all_inds_reg_filt = np.arange(all_roi_reg.shape[2])
    else:
        raise Exception('Matching method not implemented')
    
    print(f'cost_mat computed with method: {track_ops.matching_method}')
    print(f'cost_mat shape: {cost_mat.shape}')
    print(f'cost_mat min: {np.min(cost_mat)}')
    print(f'cost_mat max: {np.max(cost_mat)}')
    
    return cost_mat, all_inds_ref_filt, all_inds_reg_filt

def get_iou(all_roi_ref, all_roi_reg):

    ious = []
    for i in range(all_roi_ref.shape[2]):
        roi_ref = all_roi_ref[:,:,i]
        roi_reg = all_roi_reg[:,:,i]
        intersection = np.sum(np.logical_and(roi_ref, roi_reg))
        union = np.sum(np.logical_or(roi_ref, roi_reg))
        ious.append(intersection/union)

    return np.array(ious)

def get_cross_iou_mat(all_roi_ref, all_roi_reg, dist_thr=16):
    # if the distance between two rois is larger than dist_thr, we assume they are not the same cell and just skip the computation
    distances = get_cent_dist_mat(all_roi_ref, all_roi_reg)

    cross_iou_mat = np.zeros((all_roi_ref.shape[2], all_roi_reg.shape[2]))
    for i in range(all_roi_ref.shape[2]):
        for j in range(all_roi_reg.shape[2]):
            if distances[i,j] > dist_thr: # skipping if far apart
                continue
            # compute IOU
            intersection = np.logical_and(all_roi_ref[:,:,i], all_roi_reg[:,:,j])
            union = np.logical_or(all_roi_ref[:,:,i], all_roi_reg[:,:,j])
            iou_score = np.sum(intersection) / np.sum(union)
            cross_iou_mat[i,j] = iou_score

    return cross_iou_mat

def init_all_pl_match_mat(all_ds_all_roi_ref, all_ds_assign_thr, track_ops):
    all_pl_match_mat = []
    for i in range(track_ops.nplanes):
        # set up the match matrix for each plane it will be size of all iscell ROIs in the ref recording x number of datasets
        pl_match_mat = np.full((all_ds_all_roi_ref[0][i].shape[2], len(track_ops.all_ds_path)), None)
        all_pl_match_mat.append(pl_match_mat)
    # BUG FIX 2026-07-29 (see track2p-tracking-fix/SESSION_LOG.md, ghost-row
    # investigation on wehr5336): every row here already IS one of session 0's
    # own iscell-filtered ROIs -- that's what pl_match_mat.shape[0] above is
    # built from -- so each row's own session-0 self-identity
    # (pl_match_mat[roi_idx, 0] == roi_idx) is a trivial fact, not something
    # that needs to be earned by matching. The previous version only recorded
    # this self-identity for the SUBSET of ROIs that ALSO happened to match
    # forward into session 1 (all_ds_assign_thr[0][i]), conflating "this ROI
    # exists in session 0" with "this ROI's session0->session1 transition
    # succeeded". Any ROI that failed that one specific transition was left
    # with pl_match_mat[roi_idx, 0] == None -- and this package's own
    # get_all_pl_match_mat (match/loop.py) as well as track2p-tracking-fix's
    # gap-tolerant get_all_pl_match_mat_gap both start their per-ROI loop with
    # "if pl_match_mat[roi_idx, 0] is None: continue", so a None entry here
    # means the ROI is skipped entirely -- including by gap-tolerant chaining,
    # whose whole purpose is to rescue exactly this case (retry i->i+2,
    # i->i+3, ... after i->i+1 fails). So this line was silently preventing
    # gap-tolerant chaining from ever being attempted on ~half of session 0's
    # cells (558/1078 in the wehr5336 case), not because chaining failed on
    # them, but because they never got the chance to be tried.
    for i in range(track_ops.nplanes):
        pl_match_mat = all_pl_match_mat[i]
        pl_match_mat[:, 0] = np.arange(pl_match_mat.shape[0])

    return all_pl_match_mat

def filt_by_otsu(vect_filt, vect_comp):
    # vect_filt is the vector that we want to filter
    # vect_comp is the vector that we want to compute the otsu threshold on

    thresh = threshold_otsu(vect_comp)
    return vect_filt[vect_comp>thresh]