import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import numpy as np
import myutils
from myutils.array import make_imshowable as mims
import matplotlib.patches as patches

def area_vs_threshold(xs, psnr_map):
    areas = []
    for psnr in xs:
        thres = [1 if i > psnr else 0 for i in psnr_map]
        area = np.sum(thres) / len(thres) # Normalized
        # area = np.sum(thres)
        areas.append(area)

    return areas

def range_vs_threshold(xs, psnrs, recons):
    # Loop only one image at a time
    n_grid = int(np.sqrt(len(recons)))
    res = []
    for img_idx in range(recons.shape[1]):
        print(img_idx)
        ranges = []
        recons_per_img = recons[:,img_idx:img_idx+1,...]
        psnr_map = psnrs[:,img_idx:img_idx+1]
        # Compute metrics for single image
        for psnr in xs:
            thres = [1 if i > psnr else 0 for i in psnr_map]

            recons_above_thres = None
            psnrs_above_thres = []

            # thres = np.argwhere(psnr_map > psnr).flatten()
            # if len(thres) == 0:
            #     r = 0
            # Loop through all hp locations above threshold
            for i, t in enumerate(thres):
                if t == 1:
                    psnrs_above_thres.append(psnr_map[i])
                    if recons_above_thres is None:
                        recons_above_thres = recons_per_img[i]
                    else:
                        recons_above_thres = np.concatenate((recons_above_thres, recons_per_img[i]), axis=0)


            if recons_above_thres is None:
                r = 0
            else:
                # psnrs_above_thres = psnr_map[thres]
                # recons_above_thres = recons_per_img[thres]
                r_img = np.ptp(recons_above_thres, axis=0)
                r = np.sum(r_img)

            ranges.append(r)
        res.append(ranges)
    return np.array(res).mean(axis=0)

def variability_by_distance(xs, psnr_map, recons, dense_map, is_baseline=False):
    alphas = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.98, 0.99,0.995,0.999,1.0]
    base_locs = np.stack(np.meshgrid(alphas, alphas), -1).reshape(-1,2)

    # Loop only one image at a time
    n_grid = int(np.sqrt(len(recons)))
    recons_per_img = recons
    print('recons', recons_per_img.shape)
    # Compute metrics for single image
    for psnr in xs:
        # thres = [1 if i > psnr and i < psnr+0.3 else 0 for i in psnr_map]
        thres = [1 if i > psnr and i < psnr+0.5 else 0 for i in psnr_map]

        recons_above_thres = None
        psnrs_above_thres = []
        locs_above_thres = []
        # Loop through all hp locations above threshold
        for i, t in enumerate(thres):
            if t == 1:
                psnrs_above_thres.append(psnr_map[i])
                if is_baseline:
                    locs_above_thres.append(base_locs[i])
                else:
                    locs_above_thres.append((np.round(np.interp(i%n_grid, [0, n_grid-1], [0, 1]), 4), np.round(np.interp(i//n_grid, [0, n_grid-1], [0, 1]), 4)))
                if recons_above_thres is None:
                    recons_above_thres = recons_per_img[i]
                else:
                    recons_above_thres = np.concatenate((recons_above_thres, recons_per_img[i]), axis=0)


        print('recons', recons_above_thres.shape)
        if recons_above_thres is None:
            r = 0
        else:
            n_above_thres = len(recons_above_thres)
            r_img = np.ptp(recons_above_thres, axis=0)
            r = np.sum(r_img)

            select_psnrs = []
            reshaped = recons_above_thres.reshape(n_above_thres, -1)
            mdi = farthest_points(2, reshaped)
            recon1 = recons_above_thres[mdi[0]:mdi[0]+1]
            select_psnrs.append(psnrs_above_thres[mdi[0]])
            recon2 = recons_above_thres[mdi[1]:mdi[1]+1]
            select_psnrs.append(psnrs_above_thres[mdi[1]])
            recon3 = recons_above_thres[mdi[2]:mdi[2]+1]
            select_psnrs.append(psnrs_above_thres[mdi[2]])
            # recon4 = recons_above_thres[mdi[3]:mdi[3]+1]
            # psnr4 = psnrs_above_thres[mdi[3]]

            fig, axes = plt.subplots(1, 4, figsize=(16,6))
            fig.suptitle('Maximizing Distance, Threshold: ' + str(xs[0]), fontsize=24)

            ticks = np.linspace(0, 1, np.sqrt(len(dense_map)))
            myutils.plot.plot_over_hyperparams(dense_map, xticks=ticks, yticks=ticks, ax=axes[0], vlim=[-10, 5], all_ticks='ends')
            axes[0].text(locs_above_thres[mdi[0]][0], 1-locs_above_thres[mdi[0]][1], 'a)', color='black', fontsize=16,
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform = axes[0].transAxes)
            axes[0].text(locs_above_thres[mdi[1]][0], 1-locs_above_thres[mdi[1]][1], 'b)', color='black', fontsize=16,
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform = axes[0].transAxes)
            axes[0].text(locs_above_thres[mdi[2]][0], 1-locs_above_thres[mdi[2]][1], 'c)', color='black', fontsize=16,
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform = axes[0].transAxes)

            title = 'a) ' + str(locs_above_thres[mdi[0]])
            myutils.plot.plot_img(mims(recon1[0, 100:200, 100:200, :]), title=title, ax=axes[1], rot90=True)

            title = 'b) ' + str(locs_above_thres[mdi[1]])
            myutils.plot.plot_img(mims(recon2[0, 100:200, 100:200, :]), title=title, ax=axes[2], rot90=True)

            title = 'c) ' + str(locs_above_thres[mdi[2]])
            myutils.plot.plot_img(mims(recon3[0, 100:200, 100:200, :]), title=title, ax=axes[3], rot90=True)

            import matplotlib.patches as patches
            for j, ax in enumerate(axes.reshape(-1)): 
                if j == 0:
                    continue
                arrow = patches.Arrow(45, 35, 5, 5, width=5.0, color='r')
                ax.add_patch(arrow)
                ax.text(0.6, 0.05,'RPSNR=%.02f' % (select_psnrs[j-1]), color='white', fontsize=20,
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform = ax.transAxes)

def farthest_points(n_points, recons):
    n_above_thres = len(recons)
    if n_points == 2:
        dist_mat = squareform(pdist(recons, 'euclidean'))

        max_ind = np.argsort(dist_mat.flatten())
        mdi1 = np.array(np.unravel_index(max_ind[-1], (n_above_thres, n_above_thres)))
        i = -2
        mdi2 = np.array(np.unravel_index(max_ind[i], (n_above_thres, n_above_thres)))
        while np.array_equal(mdi2, mdi1) or np.array_equal(np.flip(mdi2), mdi1):
            i -= 1
            mdi2 = np.unravel_index(max_ind[i], (n_above_thres, n_above_thres))
        mdi = np.unique(np.concatenate((mdi1, mdi2)))
        print(mdi)
    elif n_points == 3:
        max_var = 0
        mdi = None
        for i in range(len(recons)):
            for j in range(i, len(recons)):
                for k in range(j, len(recons)):
                    if i != j and j != k:
                        print(i,j,k)
                        recon_group = recons[[i, j, k]]
                        var = np.var(recon_group)
                        if var > max_var:
                            print('found max')
                            max_var = var
                            print(var)
                            mdi = [i, j, k]
        
    return mdi

def variability_by_bestdc(psnr_map, dc_map, recons):
    n_grid = int(np.sqrt(len(recons)))
    dc_ranked_ind = np.argsort(dc_map)

    recons = recons[:,0,...]
    fig, axes = plt.subplots(2, 3, figsize=(20,10))
    fig.suptitle('Best DC, Coarse Sampling', fontsize=24)
    ticks = np.linspace(0, 1, n_grid)
    myutils.plot.plot_over_hyperparams(psnr_map, xticks=ticks, yticks=ticks, ax=axes[0,0], vlim=[-10, 5], all_ticks='ends')
    for i in range(1, 6):
        flat_ind = dc_ranked_ind[i]
        coord_ind = ((np.round(np.interp(flat_ind%n_grid, [0, n_grid-1], [0, 1]), 4), np.round(np.interp(flat_ind//n_grid, [0, n_grid-1], [0, 1]), 4)))

        axes[0,0].text(coord_ind[0], 1-coord_ind[1], 'a)', color='black', fontsize=16,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = axes[0,0].transAxes)

        title = 'a) ' + str(coord_ind)
        myutils.plot.plot_img(mims(recons[flat_ind, 100:200, 100:200, :]), title=title, ax=axes[i//3,i%3], rot90=True)

        import matplotlib.patches as patches
        arrow = patches.Arrow(45, 35, 5, 5, width=5.0, color='r')
        axes[i//3,i%3].add_patch(arrow)
        axes[i//3,i%3].text(0.7, 0.05,'RPSNR=%.02f' % (psnr_map[flat_ind]), color='white', fontsize=20,
             horizontalalignment='center',
             verticalalignment='center',
             transform = axes[i//3,i%3].transAxes)

def dc_ranked_img_plot(gt_data, psnr_map, dc_map, recons):
    n_grid = int(np.sqrt(len(recons)))
    n_chunks = 4

    fig, axes = plt.subplots(psnr_map.shape[1], n_chunks+1, figsize=(17,22))
    fig.tight_layout()
    plt.subplots_adjust(left=1, bottom=1, right=2, top=2, wspace=0, hspace=0)
    # fig.suptitle('Best DC, Coarse Sampling', fontsize=24)
    for img_idx in range(psnr_map.shape[1]):
        dc_img = dc_map[:,img_idx]
        psnr_img = psnr_map[:,img_idx]
        recon_img = recons[:,img_idx, ...]
        gt_img = gt_data[img_idx, ...]

        max_num = closestMultiple(int(len(recon_img)*0.25), n_chunks)
        # max_num = closestMultiple(int(len(recon_img)*0.5), n_chunks)
        dc_ranked_ind = np.argsort(dc_img)[:max_num] # Take top 25% of recons, ranked by DC
        sorted_psnrs = psnr_img[dc_ranked_ind]
        sorted_recons = recon_img[dc_ranked_ind]
        # sorted_gts = gt_img[dc_ranked_ind]

        myutils.plot.plot_img(mims(gt_img[100:200, 100:200, :]), ax=axes[img_idx,0], rot90=True)
        indices = get_highest_psnr_by_chunk(sorted_psnrs, n_chunks)
        for i, flat_ind in enumerate(indices):
            coord_ind = ((np.round(np.interp(flat_ind%n_grid, [0, n_grid-1], [0, 1]), 4), np.round(np.interp(flat_ind//n_grid, [0, n_grid-1], [0, 1]), 4)))

            # title = 'a) ' + str(coord_ind)
            myutils.plot.plot_img(mims(sorted_recons[flat_ind, 100:200, 100:200, :]), ax=axes[img_idx,i+1], rot90=True)

            import matplotlib.patches as patches
            arrow = patches.Arrow(45, 35, 5, 5, width=5.0, color='r')
            axes[img_idx,i+1].add_patch(arrow)
            axes[img_idx,i+1].text(0.7, 0.05,'RPSNR=%.02f' % (sorted_psnrs[flat_ind]), color='white', fontsize=20,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = axes[img_idx,i+1].transAxes)

# Python3 program to calculate  
# the smallest multiple of x  
# closest to a given number 
  
# Function to calculate 
# the smallest multiple 
def closestMultiple(n, x): 
    if x > n: 
        return x; 
    z = (int)(x / 2); 
    n = n + z; 
    n = n - (n % x); 
    return n; 

def get_highest_psnr_by_chunk(sorted_psnr, n_chunks):
    cols = len(sorted_psnr) // n_chunks
    sorted_psnr_by_chunks = sorted_psnr.reshape(n_chunks, cols)
    inds_2d = np.argsort(sorted_psnr_by_chunks, axis=1)
    return inds_2d[:, -1] + np.arange(n_chunks)*cols


def plot_by_frechet(gt_data, psnr_map, dc_map, recons):
    n_grid = int(np.sqrt(len(recons)))
    n_chunks = 3

    fig, axes = plt.subplots(psnr_map.shape[1]*2, n_chunks+1, figsize=(16,24))
    # fig.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    # fig.suptitle('Best DC, Coarse Sampling', fontsize=24)
    for img_idx in range(psnr_map.shape[1]):
        dc_img = dc_map[:,img_idx]
        psnr_img = psnr_map[:,img_idx]
        recon_img = recons[:,img_idx, ...]
        gt_img = gt_data[img_idx, ...]

        # # Take top 25% of recons, ranked by DC
        # max_num = int(len(recon_img)*0.45)
        # # max_num = int(len(recon_img))
        # dc_ranked_ind = np.argsort(dc_img)[:max_num]
        # sorted_psnrs = psnr_img[dc_ranked_ind]
        # sorted_recons = recon_img[dc_ranked_ind]

        # Take additional top 50% of recons, ranked by PSNR 
        thres = np.argwhere((psnr_img > 3.5) & (psnr_img < 4.5)).flatten()
        # psnr_ranked_ind = np.flip(np.argsort(sorted_psnrs))[:max_num]
        sorted_psnrs = psnr_img[thres]
        sorted_recons = recon_img[thres]


        if img_idx == 0:
            arrow = patches.Arrow(15, 55, 5, 5, width=5.0, color='r')
        if img_idx == 1:
            arrow = patches.Arrow(56, 33, -5, 5, width=5.0, color='r')
        if img_idx == 2:
            arrow = patches.Arrow(40, 55, 5, 5, width=5.0, color='r')
        myutils.plot.plot_img(mims(gt_img), ax=axes[2*img_idx,0], rot90=True)
        myutils.plot.plot_img(mims(gt_img[100:175, 100:175, :]), ax=axes[2*img_idx+1,0], rot90=True)
        rect = patches.Rectangle((85,100),75,75,linewidth=1,edgecolor='r',facecolor='none')
        axes[2*img_idx,0].add_patch(rect)
        # arrow = patches.Arrow(45, 35, 5, 5, width=5.0, color='r')
        axes[2*img_idx+1,0].add_patch(arrow)

        indices = get_indices_frechet(sorted_psnrs, sorted_recons)
        for i, flat_ind in enumerate(indices):
            coord_ind = ((np.round(np.interp(flat_ind%n_grid, [0, n_grid-1], [0, 1]), 4), np.round(np.interp(flat_ind//n_grid, [0, n_grid-1], [0, 1]), 4)))

            # title = 'a) ' + str(coord_ind)
            myutils.plot.plot_img(mims(sorted_recons[flat_ind]), ax=axes[2*img_idx,i+1], rot90=True)
            myutils.plot.plot_img(mims(sorted_recons[flat_ind, 100:175, 100:175, :]), ax=axes[2*img_idx+1,i+1], rot90=True)

            new_psnr = sorted_psnrs[flat_ind]

            if img_idx == 0:
                arrow = patches.Arrow(15, 55, 5, 5, width=5.0, color='r')
            if img_idx == 1:
                arrow = patches.Arrow(56, 33, -5, 5, width=5.0, color='r')
            if img_idx == 2:
                arrow = patches.Arrow(40, 55, 5, 5, width=5.0, color='r')

            axes[2*img_idx+1,i+1].add_patch(arrow)
            axes[2*img_idx,i+1].text(0.7, 0.05,'RPSNR=%.02f' % (new_psnr), color='white', fontsize=20,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = axes[2*img_idx,i+1].transAxes)
            # Create a Rectangle patch
            rect = patches.Rectangle((85,100),75,75,linewidth=1,edgecolor='r',facecolor='none')

            # Add the patch to the Axes
            axes[2*img_idx,i+1].add_patch(rect)

    fig.savefig('/nfs02/users/aw847/data/hypernet/figs/representative_slices.eps', format='eps')
    return fig

def get_indices_frechet(sorted_psnrs, sorted_recons):
    recons_reshaped = sorted_recons.reshape(len(sorted_recons), -1)
    # Create distance matrix for all recons
    dist_mat = squareform(pdist(recons_reshaped, 'euclidean'))
    # Find row with smallest summed distance
    ind1 = np.argmax(np.sum(dist_mat, axis=1))
    # Find 2nd index which is farthest away from first
    ind2 = np.argmax(dist_mat[ind1])
    # Find 3rd index which is farthest away from first two 
    ind3 = np.argmax(dist_mat[ind1]+dist_mat[ind2])
    return [ind1, ind2, ind3]


def plot_by_max_concentration(gt_data, psnr_map, dc_map, recons):
    n_grid = int(np.sqrt(len(recons)))
    n_chunks = 2

    fig, axes = plt.subplots(psnr_map.shape[1]*2, n_chunks+1, figsize=(16,24))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    for img_idx in range(psnr_map.shape[1]):
        dc_img = dc_map[:,img_idx]
        psnr_img = psnr_map[:,img_idx]
        recon_img = mims(recons[:,img_idx, ...])
        gt_img = gt_data[img_idx, ...]

        # Take additional top 50% of recons, ranked by PSNR 
        thres = np.argwhere((psnr_img > 4.0) & (psnr_img < 4.5)).flatten()
        # psnr_ranked_ind = np.flip(np.argsort(sorted_psnrs))[:max_num]
        sorted_psnrs = psnr_img[thres]
        sorted_recons = recon_img[thres]

        print('computing max distance matrix', sorted_recons[0].shape)
        max_pixel = 0
        max_inds = [0,0]
        pixel_loc = None
        for i in range(len(sorted_recons)):
            for j in range(i, len(sorted_recons)):
                pixel_val = np.max((sorted_recons[i]-sorted_recons[j])**2)
                if pixel_val > max_pixel:
                    max_pixel = pixel_val
                    max_inds[0] = i
                    max_inds[1] = j
                    pixel_loc = np.unravel_index(np.argmax((sorted_recons[i]-sorted_recons[j])**2), (256,256))
        print('done')
        print(max_pixel, pixel_loc)

        arrow = patches.Arrow(50, 33, -5, 5, width=5.0, color='r')
        myutils.plot.plot_img(mims(gt_img), ax=axes[2*img_idx,0], rot90=True)
        myutils.plot.plot_img(mims(gt_img[pixel_loc[0]-40:pixel_loc[0]+40, pixel_loc[1]-40:pixel_loc[1]+40, :]), ax=axes[2*img_idx+1,0], rot90=True)

        xy = (256-pixel_loc[0]-40, pixel_loc[1]-40)
        rect = patches.Rectangle(xy,80,80,linewidth=1,edgecolor='r',facecolor='none')
        axes[2*img_idx,0].add_patch(rect)
        # arrow = patches.Arrow(45, 35, 5, 5, width=5.0, color='r')
        axes[2*img_idx+1,0].add_patch(arrow)

        indices = max_inds
        for i, flat_ind in enumerate(indices):
            myutils.plot.plot_img(mims(sorted_recons[flat_ind]), ax=axes[2*img_idx,i+1], rot90=True)
            myutils.plot.plot_img(mims(sorted_recons[flat_ind, pixel_loc[0]-40:pixel_loc[0]+40, pixel_loc[1]-40:pixel_loc[1]+40]), ax=axes[2*img_idx+1,i+1], rot90=True)

            new_psnr = sorted_psnrs[flat_ind]

            arrow = patches.Arrow(50, 33, -5, 5, width=5.0, color='r')

            axes[2*img_idx+1,i+1].add_patch(arrow)
            axes[2*img_idx,i+1].text(0.7, 0.05,'RPSNR=%.02f' % (new_psnr), color='white', fontsize=20,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = axes[2*img_idx,i+1].transAxes)
            # Create a Rectangle patch
            xy = (256-pixel_loc[0]-40, pixel_loc[1]-40)
            rect = patches.Rectangle(xy,80,80,linewidth=1,edgecolor='r',facecolor='none')

            # Add the patch to the Axes
            axes[2*img_idx,i+1].add_patch(rect)

    fig.savefig('/nfs02/users/aw847/data/hypernet/figs/representative_slices_max.png', format='png')
    return fig


def plot_by_max_l2(gt_data, psnr_map, dc_map, recons):
    n_grid = int(np.sqrt(len(recons)))
    n_chunks = 2

    fig, axes = plt.subplots(psnr_map.shape[1]*2, n_chunks+1, figsize=(14,24))
    # fig.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    # fig.suptitle('Best DC, Coarse Sampling', fontsize=24)
    for img_idx in range(psnr_map.shape[1]):
        dc_img = dc_map[:,img_idx]
        psnr_img = psnr_map[:,img_idx]
        recon_img = recons[:,img_idx, ...]
        gt_img = gt_data[img_idx, ...]

        # # Take top 25% of recons, ranked by DC
        # max_num = int(len(recon_img)*0.45)
        # # max_num = int(len(recon_img))
        # dc_ranked_ind = np.argsort(dc_img)[:max_num]
        # sorted_psnrs = psnr_img[dc_ranked_ind]
        # sorted_recons = recon_img[dc_ranked_ind]

        # Take additional top 50% of recons, ranked by PSNR 
        thres = np.argwhere((psnr_img > 2.0) & (psnr_img < 2.5)).flatten()
        # psnr_ranked_ind = np.flip(np.argsort(sorted_psnrs))[:max_num]
        sorted_psnrs = psnr_img[thres]
        sorted_recons = recon_img[thres]


        if img_idx == 0:
            arrow = patches.Arrow(15, 55, 5, 5, width=5.0, color='r')
        if img_idx == 1:
            arrow = patches.Arrow(56, 33, -5, 5, width=5.0, color='r')
        if img_idx == 2:
            arrow = patches.Arrow(40, 55, 5, 5, width=5.0, color='r')
        myutils.plot.plot_img(mims(gt_img), ax=axes[2*img_idx,0], rot90=True)
        myutils.plot.plot_img(mims(gt_img[100:175, 100:175, :]), ax=axes[2*img_idx+1,0], rot90=True)
        rect = patches.Rectangle((85,100),75,75,linewidth=1,edgecolor='r',facecolor='none')
        axes[2*img_idx,0].add_patch(rect)
        # arrow = patches.Arrow(45, 35, 5, 5, width=5.0, color='r')
        # axes[2*img_idx+1,0].add_patch(arrow)

        indices = get_indices_l2_dist(sorted_psnrs, sorted_recons)
        for i, flat_ind in enumerate(indices):
            myutils.plot.plot_img(mims(sorted_recons[flat_ind]), ax=axes[2*img_idx,i+1], rot90=True)
            myutils.plot.plot_img(mims(sorted_recons[flat_ind, 100:175, 100:175, :]), ax=axes[2*img_idx+1,i+1], rot90=True)

            new_psnr = sorted_psnrs[flat_ind]

            if img_idx == 0:
                arrow = patches.Arrow(15, 55, 5, 5, width=5.0, color='r')
            if img_idx == 1:
                arrow = patches.Arrow(56, 33, -5, 5, width=5.0, color='r')
            if img_idx == 2:
                arrow = patches.Arrow(40, 55, 5, 5, width=5.0, color='r')

            # axes[2*img_idx+1,i+1].add_patch(arrow)
            axes[2*img_idx,i+1].text(0.7, 0.05,'RPSNR=%.02f' % (new_psnr), color='white', fontsize=20,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = axes[2*img_idx,i+1].transAxes)
            # Create a Rectangle patch
            rect = patches.Rectangle((85,100),75,75,linewidth=1,edgecolor='r',facecolor='none')

            # Add the patch to the Axes
            axes[2*img_idx,i+1].add_patch(rect)

    fig.savefig('/nfs02/users/aw847/data/hypernet/figs/representative_slices_l2.eps', format='eps')
    return fig


def get_indices_l2_dist(sorted_psnrs, sorted_recons):
    recons_reshaped = sorted_recons.reshape(len(sorted_recons), -1)
    dist_dim = len(sorted_recons)
    # Create distance matrix for all recons
    dist_mat = squareform(pdist(recons_reshaped, 'euclidean'))
    # Find row with smallest summed distance
    two_inds = np.unravel_index(np.argmax(dist_mat), (dist_dim, dist_dim))
    # dist_mat[two_inds[0]
    return np.unravel_index(np.argmax(dist_mat), (dist_dim, dist_dim))
