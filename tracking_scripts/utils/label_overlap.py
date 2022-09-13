from skimage.measure import regionprops_table
import pandas as pd
import numpy as np

class LabelOverlap:
    def _intersect_bbox(self,r1,r2):
        y0 = max(r1["bbox-0"], r2["bbox-0"])
        y1 = min(r1["bbox-2"], r2["bbox-2"])
        x0 = max(r1["bbox-1"], r2["bbox-1"])
        x1 = min(r1["bbox-3"], r2["bbox-3"])
        if y0 <= y1 and x0 <= x1:
            return [(y0,y1),(x0,x1)]
        else :
            return None

    def _union_bbox(self,r1,r2):
        y0 = min(r1["bbox-0"], r2["bbox-0"])
        y1 = max(r1["bbox-2"], r2["bbox-2"])
        x0 = min(r1["bbox-1"], r2["bbox-1"])
        x1 = max(r1["bbox-3"], r2["bbox-3"])
        return [(y0,y1),(x0,x1)]

    def __init__(self, label_images):
        self.label_images = label_images
        dfs=[]
        for frame in range(label_images.shape[0]):
            df=pd.DataFrame(
                regionprops_table(label_images[frame],
                properties=["label","bbox"]))
            df["frame"]=frame
            dfs.append(df)
        self.regionprops_df = pd.concat(dfs).set_index(["frame","label"])
    def calc_overlap(self,frame1,label1,frame2,label2):
        """
        returns overlap_pixels, IoU 
        """
        r1=self.regionprops_df.loc[(frame1,label1)]
        r2=self.regionprops_df.loc[(frame2,label2)]
        bbox = self._intersect_bbox(r1,r2)
        if bbox is None:
            return 0, 0.0, 0.0, 0.0
        else:
            u_bbox = self._union_bbox(r1,r2)
            window = tuple([slice(r[0],r[1]+1) for r in u_bbox])
            b1 = self.label_images[frame1][window] == label1
            b2 = self.label_images[frame2][window] == label2
            overlap = np.sum(b1&b2)
            union = np.sum(b1|b2)
            
            return overlap, overlap/union, overlap/np.sum(b1), overlap/np.sum(b2)